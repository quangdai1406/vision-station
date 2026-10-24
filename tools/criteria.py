"""MES inspection criteria for a part -> station config pieces and a sample-collection checklist.

Standard library only. The MES token is read from an environment variable, sent only as
the Authorization header to the MES, and never printed or written to a file.

Proxies: by default the MES is contacted directly and environment proxy settings
(HTTP_PROXY, http_proxy, ...) are ignored. This is the safe default for a local or LAN MES:
the bearer token goes straight to the MES and no proxy sees it. Use --use-env-proxy only when
the MES is reachable only through a corporate proxy. A bearer token sent over plain http
through a proxy is visible to that proxy, so prefer an https MES URL whenever a proxy is used.

Usage:
  python -m tools.criteria --base-url URL --token-env NAME --part PART [--out-dir DIR] [--use-env-proxy]
  python -m tools.criteria --from-json FILE --part PART [--out-dir DIR]
"""

import argparse
import csv
import http.client
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

PLACEHOLDER = "[Quality to define]"
CHECK_TYPES = ("visual", "measurement")
CHECKLIST_FIELDS = [
    "criterion_id",
    "defect_class",
    "criteria_name",
    "criteria_description",
    "check_type",
    "standard_picture_path",
    "good_samples_needed",
    "bad_samples_needed",
    "collected_good",
    "collected_bad",
]
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {
    f"lpt{i}" for i in range(1, 10)
}
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CriteriaError(Exception):
    """Criteria data is missing or malformed."""


class CriteriaFetchError(CriteriaError):
    """The MES could not supply the criteria. The message never contains the token."""


class CriteriaFileError(CriteriaError):
    """The offline criteria JSON file is missing, unreadable or malformed."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(data, source):
    """Check a criteria list and return normalised copies of its entries."""
    if not isinstance(data, list):
        raise CriteriaError(f"{source}: expected a JSON list of criteria, got {type(data).__name__}")
    out = []
    seen = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise CriteriaError(f"{source}: entry {i} is not an object")
        cid = item.get("id")
        if isinstance(cid, bool) or not isinstance(cid, (int, str)) or not str(cid).strip():
            raise CriteriaError(f"{source}: entry {i} has no usable 'id'")
        key = str(cid).strip()
        where = f"{source}: criterion {key}"
        if key in seen:
            raise CriteriaError(f"{where} appears more than once")
        seen.add(key)
        name = item.get("criteria_name")
        if not isinstance(name, str) or not name.strip():
            raise CriteriaError(f"{where} has no 'criteria_name'")
        ctype = item.get("check_type")
        if not isinstance(ctype, str) or ctype not in CHECK_TYPES:
            raise CriteriaError(f"{where} has check_type {ctype!r}; expected 'visual' or 'measurement'")
        active = item.get("active")
        if isinstance(active, int) and not isinstance(active, bool) and active in (0, 1):
            active = bool(active)
        if not isinstance(active, bool):
            raise CriteriaError(f"{where} has no boolean 'active'")
        entry = dict(item)
        for optional in (
            "scope_type",
            "scope_value",
            "measurement_uom",
            "measurement_target",
            "measurement_tolerance",
            "standard_picture_path",
        ):
            entry.setdefault(optional, None)
        entry["criteria_description"] = item.get("criteria_description") or ""
        entry["id"] = cid
        entry["criteria_name"] = name
        entry["check_type"] = ctype
        entry["active"] = active
        out.append(entry)
    return out


def _key(c):
    return str(c["id"]).strip()


def _safe_id(key):
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") or "x"


# ---------------------------------------------------------------------------
# Fetch and load
# ---------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: the token must not be sent to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _scrub(text, token):
    return text.replace(token, "<token>") if token else text


def _http_message(code, part):
    if code == 401:
        return ("MES rejected the token (401 Unauthorized). Check the token in the environment "
                "variable and that the machine token has not been revoked.")
    if code == 403:
        return (f"MES refused access to the criteria for part {part} (403 Forbidden). "
                "The station user may lack permission.")
    if code == 404:
        return f"MES has no criteria for part {part} (404 Not Found). Check the part number."
    if 300 <= code < 400:
        return (f"MES answered with a redirect ({code}); redirects are not followed. "
                "Check the MES base URL.")
    if code >= 500:
        return (f"MES server error ({code}) while fetching criteria for part {part}. "
                "Try again later.")
    return f"MES answered {code} while fetching criteria for part {part}."


def fetch_criteria(base_url, token, part_number, timeout_s=10.0, use_env_proxy=False):
    """GET {base_url}/api/machine/criteria/<part> with a bearer token and return the normalised list.

    use_env_proxy=False (default): environment proxies are bypassed; the token goes straight to the MES.
    use_env_proxy=True: the normal environment/system proxy settings are used. Only needed when the
    MES is reachable only through a corporate proxy; a token sent over plain http is visible to it.
    """
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise CriteriaFetchError("MES base URL must start with http:// or https://")
    if not token:
        raise CriteriaFetchError("no MES token supplied")
    part = str(part_number).strip()
    if not part:
        raise CriteriaFetchError("part number is empty")

    url = base_url.rstrip("/") + "/api/machine/criteria/" + urllib.parse.quote(part, safe="")
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    if use_env_proxy:
        # Without an explicit ProxyHandler, build_opener adds one that reads the environment.
        opener = urllib.request.build_opener(_NoRedirect)
    else:
        # No environment proxy: the bearer token must only go straight to the MES.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        raise CriteriaFetchError(_scrub(_http_message(exc.code, part), token)) from None
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", None) or type(exc).__name__
        raise CriteriaFetchError(
            _scrub(f"could not reach MES at {base_url}: {reason}", token)
        ) from None
    except ValueError:
        raise CriteriaFetchError("invalid MES request (check the base URL and token)") from None

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise CriteriaFetchError(f"MES returned something other than JSON for part {part}") from None
    try:
        return _validate(data, f"MES response for part {part}")
    except CriteriaError as exc:
        raise CriteriaFetchError(_scrub(str(exc), token)) from None


def load_criteria_json(path):
    """Load a criteria list saved from the MES (same format as the API) for offline use."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CriteriaFileError(f"criteria file not found: {p}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise CriteriaFileError(f"cannot read criteria file {p}: {exc}") from None
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise CriteriaFileError(f"{p} is not valid JSON: {exc}") from None
    try:
        return _validate(data, str(p))
    except CriteriaError as exc:
        raise CriteriaFileError(str(exc)) from None


# ---------------------------------------------------------------------------
# Naming and splitting
# ---------------------------------------------------------------------------

def slugify(name):
    """Lowercase ASCII snake_case from a criteria name. Raises ValueError if nothing is left."""
    if not isinstance(name, str):
        raise ValueError("criteria name must be a string")
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", folded.lower()).strip("_")
    if not slug:
        raise ValueError(f"criteria name {name!r} has no letters or digits")
    return slug


def derive_class_names(criteria):
    """Map every criterion id to a defect-class name.

    Names that collide (within the given list) get an '_<id>' suffix on every member of the
    collision, so the mapping does not depend on list order.
    """
    criteria = _validate(criteria, "criteria")
    base = {}
    for c in criteria:
        try:
            base[_key(c)] = slugify(c["criteria_name"])
        except ValueError:
            base[_key(c)] = None
    counts = Counter(v for v in base.values() if v)
    names = {}
    for c in criteria:
        key = _key(c)
        slug = base[key]
        if slug is None:
            names[key] = f"criterion_{_safe_id(key)}"
        elif counts[slug] > 1:
            names[key] = f"{slug}_{_safe_id(key)}"
        else:
            names[key] = slug
    if len(set(names.values())) != len(names):
        raise CriteriaError("could not derive unique defect class names for these criteria")
    return names


def split_by_type(criteria):
    """Return (active visual, active measurement). Inactive criteria are in neither list."""
    criteria = _validate(criteria, "criteria")
    visual = [c for c in criteria if c["active"] and c["check_type"] == "visual"]
    measurement = [c for c in criteria if c["active"] and c["check_type"] == "measurement"]
    return visual, measurement


def summarize(criteria):
    """Counts, class names and warnings for the station owner and Quality."""
    criteria = _validate(criteria, "criteria")
    visual, measurement = split_by_type(criteria)
    names = derive_class_names(criteria)
    warnings = []
    for c in measurement:
        warnings.append(
            f"measurement criterion {_key(c)} '{c['criteria_name']}' has no camera defect class: "
            "the camera classifier does not check it, and the station would post it as 'pass' "
            "unless it is handled (manual check or a measurement method)"
        )
    if not visual:
        warnings.append("no active visual criteria: the model has no defect classes to output")
    warnings.append("default_criterion_id is left null: decision for Quality")
    return {
        "visual_count": len(visual),
        "measurement_count": len(measurement),
        "inactive_count": sum(1 for c in criteria if not c["active"]),
        "classes": [names[_key(c)] for c in visual],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def build_config_pieces(criteria):
    """Config values for config.json's mes section. Keys are derived from MES unless noted."""
    criteria = _validate(criteria, "criteria")
    names = derive_class_names(criteria)
    visual, _ = split_by_type(criteria)
    summary = summarize(criteria)
    return {
        # Derived from MES: the MES needs a result for every active criterion.
        "criteria_ids": [_key(c) for c in criteria if c["active"]],
        # Derived from MES: active visual criteria only.
        "class_to_criterion": {names[_key(c)]: _key(c) for c in visual},
        # Decision for Quality, not derived from MES.
        "default_criterion_id": None,
        "_notes": {
            "derived_from_mes": ["criteria_ids", "class_to_criterion"],
            "decision_for_quality": ["default_criterion_id"],
            "remove_before_config": "_notes is for people; delete it when pasting into config.json",
            "warnings": summary["warnings"],
        },
    }


def build_checklist_rows(criteria):
    """Sample-collection rows for every active criterion. Sample counts are never filled in."""
    criteria = _validate(criteria, "criteria")
    names = derive_class_names(criteria)
    rows = []
    for c in criteria:
        if not c["active"]:
            continue
        key = _key(c)
        rows.append({
            "criterion_id": key,
            "defect_class": names[key] if c["check_type"] == "visual" else "",
            "criteria_name": c["criteria_name"],
            "criteria_description": c["criteria_description"],
            "check_type": c["check_type"],
            "standard_picture_path": c.get("standard_picture_path") or "",
            "good_samples_needed": PLACEHOLDER,
            "bad_samples_needed": PLACEHOLDER,
            "collected_good": "",
            "collected_bad": "",
        })
    return rows


def safe_part_name(part):
    """A part number safe for file names on Windows and Linux."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(part).strip())
    s = s.strip("._ ")[:100].rstrip("._ ")
    if not s:
        s = "part"
    if s.split(".")[0].lower() in _WINDOWS_RESERVED:
        s = "part_" + s
    return s


def write_outputs(criteria, part, out_dir):
    """Write criteria_<part>.json, class_map.json and collection_checklist.csv into out_dir."""
    criteria = _validate(criteria, "criteria")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "criteria": out / f"criteria_{safe_part_name(part)}.json",
        "class_map": out / "class_map.json",
        "checklist": out / "collection_checklist.csv",
    }
    paths["criteria"].write_text(json.dumps(criteria, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["class_map"].write_text(
        json.dumps(build_config_pieces(criteria), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with paths["checklist"].open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CHECKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(build_checklist_rows(criteria))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_summary(part, source, summary, paths):
    lines = [
        f"Part: {part}   source: {source}",
        f"Active visual (camera defect classes): {summary['visual_count']}",
        f"Active measurement (NOT checked by the camera): {summary['measurement_count']}",
        f"Inactive (ignored): {summary['inactive_count']}",
        "Defect classes: " + (", ".join(summary["classes"]) or "(none)"),
    ]
    for w in summary["warnings"]:
        lines.append(f"WARNING: {w}")
    lines.append("Wrote:")
    lines.extend(f"  {p}" for p in paths.values())
    return "\n".join(lines)


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python -m tools.criteria",
        description="Turn MES inspection criteria for a part into station config pieces and a checklist.",
    )
    p.add_argument("--base-url", help="MES base URL, e.g. http://127.0.0.1:3000")
    p.add_argument("--token-env", metavar="NAME",
                   help="NAME of the environment variable holding the MES token (not the token itself)")
    p.add_argument("--from-json", metavar="FILE", help="criteria list saved from the MES, for offline use")
    p.add_argument("--part", required=True, help="part number")
    p.add_argument("--out-dir", default="criteria_out", help="output folder (default: criteria_out)")
    p.add_argument(
        "--use-env-proxy", action="store_true",
        help="use the environment/system proxy settings (HTTP_PROXY etc.) to reach the MES. "
             "Default: connect directly and ignore proxies, so the token never goes to a proxy. "
             "Use only when the MES is reachable only through a corporate proxy. A token sent over "
             "plain http through a proxy is visible to that proxy: prefer an https MES URL with a proxy.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.from_json:
        if args.base_url or args.token_env:
            print("error: use either --from-json or --base-url with --token-env, not both", file=sys.stderr)
            return 2
        if args.use_env_proxy:
            print("error: --use-env-proxy only applies with --base-url", file=sys.stderr)
            return 2
        source = args.from_json
    else:
        if not args.base_url or not args.token_env:
            print("error: give --base-url and --token-env (or --from-json FILE)", file=sys.stderr)
            return 2
        if not _ENV_NAME_RE.match(args.token_env):
            print("error: --token-env must be the NAME of an environment variable, not the token",
                  file=sys.stderr)
            return 2
        token = os.environ.get(args.token_env)
        if not token:
            print(f"error: environment variable {args.token_env} is not set or empty", file=sys.stderr)
            return 2
        source = args.base_url

    try:
        if args.from_json:
            criteria = load_criteria_json(args.from_json)
        else:
            criteria = fetch_criteria(args.base_url, token, args.part, use_env_proxy=args.use_env_proxy)
        paths = write_outputs(criteria, args.part, args.out_dir)
    except CriteriaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1

    print(_format_summary(args.part, source, summarize(criteria), paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
