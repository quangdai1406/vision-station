"""Candidate thresholds (t_low, t_img) from labelled model scores.

Standard library only. Input CSV: columns label (good|bad), score (0.0..1.0,
higher = more defective), optional defect_class and part_id.

Decision rule (station/contracts.py):
    score >= t_img            -> BAD
    t_low <= score < t_img    -> REVIEW
    otherwise                 -> GOOD

Every number here is computed from the labelled samples given. Nothing is an
approved threshold: Quality approves the values and confirms the sample counts.

CLI:
    python -m tools.thresholds scores.csv --min-good N --min-bad N \
        --review-from-good-percentile P [--model-version V] [--out report.md] [--json out.json] \
        [--evaluate-t-img VALUE]

--evaluate-t-img evaluates a threshold Quality has chosen (recall, escapes and false rejects
with Wilson intervals). It does not recommend a value.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

NOTE = (
    "Candidate values from labelled samples only. They are not approved thresholds; "
    "Quality must approve them and confirm the sample counts are adequate."
)

REQUIRED_COLUMNS = ("label", "score")
LABELS = ("good", "bad")
UNSPECIFIED_CLASS = "(unspecified)"


class ThresholdInputError(ValueError):
    """The scores file or an argument is unusable."""


# ---------------------------------------------------------------- input

def load_scores(path) -> list[dict]:
    """Read the labelled scores CSV.

    Returns one dict per row: {"label": "good"|"bad", "score": float,
    "defect_class": str|None, "part_id": str|None}.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ThresholdInputError(f"cannot read {p}: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ThresholdInputError(f"{p}: empty file, no header row")

    columns: dict[str, str] = {}
    for name in reader.fieldnames:
        key = (name or "").strip().lower()
        if key in columns:
            raise ThresholdInputError(f"{p}: duplicate column {key!r}")
        columns[key] = name

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ThresholdInputError(f"{p}: missing required column(s): {', '.join(missing)}")

    rows: list[dict] = []
    for line_no, rec in enumerate(reader, start=2):

        def cell(col: str) -> str:
            if col not in columns:
                return ""
            value = rec.get(columns[col])
            return "" if value is None else value.strip()

        raw_label = cell("label")
        label = raw_label.lower()
        if label not in LABELS:
            raise ThresholdInputError(
                f"{p}: line {line_no}: label must be 'good' or 'bad', got {raw_label!r}"
            )

        raw_score = cell("score")
        try:
            score = float(raw_score)
        except ValueError:
            raise ThresholdInputError(
                f"{p}: line {line_no}: score is not a number: {raw_score!r}"
            ) from None
        if math.isnan(score):
            raise ThresholdInputError(f"{p}: line {line_no}: score is NaN")
        if not 0.0 <= score <= 1.0:
            raise ThresholdInputError(
                f"{p}: line {line_no}: score {score} is outside 0..1"
            )

        rows.append(
            {
                "label": label,
                "score": score,
                "defect_class": cell("defect_class") or None,
                "part_id": cell("part_id") or None,
            }
        )
    return rows


# ---------------------------------------------------------- statistics

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n. n == 0 gives (0.0, 1.0)."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three_upper(n: int) -> float:
    """95% upper bound on a rate after 0 events in n trials: 3/n.

    Capped at 1.0 (a rate cannot exceed 1); n <= 0 returns 1.0.
    """
    if n <= 0:
        return 1.0
    return min(1.0, 3.0 / n)


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile (0..100) of a non-empty sequence."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be 0..100, got {percentile}")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered) / 100.0)
    rank = min(len(ordered), max(1, rank))
    return ordered[rank - 1]


def t_low_candidate(good_scores: Sequence[float], percentile: float, t_img: float) -> tuple[float, bool]:
    """Percentile of good scores, never above t_img.

    Returns (value, capped). capped is True when the percentile was above t_img
    and the value was set equal to t_img.
    """
    raw = nearest_rank_percentile(good_scores, percentile)
    if raw > t_img:
        return (t_img, True)
    return (raw, False)


# ------------------------------------------------------- candidate table

def candidate_table(good: Sequence[float], bad: Sequence[float]) -> list[dict]:
    """One row per candidate threshold, sorted ascending.

    Candidates: 0.0, every observed distinct score, and a value just above the
    maximum (the last row, where nothing is BAD). Escape: bad with score <
    threshold. False reject: good with score >= threshold.
    """
    good_sorted = sorted(float(s) for s in good)
    bad_sorted = sorted(float(s) for s in bad)
    n_good, n_bad = len(good_sorted), len(bad_sorted)

    observed = set(good_sorted) | set(bad_sorted) | {0.0}
    top = math.nextafter(max(observed), math.inf)
    candidates = sorted(observed) + [top]

    table: list[dict] = []
    for t in candidates:
        escapes = _count_below(bad_sorted, t)
        false_rejects = n_good - _count_below(good_sorted, t)
        esc_lo, esc_hi = wilson_interval(escapes, n_bad)
        fr_lo, fr_hi = wilson_interval(false_rejects, n_good)
        table.append(
            {
                "threshold": t,
                "escapes": escapes,
                "escape_rate": escapes / n_bad if n_bad else 0.0,
                "escape_ci_low": esc_lo,
                "escape_ci_high": esc_hi,
                "false_rejects": false_rejects,
                "false_reject_rate": false_rejects / n_good if n_good else 0.0,
                "false_reject_ci_low": fr_lo,
                "false_reject_ci_high": fr_hi,
            }
        )
    return table


def _count_below(sorted_values: Sequence[float], t: float) -> int:
    """Number of values strictly below t in an ascending list."""
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _window(table: list[dict], lo: float, hi: float) -> list[dict]:
    """Table rows with lo <= threshold <= hi, plus one neighbour on each side."""
    if not table:
        return []
    thresholds = [r["threshold"] for r in table]
    i0 = next((i for i, t in enumerate(thresholds) if t >= lo), len(thresholds))
    i1 = max((i for i, t in enumerate(thresholds) if t <= hi), default=-1)
    start = max(0, i0 - 1)
    end = min(len(thresholds) - 1, i1 + 1)
    return table[start : end + 1]


def per_class_counts(bad_rows: Sequence[dict]) -> list[dict]:
    """Per defect class: n bad parts and the rule-of-three bound on the miss rate.

    Recall is not computed here: at t_img = min(bad) it is 100% by construction.
    Rows without a class are grouped as '(unspecified)'. Sorted by class name.
    """
    counts: dict[str, int] = {}
    for r in bad_rows:
        key = r.get("defect_class") or UNSPECIFIED_CLASS
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "defect_class": name,
            "n_bad": counts[name],
            "miss_rule_of_three_upper": rule_of_three_upper(counts[name]),
        }
        for name in sorted(counts)
    ]


def evaluate_threshold(good_scores: Sequence[float], bad_rows: Sequence[dict], t_img: float) -> dict:
    """Recall, escapes and false rejects at a given t_img, with Wilson 95% intervals.

    Escape: bad with score < t_img. False reject: good with score >= t_img.
    Per-class figures are included only when the bad rows carry defect_class values.
    """
    g = [float(s) for s in good_scores]
    n_good = len(g)
    false_rejects = sum(1 for s in g if s >= t_img)
    n_bad = len(bad_rows)
    caught = sum(1 for r in bad_rows if float(r["score"]) >= t_img)
    escapes = n_bad - caught
    fr_lo, fr_hi = wilson_interval(false_rejects, n_good)
    esc_lo, esc_hi = wilson_interval(escapes, n_bad)
    rec_lo, rec_hi = wilson_interval(caught, n_bad)

    per_class = None
    if any(r.get("defect_class") for r in bad_rows):
        per_class = []
        for c in per_class_recall(bad_rows, t_img):
            lo, hi = wilson_interval(c["caught"], c["n_bad"])
            per_class.append(
                {
                    "defect_class": c["defect_class"],
                    "n_bad": c["n_bad"],
                    "caught": c["caught"],
                    "escapes": c["n_bad"] - c["caught"],
                    "recall": c["recall"],
                    "recall_ci_low": lo,
                    "recall_ci_high": hi,
                }
            )
    return {
        "t_img": float(t_img),
        "good": {
            "n": n_good,
            "false_rejects": false_rejects,
            "false_reject_rate": false_rejects / n_good if n_good else 0.0,
            "false_reject_ci_low": fr_lo,
            "false_reject_ci_high": fr_hi,
        },
        "bad": {
            "n": n_bad,
            "caught": caught,
            "escapes": escapes,
            "escape_rate": escapes / n_bad if n_bad else 0.0,
            "escape_ci_low": esc_lo,
            "escape_ci_high": esc_hi,
            "recall": caught / n_bad if n_bad else 0.0,
            "recall_ci_low": rec_lo,
            "recall_ci_high": rec_hi,
        },
        "per_class": per_class,
    }


def per_class_recall(bad_rows: Sequence[dict], t_img: float) -> list[dict]:
    """Per defect class: n bad parts, how many score >= t_img, recall, and the
    rule-of-three bound on the miss rate. Rows without a class are grouped as
    '(unspecified)'. Sorted by class name."""
    groups: dict[str, list[float]] = {}
    for r in bad_rows:
        key = r.get("defect_class") or UNSPECIFIED_CLASS
        groups.setdefault(key, []).append(float(r["score"]))

    out: list[dict] = []
    for name in sorted(groups):
        scores = groups[name]
        n = len(scores)
        caught = sum(1 for s in scores if s >= t_img)
        out.append(
            {
                "defect_class": name,
                "n_bad": n,
                "caught": caught,
                "recall": caught / n,
                "miss_rule_of_three_upper": rule_of_three_upper(n),
            }
        )
    return out


def _verdict_counts(scores: Sequence[float], t_low: float, t_img: float) -> dict:
    counts = {"GOOD": 0, "REVIEW": 0, "BAD": 0}
    for s in scores:
        if s >= t_img:
            counts["BAD"] += 1
        elif s >= t_low:
            counts["REVIEW"] += 1
        else:
            counts["GOOD"] += 1
    return counts


# ------------------------------------------------------------ suggest

def _attach_evaluation(result: dict, good_scores, bad_rows, t_img) -> dict:
    if t_img is not None:
        result["evaluation"] = evaluate_threshold(good_scores, bad_rows, t_img)
    return result


def suggest(good, bad, *, min_good, min_bad, review_from_good_percentile, t_img=None) -> dict:
    """Candidate thresholds from labelled rows.

    good and bad are lists of row dicts as returned by load_scores().
    min_good, min_bad and review_from_good_percentile are required keyword
    arguments: they are Quality's decisions and have no defaults here.
    t_img (optional, float 0..1) is a threshold Quality has chosen. When given,
    the result also carries 'evaluation': recall, escapes and false rejects at that
    value. It is an evaluation, not a recommendation. Refused results have no evaluation.
    """
    if not isinstance(min_good, int) or not isinstance(min_bad, int) or min_good < 0 or min_bad < 0:
        raise ValueError("min_good and min_bad must be non-negative integers")
    if not 0.0 <= review_from_good_percentile <= 100.0:
        raise ValueError("review_from_good_percentile must be 0..100")
    chosen = None  # Quality's t_img to evaluate; kept apart from the candidate t_img below.
    if t_img is not None:
        if isinstance(t_img, bool) or not isinstance(t_img, (int, float)) or not 0.0 <= float(t_img) <= 1.0:
            raise ValueError(f"t_img must be a number 0..1, got {t_img!r}")
        chosen = float(t_img)

    g_scores = [float(r["score"]) for r in good]
    b_scores = [float(r["score"]) for r in bad]
    n_good, n_bad = len(g_scores), len(b_scores)

    base = {
        "note": NOTE,
        "counts": {"good": n_good, "bad": n_bad},
        "min_good": min_good,
        "min_bad": min_bad,
        "review_from_good_percentile": review_from_good_percentile,
    }

    if n_good < min_good or n_bad < min_bad:
        return {
            **base,
            "ok": False,
            "reason": (
                f"Not enough labelled samples: good {n_good} (minimum {min_good}), "
                f"bad {n_bad} (minimum {min_bad}). No candidate values are given."
            ),
        }
    if n_good == 0 or n_bad == 0:
        return {
            **base,
            "ok": False,
            "reason": "Need at least one good and one bad sample. No candidate values are given.",
        }

    table = candidate_table(g_scores, b_scores)
    lowest_bad = min(b_scores)
    highest_good = max(g_scores)
    separable = highest_good < lowest_bad

    if not separable:
        overlap_good = sum(1 for s in g_scores if lowest_bad <= s <= highest_good)
        overlap_bad = sum(1 for s in b_scores if lowest_bad <= s <= highest_good)
        result = {
            **base,
            "ok": True,
            "separable": False,
            "overlap": {
                "low": lowest_bad,
                "high": highest_good,
                "good": overlap_good,
                "bad": overlap_bad,
                "total": overlap_good + overlap_bad,
            },
            "t_img": None,
            "t_low": None,
            "per_class": None,
            "table": _window(table, lowest_bad, highest_good),
            "message": (
                "Good and bad scores overlap, so no single t_img separates them. "
                "No threshold is recommended. Quality chooses a trade-off from the table; "
                "the row at the lowest bad score has zero escapes but rejects good parts."
            ),
        }
        return _attach_evaluation(result, g_scores, bad, chosen)

    # Separable: t_img is the largest value that still catches every labelled bad part.
    t_img = lowest_bad
    t_low, capped = t_low_candidate(g_scores, review_from_good_percentile, t_img)
    at = next(r for r in table if r["threshold"] == t_img)
    verdicts = {
        "good_parts": _verdict_counts(g_scores, t_low, t_img),
        "bad_parts": _verdict_counts(b_scores, t_low, t_img),
    }
    result = {
        **base,
        "ok": True,
        "separable": True,
        "t_img": t_img,
        "t_low": t_low,
        "t_low_capped": capped,
        "at_t_img": {
            "escapes": at["escapes"],
            "escape_rate": at["escape_rate"],
            "escape_ci_low": at["escape_ci_low"],
            "escape_ci_high": at["escape_ci_high"],
            "false_rejects": at["false_rejects"],
            "false_reject_rate": at["false_reject_rate"],
            "false_reject_ci_low": at["false_reject_ci_low"],
            "false_reject_ci_high": at["false_reject_ci_high"],
        },
        "rule_of_three_escape_upper": rule_of_three_upper(n_bad) if at["escapes"] == 0 else None,
        "verdict_counts": verdicts,
        "per_class": per_class_counts(bad) if any(r.get("defect_class") for r in bad) else None,
        "table": _window(table, t_low, t_img),
        "message": (
            "Scores separate good from bad in this sample. t_img is the lowest bad score, "
            "so the sample has zero escapes at that value. Whether the sample is large enough "
            "is for Quality to decide."
        ),
    }
    return _attach_evaluation(result, g_scores, bad, chosen)


# ------------------------------------------------------------- report

LIMITS = [
    "The samples may not represent future production (other lots, operators, lighting, fixtures or part mix).",
    "Defect classes that never appear in the samples are not covered at all.",
    "Scores depend on the model version. Record it (see the header) and re-validate if it changes.",
    "Thresholds must be re-validated after any change to the model, cameras, lighting, fixtures or the scoring code.",
    "Wilson intervals assume independent parts. Rule-of-three bounds assume a fixed sample drawn the same way as production.",
    "Zero escapes in a small sample says little about the true escape rate; the rule-of-three bound shows how little.",
]


def _fmt_rate(x: float) -> str:
    return f"{x:.4f}"


def _fmt_thr(x: float) -> str:
    return repr(float(x))


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _candidate_rows(rows: list[dict]) -> str:
    return _md_table(
        ["threshold", "escapes", "escape rate", "escape 95% CI", "false rejects", "false reject rate", "false reject 95% CI"],
        [
            [
                _fmt_thr(r["threshold"]),
                str(r["escapes"]),
                _fmt_rate(r["escape_rate"]),
                f"{_fmt_rate(r['escape_ci_low'])} to {_fmt_rate(r['escape_ci_high'])}",
                str(r["false_rejects"]),
                _fmt_rate(r["false_reject_rate"]),
                f"{_fmt_rate(r['false_reject_ci_low'])} to {_fmt_rate(r['false_reject_ci_high'])}",
            ]
            for r in rows
        ],
    )


PER_CLASS_COUNTS_LABEL = (
    "Per-class sample counts and rule-of-three bounds (recall is 100% at t_img = min(bad) by "
    "construction, so it is not evidence of detection quality on new parts)"
)


def _evaluation_lines(ev: dict) -> list[str]:
    """Markdown for evaluate_threshold() output: a chosen threshold, not a recommendation."""
    b, g = ev["bad"], ev["good"]
    lines = [
        "## Evaluation of a chosen threshold",
        "",
        f"Evaluation of t_img = `{_fmt_thr(ev['t_img'])}` as chosen by Quality. "
        "This is not a recommended value; the numbers describe this sample only.",
        "",
        f"- Recall: {b['caught']} of {b['n']} bad caught "
        f"(rate {_fmt_rate(b['recall'])}, 95% CI {_fmt_rate(b['recall_ci_low'])} to {_fmt_rate(b['recall_ci_high'])})",
        f"- Escapes: {b['escapes']} of {b['n']} bad "
        f"(rate {_fmt_rate(b['escape_rate'])}, 95% CI {_fmt_rate(b['escape_ci_low'])} to {_fmt_rate(b['escape_ci_high'])})",
        f"- False rejects: {g['false_rejects']} of {g['n']} good "
        f"(rate {_fmt_rate(g['false_reject_rate'])}, 95% CI {_fmt_rate(g['false_reject_ci_low'])} to {_fmt_rate(g['false_reject_ci_high'])})",
        "",
    ]
    if ev["per_class"] is None:
        lines += ["No defect_class column values in the bad samples; per-class recall not computed.", ""]
    else:
        lines += [
            "### Per defect class at the chosen t_img",
            "",
            _md_table(
                ["defect class", "n bad", "caught", "escapes", "recall", "recall 95% CI"],
                [
                    [
                        c["defect_class"],
                        str(c["n_bad"]),
                        str(c["caught"]),
                        str(c["escapes"]),
                        _fmt_rate(c["recall"]),
                        f"{_fmt_rate(c['recall_ci_low'])} to {_fmt_rate(c['recall_ci_high'])}",
                    ]
                    for c in ev["per_class"]
                ],
            ),
            "",
        ]
    return lines


def render_report(result: dict, *, source, model_version: Optional[str], n_rows: int) -> str:
    """Markdown report for one suggest() result."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts = result["counts"]
    lines = [
        "# Threshold candidates",
        "",
        f"> {NOTE}",
        "",
        f"- Source: `{source}` ({n_rows} rows read)",
        f"- Generated (UTC): {generated}",
        f"- Model version: {model_version if model_version else 'NOT RECORDED (pass --model-version)'}",
        "",
        "## Assumptions",
        "",
        "- Score is 0.0 to 1.0, higher = more defective.",
        "- BAD if score >= t_img; REVIEW if t_low <= score < t_img; GOOD otherwise.",
        "- Candidate values come only from the scores in this file.",
        "- Escape: a bad part with score < threshold (passed as not BAD). "
        "False reject: a good part with score >= threshold.",
        f"- Minimum good samples: {result['min_good']}. Minimum bad samples: {result['min_bad']}.",
        f"- REVIEW band starts at the {result['review_from_good_percentile']:g}th percentile of good scores (nearest rank).",
        "- Intervals are Wilson 95%. Zero-event upper bounds use the rule of three (3/n).",
        "",
        "## Counts",
        "",
        f"- Good parts: {counts['good']}",
        f"- Bad parts: {counts['bad']}",
        "",
    ]

    if not result["ok"]:
        lines += ["## Result", "", f"Refused: {result['reason']}", ""]
    elif not result["separable"]:
        ov = result["overlap"]
        lines += [
            "## Result: scores overlap (not separable)",
            "",
            f"- Overlap range: {_fmt_thr(ov['low'])} to {_fmt_thr(ov['high'])}",
            f"- Parts inside the overlap range: {ov['total']} ({ov['good']} good, {ov['bad']} bad)",
            "",
            result["message"],
            "",
            "### Trade-off table around the overlap",
            "",
            _candidate_rows(result["table"]),
            "",
        ]
    else:
        at = result["at_t_img"]
        rule3 = result["rule_of_three_escape_upper"]
        cap_note = " (capped at t_img)" if result["t_low_capped"] else ""
        lines += [
            "## Candidate values",
            "",
            f"- t_img (largest value that still catches every labelled bad part): `{_fmt_thr(result['t_img'])}`",
            f"- t_low (good-score percentile{cap_note}): `{_fmt_thr(result['t_low'])}`",
            "",
            "At t_img:",
            "",
            f"- Escapes: {at['escapes']} of {counts['bad']} bad "
            f"(rate {_fmt_rate(at['escape_rate'])}, 95% CI {_fmt_rate(at['escape_ci_low'])} to {_fmt_rate(at['escape_ci_high'])})",
            f"- False rejects: {at['false_rejects']} of {counts['good']} good "
            f"(rate {_fmt_rate(at['false_reject_rate'])}, 95% CI {_fmt_rate(at['false_reject_ci_low'])} to {_fmt_rate(at['false_reject_ci_high'])})",
        ]
        if rule3 is not None:
            lines.append(f"- Rule of three: with 0 escapes in {counts['bad']} bad parts, the escape rate is below {_fmt_rate(rule3)} (95%).")
        lines += ["", result["message"], "", "### Part counts for the candidate pair", ""]
        vc = result["verdict_counts"]
        lines.append(
            _md_table(
                ["parts", "GOOD", "REVIEW", "BAD"],
                [
                    ["good (label)", str(vc["good_parts"]["GOOD"]), str(vc["good_parts"]["REVIEW"]), str(vc["good_parts"]["BAD"])],
                    ["bad (label)", str(vc["bad_parts"]["GOOD"]), str(vc["bad_parts"]["REVIEW"]), str(vc["bad_parts"]["BAD"])],
                ],
            )
        )
        lines += ["", f"### {PER_CLASS_COUNTS_LABEL}", ""]
        if result["per_class"] is None:
            lines.append("No defect_class column values in the bad samples; per-class numbers not computed.")
        else:
            lines.append(
                _md_table(
                    ["defect class", "n bad", "miss rate bound (rule of three)"],
                    [
                        [
                            c["defect_class"],
                            str(c["n_bad"]),
                            _fmt_rate(c["miss_rule_of_three_upper"]),
                        ]
                        for c in result["per_class"]
                    ],
                )
            )
        lines += ["", "### Table around the candidates", "", _candidate_rows(result["table"]), ""]

    if result.get("evaluation"):
        lines += _evaluation_lines(result["evaluation"])

    lines += ["## What these numbers do NOT show", ""]
    lines += [f"- {item}" for item in LIMITS]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.thresholds",
        description="Candidate t_low / t_img from labelled scores. Not approved thresholds.",
    )
    parser.add_argument("scores", help="CSV with columns label, score (optional defect_class, part_id)")
    parser.add_argument("--min-good", type=int, required=True, help="minimum good samples (Quality decides)")
    parser.add_argument("--min-bad", type=int, required=True, help="minimum bad samples (Quality decides)")
    parser.add_argument(
        "--review-from-good-percentile",
        type=float,
        required=True,
        help="percentile (0..100) of good scores that starts the REVIEW band (Quality decides)",
    )
    parser.add_argument("--model-version", default=None, help="model version the scores came from (recorded in the report)")
    parser.add_argument("--out", default=None, help="write the Markdown report to this file")
    parser.add_argument("--json", dest="json_out", default=None, help="write the result dict as JSON to this file")
    parser.add_argument(
        "--evaluate-t-img",
        type=float,
        default=None,
        metavar="VALUE",
        help="also evaluate this threshold (0..1, chosen by Quality): recall, escapes and false rejects "
             "with 95%% Wilson intervals. Not a recommendation.",
    )
    args = parser.parse_args(argv)

    try:
        rows = load_scores(args.scores)
        good = [r for r in rows if r["label"] == "good"]
        bad = [r for r in rows if r["label"] == "bad"]
        result = suggest(
            good,
            bad,
            min_good=args.min_good,
            min_bad=args.min_bad,
            review_from_good_percentile=args.review_from_good_percentile,
            t_img=args.evaluate_t_img,
        )
        report = render_report(result, source=args.scores, model_version=args.model_version, n_rows=len(rows))
        if args.out:
            Path(args.out).write_text(report, encoding="utf-8")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
