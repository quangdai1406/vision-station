#!/usr/bin/env python3
"""Lightweight code graph for Claude Code: files, definitions, imports, references, tests.

Standard library only. Static and approximate: Python is parsed with `ast`,
JS/TS with regular expressions. Anything it cannot resolve is reported as external
or unknown, never guessed.

Usage: codegraph.py [--root DIR] [--rebuild] [--json] <command> ...
Commands: build | summary | impact PATH | deps PATH | symbol NAME | tests-for PATH | changed
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque

VERSION = 1
SKIP_DIRS = {
    "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build", "out",
    "target", "vendor", "_site", "site-packages", "coverage", "bower_components",
}
PY_EXT = {".py"}
JS_EXT = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
MAX_FILE_BYTES = 1_000_000
AMBIGUOUS_LIMIT = 3
COMMON_NAMES = {
    "main", "init", "run", "get", "set", "test", "setup", "teardown", "handler", "index",
    "app", "name", "value", "data", "type", "id", "args", "kwargs", "self", "cls", "call",
    "close", "open", "read", "write", "start", "stop", "update", "create", "delete", "default",
    "constructor", "render", "build", "load", "save", "parse", "format", "log", "next", "key",
}
TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs"}


# ---------------------------------------------------------------- discovery

def find_root(arg):
    if arg:
        return os.path.abspath(arg)
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    return os.getcwd()


def walk_files(root):
    found, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in PY_EXT and ext not in JS_EXT:
                continue
            if fn.endswith(".min.js"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    skipped.append(rel)
                    continue
            except OSError:
                continue
            found.append(rel)
    return found, skipped


def is_test_path(rel):
    parts = rel.split("/")
    base = parts[-1]
    if any(p in TEST_DIRS for p in parts[:-1]):
        return True
    return bool(
        re.match(r"^test_.*\.py$", base) or re.match(r".*_test\.py$", base)
        or re.match(r".*\.(test|spec)\.[a-z]+$", base)
    )


# ---------------------------------------------------------------- python

def find_py(files, parts):
    for prefix in ("", "src/", "lib/"):
        base = prefix + "/".join(parts)
        for cand in (base + ".py", base + "/__init__.py"):
            if cand in files:
                return cand
    return None


def py_defs(tree):
    defs = []

    def visit(body, prefix, in_class):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append({"name": n.name, "qual": prefix + n.name,
                             "kind": "method" if in_class else "function", "line": n.lineno})
                visit(n.body, prefix + n.name + ".", False)
            elif isinstance(n, ast.ClassDef):
                defs.append({"name": n.name, "qual": prefix + n.name, "kind": "class", "line": n.lineno})
                visit(n.body, prefix + n.name + ".", True)
            else:
                for attr in ("body", "orelse", "finalbody"):
                    sub = getattr(n, attr, None)
                    if isinstance(sub, list):
                        visit(sub, prefix, in_class)
                for h in getattr(n, "handlers", []) or []:
                    visit(h.body, prefix, in_class)

    visit(tree.body, "", False)
    return defs


def parse_python(rel, text, files):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    imports, external, import_names = set(), set(), defaultdict(set)
    here = rel.split("/")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                target = find_py(files, a.name.split("."))
                if target:
                    imports.add(target)
                else:
                    external.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = node.module.split(".") if node.module else []
            if node.level:
                up = node.level - 1
                if up > len(here):
                    continue
                pkg = here[: len(here) - up]
                target = None
                for cand in ("/".join(pkg + base) + ".py", "/".join(pkg + base) + "/__init__.py"):
                    if cand in files:
                        target = cand
                        break
                sub_parts = pkg + base
            else:
                target = find_py(files, base) if base else None
                sub_parts = base
            hit = bool(target)
            unresolved = False
            for a in node.names:
                if a.name == "*":
                    unresolved = True
                    continue
                sub = find_py(files, sub_parts + [a.name]) if (sub_parts and (node.level or base)) else None
                if sub:
                    imports.add(sub)
                    hit = True
                else:
                    unresolved = True
                    if target:
                        import_names[target].add(a.name)
            # `from pkg import submodule` only needs the submodule, not the package __init__.py
            if target and (unresolved or not target.endswith("__init__.py")):
                imports.add(target)
            if not hit and not node.level and base:
                external.add(base[0])
    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            refs.add(node.attr)
    return {
        "defs": py_defs(tree), "imports": sorted(imports), "external": sorted(external),
        "import_names": {k: sorted(v) for k, v in import_names.items()}, "refs": refs,
    }


# ---------------------------------------------------------------- javascript / typescript

JS_IMPORT_RES = [
    re.compile(r"""\bimport\s+(?:[\w*${}\s,]+?\s+from\s+)?['"]([^'"]+)['"]"""),
    re.compile(r"""\bexport\s+(?:\*|\{[^}]*\})\s*(?:as\s+\w+\s*)?from\s+['"]([^'"]+)['"]"""),
    re.compile(r"""\brequire\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""\bimport\(\s*['"]([^'"]+)['"]\s*\)"""),
]
JS_FROM_RE = re.compile(r"""\b(?:import|export)\s+(?:type\s+)?([^'";]*?)\s+from\s+['"]([^'"]+)['"]""", re.S)
JS_DEF_RES = [
    ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)", re.M)),
    ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.M)),
    ("type", re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)", re.M)),
    ("function", re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*(?::[^=]+)?=>", re.M)),
    ("function", re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function", re.M)),
]
JS_IDENT = re.compile(r"[A-Za-z_$][\w$]*")


def strip_js_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


def resolve_js(files, rel, spec):
    if spec.startswith("."):
        base = os.path.normpath(os.path.join(os.path.dirname(rel), spec)).replace(os.sep, "/")
        bases = [base]
    else:
        bases = [spec, "src/" + spec]
    for b in bases:
        if b in files:
            return b
        for ext in sorted(JS_EXT):
            if b + ext in files:
                return b + ext
        for ext in sorted(JS_EXT):
            if b + "/index" + ext in files:
                return b + "/index" + ext
        # TypeScript ESM style: import "./x.js" for x.ts
        stem, ext = os.path.splitext(b)
        if ext in (".js", ".jsx", ".mjs", ".cjs"):
            for alt in (".ts", ".tsx"):
                if stem + alt in files:
                    return stem + alt
    return None


def js_clause_names(clause):
    names = set()
    clause = clause.strip()
    if "*" in clause:
        names.add("*")
    brace = re.search(r"\{([^}]*)\}", clause)
    if brace:
        for part in brace.group(1).split(","):
            part = re.sub(r"^\s*type\s+", "", part.strip())
            if part:
                names.add(part.split(" as ")[0].strip())
    head = clause.split("{")[0].strip().rstrip(",").strip()
    if head and "*" not in head:
        names.add("default")
    return names


def parse_js(rel, text, files):
    clean = strip_js_comments(text)
    imports, external = set(), set()
    import_names = defaultdict(set)
    names_for_spec = defaultdict(set)
    for m in JS_FROM_RE.finditer(clean):
        names_for_spec[m.group(2)] |= js_clause_names(m.group(1))
    for rx in JS_IMPORT_RES:
        for m in rx.finditer(clean):
            spec = m.group(1)
            target = resolve_js(files, rel, spec)
            if target and target != rel:
                imports.add(target)
                import_names[target] |= names_for_spec.get(spec, set())
            elif not spec.startswith("."):
                parts = spec.split("/")
                external.add("/".join(parts[:2]) if spec.startswith("@") else parts[0])
    defs = []
    for kind, rx in JS_DEF_RES:
        for m in rx.finditer(clean):
            line = clean.count("\n", 0, m.start(1)) + 1
            defs.append({"name": m.group(1), "qual": m.group(1), "kind": kind, "line": line})
    seen, uniq = set(), []
    for d in sorted(defs, key=lambda d: d["line"]):
        if d["name"] not in seen:
            seen.add(d["name"])
            uniq.append(d)
    return {"defs": uniq, "imports": sorted(imports), "external": sorted(external),
            "import_names": {k: sorted(v) for k, v in import_names.items()},
            "refs": set(JS_IDENT.findall(clean))}


# ---------------------------------------------------------------- build / load

def build(root):
    files_list, skipped = walk_files(root)
    files = set(files_list)
    nodes, refs_by_file = {}, {}
    for rel in files_list:
        full = os.path.join(root, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            mtime = int(os.path.getmtime(full))
        except OSError:
            continue
        lang = "python" if rel.endswith(".py") else "js"
        parsed = parse_python(rel, text, files) if lang == "python" else parse_js(rel, text, files)
        if parsed is None:
            skipped.append(rel + " (parse error)")
            continue
        refs_by_file[rel] = parsed.pop("refs")
        parsed.update({"lang": lang, "lines": text.count("\n") + 1, "mtime": mtime, "is_test": is_test_path(rel)})
        nodes[rel] = parsed

    edges = {}
    for src, node in nodes.items():
        for dst in node["imports"]:
            edges[(src, dst, "import")] = set(node["import_names"].get(dst, []))

    index = defaultdict(set)
    for rel, node in nodes.items():
        for d in node["defs"]:
            index[d["name"]].add(rel)
    # Name-only edges need a distinctive name: a plain lowercase word (branches, locations)
    # also appears in SQL strings, comments and prose, so it would create false links.
    usable = {
        n: fs for n, fs in index.items()
        if len(n) >= 4 and n not in COMMON_NAMES and not n.startswith("__")
        and re.search(r"[A-Z_0-9]", n) and len(fs) <= AMBIGUOUS_LIMIT
    }
    for src, refs in refs_by_file.items():
        own = {d["name"] for d in nodes[src]["defs"]}
        for name in refs & usable.keys():
            if name in own:
                continue
            for dst in usable[name]:
                if dst == src or (src, dst, "import") in edges:
                    continue
                edges.setdefault((src, dst, "ref"), set()).add(name)

    graph = {
        "version": VERSION, "root": root, "generated": int(time.time()),
        "files": nodes, "skipped": skipped,
        "edges": [{"src": s, "dst": d, "kind": k, "names": sorted(v)} for (s, d, k), v in sorted(edges.items())],
    }
    out_dir = os.path.join(root, ".codegraph")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, ".gitignore"), "w") as fh:
        fh.write("*\n")
    with open(os.path.join(out_dir, "graph.json"), "w") as fh:
        json.dump(graph, fh)
    return graph


def load(root, rebuild):
    path = os.path.join(root, ".codegraph", "graph.json")
    if rebuild or not os.path.exists(path):
        return build(root), True
    with open(path) as fh:
        graph = json.load(fh)
    if graph.get("version") != VERSION:
        return build(root), True
    current, _ = walk_files(root)
    changed = removed = 0
    for rel, node in graph["files"].items():
        try:
            if int(os.path.getmtime(os.path.join(root, rel))) != node["mtime"]:
                changed += 1
        except OSError:
            removed += 1
    new = len(set(current) - set(graph["files"]))
    if changed or removed or new:
        print(f"WARNING: graph is stale ({changed} changed, {new} new, {removed} removed files). "
              f"Re-run with --rebuild or the build command.", file=sys.stderr)
    return graph, False


class G:
    def __init__(self, graph):
        self.graph = graph
        self.files = graph["files"]
        self.fwd = defaultdict(list)
        self.rev = defaultdict(list)
        for e in graph["edges"]:
            self.fwd[e["src"]].append(e)
            self.rev[e["dst"]].append(e)

    def seeds(self, arg):
        arg = os.path.relpath(os.path.abspath(arg), self.graph["root"]).replace(os.sep, "/") \
            if os.path.exists(arg) else arg.strip("/")
        if arg in self.files:
            return [arg]
        pref = [f for f in self.files if f.startswith(arg.rstrip("/") + "/")]
        if pref:
            return sorted(pref)
        suff = [f for f in self.files if f.endswith("/" + arg) or os.path.basename(f) == arg]
        if suff:
            return sorted(suff)
        raise SystemExit(f"No indexed file or directory matches '{arg}'. Run `summary` to see what is indexed.")

    def walk(self, seeds, depth, direction, include_refs=True):
        adj = self.rev if direction == "rev" else self.fwd
        seen = {s: (0, "import", None) for s in seeds}
        queue = deque((s, 0, "import") for s in seeds)
        while queue:
            cur, d, conf = queue.popleft()
            if d >= depth:
                continue
            for e in adj.get(cur, []):
                if e["kind"] == "ref" and not include_refs:
                    continue
                nxt = e["src"] if direction == "rev" else e["dst"]
                kind = "ref" if (conf == "ref" or e["kind"] == "ref") else "import"
                if nxt not in seen or (seen[nxt][1] == "ref" and kind == "import"):
                    seen[nxt] = (d + 1, kind, (cur, e["names"]))
                    queue.append((nxt, d + 1, kind))
        return {k: v for k, v in seen.items() if k not in seeds}


# ---------------------------------------------------------------- commands

def fmt_hits(hits, files, limit):
    lines = []
    ordered = sorted(hits.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    for f, (d, kind, via) in ordered[:limit]:
        label = "certain " if kind == "import" else "possible"
        names = f"  uses: {', '.join(via[1][:5])}" if via and via[1] else ""
        test = " [test]" if files[f]["is_test"] else ""
        lines.append(f"  d{d} {label} {f}{test}{names}")
    if len(ordered) > limit:
        lines.append(f"  ... {len(ordered) - limit} more (use --limit)")
    return lines


def name_matches_test(src, test):
    stem = os.path.splitext(os.path.basename(src))[0].lower()
    t = os.path.basename(test).lower()
    return stem in t and stem not in {"index", "__init__", "main", "app"}


def impact_report(gr, seeds, depth, limit, as_json):
    hits = gr.walk(seeds, depth, "rev")
    files = gr.files
    tests = {f: v for f, v in hits.items() if files[f]["is_test"]}
    for t, node in files.items():
        if node["is_test"] and t not in tests and any(name_matches_test(s, t) for s in seeds):
            tests[t] = (99, "name", None)
    for s in seeds:
        if files[s]["is_test"]:
            tests[s] = (0, "import", None)
    non_tests = {f: v for f, v in hits.items() if not files[f]["is_test"]}
    if as_json:
        print(json.dumps({"seeds": seeds, "impacted": {f: {"depth": v[0], "confidence": v[1]} for f, v in non_tests.items()},
                          "tests": {f: {"depth": v[0], "confidence": v[1]} for f, v in tests.items()}}, indent=1))
        return
    print(f"Seeds ({len(seeds)}): {', '.join(seeds[:8])}{' ...' if len(seeds) > 8 else ''}")
    print(f"Blast radius: {len(non_tests)} dependent file(s), {len(tests)} test file(s) (depth <= {depth})")
    if non_tests:
        print("Dependents (certain = import chain, possible = name reference only):")
        print("\n".join(fmt_hits(non_tests, files, limit)))
    if tests:
        print("Tests to run (closest first):")
        for f, (d, kind, via) in sorted(tests.items(), key=lambda kv: (kv[1][0], kv[0]))[:limit]:
            how = ("name match" if kind == "name" else "changed itself" if d == 0
                   else f"d{d} {'certain' if kind == 'import' else 'possible'}")
            print(f"  {f}  ({how})")
    else:
        print("Tests: none found through the graph. Check test coverage by hand before relying on this.")


def cmd_summary(gr, limit, as_json):
    files, edges = gr.files, gr.graph["edges"]
    indeg, outdeg = defaultdict(int), defaultdict(int)
    for e in edges:
        if e["kind"] == "import":
            indeg[e["dst"]] += 1
            outdeg[e["src"]] += 1
    tests = [f for f, n in files.items() if n["is_test"]]
    code = [f for f in files if f not in tests and not (f.endswith("__init__.py") and not files[f]["defs"])]
    entry = [f for f in code if indeg[f] == 0]
    covered, queue = set(), deque(tests)
    while queue:
        cur = queue.popleft()
        for e in gr.fwd.get(cur, []):
            if e["kind"] == "import" and e["dst"] not in covered:
                covered.add(e["dst"])
                queue.append(e["dst"])
    untested = [f for f in code if f not in covered and not any(name_matches_test(f, t) for t in tests)]
    ext = defaultdict(int)
    for n in files.values():
        for x in n["external"]:
            ext[x] += 1
    sccs = find_cycles(files, edges)
    if as_json:
        print(json.dumps({"files": len(files), "tests": len(tests), "edges": len(edges), "cycles": sccs,
                          "untested": untested[:limit]}, indent=1))
        return
    langs = defaultdict(int)
    for n in files.values():
        langs[n["lang"]] += 1
    print(f"Graph: {len(files)} files ({', '.join(f'{k}: {v}' for k, v in sorted(langs.items()))}), "
          f"{len(tests)} test files, {sum(1 for e in edges if e['kind'] == 'import')} import edges, "
          f"{sum(1 for e in edges if e['kind'] == 'ref')} name-reference edges")
    print("Most depended-on (change with care):")
    for f, c in sorted(indeg.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
        print(f"  {c:3d} importers  {f}")
    print("Entry points (code nothing imports):")
    for f in entry[:limit]:
        print(f"  {f}")
    print(f"Possibly untested code files ({len(untested)}):")
    for f in untested[:limit]:
        print(f"  {f}")
    if sccs:
        print("Import cycles:")
        for comp in sccs[:limit]:
            print("  " + " <-> ".join(comp[:6]) + (" ..." if len(comp) > 6 else ""))
    print("Top external packages: " + (", ".join(f"{k} ({v})" for k, v in sorted(ext.items(), key=lambda kv: -kv[1])[:10]) or "none"))
    if gr.graph["skipped"]:
        print(f"Skipped files: {len(gr.graph['skipped'])} (too large or unparseable)")


def find_cycles(files, edges):
    sys.setrecursionlimit(max(10000, len(files) * 4))
    adj = defaultdict(list)
    for e in edges:
        if e["kind"] == "import":
            adj[e["src"]].append(e["dst"])
    index, low, on, stack, out, counter = {}, {}, set(), [], [], [0]

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in adj[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(sorted(comp))

    for v in sorted(files):
        if v not in index:
            strong(v)
    return out


def cmd_symbol(gr, name, limit, as_json):
    defs = [(f, d) for f, n in gr.files.items() for d in n["defs"] if d["name"] == name or d["qual"] == name]
    users = []
    defining = {f for f, _ in defs}
    for e in gr.graph["edges"]:
        if name in e["names"]:
            users.append((e["src"], e["dst"], e["kind"]))
        elif "*" in e["names"] and e["dst"] in defining:
            users.append((e["src"], e["dst"], "ref"))  # namespace or star import: may use it
    if as_json:
        print(json.dumps({"defined": [{"file": f, **d} for f, d in defs], "referenced_by": users}, indent=1))
        return
    if not defs:
        print(f"'{name}' is not defined in any indexed file (it may be external, dynamic or too short to track).")
    for f, d in defs[:limit]:
        print(f"Defined: {f}:{d['line']}  {d['kind']} {d['qual']}")
    if users:
        print("Referenced by:")
        for s, dst, kind in sorted(users)[:limit]:
            t = " [test]" if gr.files[s]["is_test"] else ""
            print(f"  {'certain ' if kind == 'import' else 'possible'} {s}{t} -> {dst}")
    elif defs:
        print("No other file references it through the graph. Double-check with grep before deleting or renaming.")


def git_changed(root, base):
    cmds = []
    if base:
        cmds.append(["git", "diff", "--name-only", f"{base}...HEAD"])
    else:
        cmds += [["git", "diff", "--name-only", "HEAD"], ["git", "ls-files", "--others", "--exclude-standard"]]
    out = set()
    for c in cmds:
        r = subprocess.run(c, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"git failed: {r.stderr.strip()}")
        out.update(x for x in r.stdout.splitlines() if x)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the graph before answering")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=40)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("summary")
    for name in ("impact", "deps", "tests-for"):
        p = sub.add_parser(name)
        p.add_argument("path")
        p.add_argument("--depth", type=int, default=3)
    p = sub.add_parser("symbol")
    p.add_argument("name")
    p = sub.add_parser("changed")
    p.add_argument("--base")
    p.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    root = find_root(args.root)
    if args.cmd == "build":
        g = build(root)
        langs = defaultdict(int)
        for n in g["files"].values():
            langs[n["lang"]] += 1
        print(f"Built {root}/.codegraph/graph.json: {len(g['files'])} files "
              f"({', '.join(f'{k}: {v}' for k, v in sorted(langs.items())) or 'no python or js/ts files'}), "
              f"{len(g['edges'])} edges, {len(g['skipped'])} skipped")
        return
    graph, _ = load(root, args.rebuild)
    gr = G(graph)
    if not gr.files:
        print("No Python or JS/TS files indexed. This skill does not support this repo's languages; use grep/LSP instead.")
        return
    if args.cmd == "summary":
        cmd_summary(gr, args.limit, args.json)
    elif args.cmd == "impact":
        impact_report(gr, gr.seeds(args.path), args.depth, args.limit, args.json)
    elif args.cmd == "deps":
        seeds = gr.seeds(args.path)
        hits = gr.walk(seeds, args.depth, "fwd", include_refs=True)
        if args.json:
            print(json.dumps({f: {"depth": v[0], "confidence": v[1]} for f, v in hits.items()}, indent=1))
        else:
            print(f"Dependencies of {', '.join(seeds[:5])} (depth <= {args.depth}):")
            print("\n".join(fmt_hits(hits, gr.files, args.limit)) or "  none inside the repo")
            ext = sorted({x for s in seeds for x in gr.files[s]["external"]})
            print("External packages: " + (", ".join(ext) or "none"))
    elif args.cmd == "tests-for":
        impact_report(gr, gr.seeds(args.path), args.depth, args.limit, args.json)
    elif args.cmd == "symbol":
        cmd_symbol(gr, args.name, args.limit, args.json)
    elif args.cmd == "changed":
        changed = [c for c in git_changed(root, args.base) if c in gr.files]
        if not changed:
            print("No changed Python or JS/TS files found in git. Nothing to analyze.")
            return
        print(f"Changed indexed files ({len(changed)}): {', '.join(changed[:10])}{' ...' if len(changed) > 10 else ''}")
        impact_report(gr, changed, args.depth, args.limit, args.json)


if __name__ == "__main__":
    main()
