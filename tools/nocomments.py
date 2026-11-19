# ruff: noqa
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# fmt: off
import ast
import difflib
import fnmatch
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path, PurePath

_ENCODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")
_DIRECTIVE_PY = re.compile(
    r"#\s*(noqa|type:|pragma:|nosec|pylint:|pyright:|mypy:|fmt:|isort:|ruff:|flake8:|yapf:|pytype:|coverage:|nocover|skip-file)",
    re.I,
)
_DIRECTIVE_JS = re.compile(
    r"(//|/\*)\s*(@ts-|eslint|prettier|biome|istanbul|c8\b|v8 ignore|tslint|jshint|jscs|globals?\b|webpack|#__PURE__|@__PURE__|@vite-ignore)"
)
_DIRECTIVE_HASH = re.compile(
    r"#\s*(shellcheck|yamllint|noqa|nosec|pragma|fmt:|prettier-ignore|hadolint|tflint|checkov|trivy|cspell)",
    re.I,
)


def _python_info(src):
    lines = io.StringIO(src).readlines()
    cuts = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        if row == 1 and tok.string.startswith("#!"):
            continue
        if row <= 2 and _ENCODING.match(lines[row - 1]):
            continue
        if _DIRECTIVE_PY.match(tok.string):
            continue
        cuts[row] = col
    tree = ast.parse(src)
    docs = []
    doc_names = []
    unsafe = False
    used = set()
    futures_end = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
            if node.id == "__doc__":
                if isinstance(node.ctx, ast.Load):
                    doc_names.append((node.lineno, node.col_offset))
                else:
                    unsafe = True
        elif isinstance(node, ast.Attribute) and node.attr == "__doc__":
            unsafe = True
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docs.append(
                    {
                        "start": first.lineno,
                        "end": first.end_lineno,
                        "col": first.col_offset,
                        "end_col": first.end_col_offset,
                        "sole": len(node.body) == 1,
                        "module": isinstance(node, ast.Module),
                    }
                )
    imports_end = 0
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0].value, "value", None), str)
    ):
        body = body[1:]
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports_end = node.end_lineno
        else:
            break
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures_end = max(futures_end, node.end_lineno)
    return lines, cuts, docs, doc_names, unsafe, used, futures_end, imports_end


def find_python(src):
    _, cuts, docs, _, _, _, _, _ = _python_info(src)
    return sorted(set(cuts) | {d["start"] for d in docs})


def strip_python(src):
    try:
        lines, cuts, docs, doc_names, unsafe, used, futures_end, imports_end = (
            _python_info(src)
        )
    except (SyntaxError, tokenize.TokenError) as exc:
        raise ValueError(f"cannot parse python: {exc}") from exc
    if docs and unsafe:
        raise ValueError(
            "code reads or writes __doc__ on an object: cannot move it safely"
        )
    module_doc = next((d for d in docs if d["module"]), None)
    rename = bool(doc_names) and module_doc is not None
    if rename and "DOC_TEXT" in used:
        raise ValueError("DOC_TEXT already exists: cannot move the module docstring")
    dropped = set()
    replaced = {}
    insert_after = {}
    for d in docs:
        first = lines[d["start"] - 1]
        last = lines[d["end"] - 1]
        tail = last.encode("utf-8")[d["end_col"] :].decode("utf-8")
        if first[: d["col"]].strip() or tail.strip():
            raise ValueError(
                f"docstring shares a line with code (line {d['start']}): rewrite it first"
            )
        span = range(d["start"], d["end"] + 1)
        if d["module"] and rename:
            text = "".join(lines[d["start"] - 1 : d["end"]])
            if not text.endswith("\n"):
                text += "\n"
            if imports_end > d["end"]:
                if any(row <= imports_end for row, _ in doc_names):
                    raise ValueError("__doc__ is read inside the import block")
                dropped.update(span)
                gap = (
                    "\n"
                    if imports_end < len(lines) and lines[imports_end].strip()
                    else ""
                )
                insert_after[imports_end] = "\nDOC_TEXT = " + text + gap
            else:
                replaced[d["start"]] = "DOC_TEXT = " + text
                dropped.update(range(d["start"] + 1, d["end"] + 1))
        elif d["sole"] and not d["module"]:
            eol = last[len(last.rstrip("\r\n")) :] or "\n"
            replaced[d["start"]] = first[: d["col"]] + "pass" + eol
            dropped.update(range(d["start"] + 1, d["end"] + 1))
        else:
            dropped.update(span)
    renames = {}
    if rename:
        for row, col in doc_names:
            renames.setdefault(row, []).append(col)
    out = []
    marks = set()
    for number, line in enumerate(lines, start=1):
        if number in replaced:
            out.append(replaced[number])
        elif number in dropped:
            marks.add(len(out))
        else:
            if number in cuts:
                body = line[: cuts[number]].rstrip()
                if not body:
                    marks.add(len(out))
                    continue
                eol = line[len(line.rstrip("\r\n")) :]
                line = body + eol
            if number in renames:
                raw = line.encode("utf-8")
                for col in sorted(renames[number], reverse=True):
                    raw = raw[:col] + b"DOC_TEXT" + raw[col + len(b"__doc__") :]
                line = raw.decode("utf-8")
            out.append(line)
        if number in insert_after:
            out.append(insert_after[number])
    _collapse(out, marks, opener=True)
    result = "".join(out)
    try:
        ast.parse(result)
    except SyntaxError as exc:
        raise ValueError(f"stripping produced invalid python: {exc}") from exc
    return result


_REGEX_PREV = set("(,=:[!&|?{;+-*%<>~^")
_COND_WORDS = {"if", "while", "for", "with"}
_REGEX_WORDS = {
    "return",
    "typeof",
    "case",
    "in",
    "of",
    "delete",
    "void",
    "throw",
    "new",
    "else",
    "do",
    "await",
    "yield",
    "instanceof",
}


def _collapse(out, marks, opener=False):
    for p in sorted(marks, reverse=True):
        before = p
        while before > 0 and not out[before - 1].strip():
            before -= 1
        after = p
        while after < len(out) and not out[after].strip():
            after += 1
        prev_run = p - before
        next_run = after - p
        if p == 0:
            del out[0:after]
        elif p >= len(out):
            del out[before:p]
        elif opener and prev_run == 0 and out[p - 1].rstrip().endswith(":"):
            del out[p:after]
        elif prev_run and next_run:
            del out[p : p + min(prev_run, next_run)]


def _tidy(text, touched):
    out = []
    marks = set()
    for index, line in enumerate(io.StringIO(text).readlines()):
        if index in touched:
            body = line.rstrip("\r\n")
            eol = line[len(body) :]
            body = body.rstrip(" \t")
            if not body:
                marks.add(len(out))
                continue
            line = body + eol
        out.append(line)
    _collapse(out, marks)
    return "".join(out)


def _skip_quoted(src, i):
    quote = src[i]
    j = i + 1
    while j < len(src) and src[j] != quote:
        if src[j] == "\\":
            j += 1
        elif src[j] == "\n":
            return j
        j += 1
    return j + 1


def _skip_template(src, i):
    n = len(src)
    j = i + 1
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
        elif c == "`":
            return j + 1
        elif c == "$" and src[j + 1 : j + 2] == "{":
            j += 2
            depth = 1
            while j < n and depth:
                d = src[j]
                if d in "'\"":
                    j = _skip_quoted(src, j)
                elif d == "`":
                    j = _skip_template(src, j)
                else:
                    depth += (d == "{") - (d == "}")
                    j += 1
        else:
            j += 1
    return n


def strip_c_like(src, line_comments, regex=False):
    out = []
    touched = set()
    newlines = 0
    prev = ""
    word = ""
    parens = []
    after_cond = False
    i = 0
    n = len(src)

    def emit(text):
        nonlocal newlines
        out.append(text)
        newlines += text.count("\n")

    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if two == "/*":
            end = src.find("*/", i + 2)
            stop = n if end < 0 else end + 2
            text = src[i:stop]
            if (
                text.startswith("/*!")
                or "@license" in text
                or "@preserve" in text
                or _DIRECTIVE_JS.match(text)
            ):
                emit(text)
                i = stop
                prev, word = "x", ""
                continue
            touched.add(newlines)
            i = stop
            continue
        if line_comments and two == "//":
            end = src.find("\n", i)
            end = n if end < 0 else end
            if end > i and src[end - 1] == "\r":
                end -= 1
            text = src[i:end]
            if _DIRECTIVE_JS.match(text) or text.startswith("/// <"):
                emit(text)
                i = end
                continue
            touched.add(newlines)
            i = end
            continue
        if regex and c == "`":
            end = _skip_template(src, i)
            emit(src[i:end])
            i = end
            prev, word = c, ""
            continue
        if c in "'\"":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    j -= 1
                    break
                j += 1
            emit(src[i : j + 1])
            i = j + 1
            prev, word = c, ""
            continue
        if (
            not regex
            and src[i : i + 4].lower() == "url("
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] in "-_"))
        ):
            j = i + 4
            while j < n and src[j] in " \t":
                j += 1
            if j < n and src[j] not in "'\"":
                end = src.find(")", j)
                end = n if end < 0 else end + 1
                emit(src[i:end])
                i = end
                prev, word = ")", ""
                continue
        if (
            regex
            and c == "/"
            and src[i - 1 : i] != "<"
            and (
                prev == ""
                or prev in _REGEX_PREV
                or word in _REGEX_WORDS
                or (prev == ")" and after_cond)
            )
        ):
            j = i + 1
            in_class = False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 1
                elif src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                j += 1
            if j < n and src[j] == "/":
                j += 1
                while j < n and (src[j].isalpha()):
                    j += 1
                emit(src[i:j])
                i = j
                prev, word = "x", ""
                continue
        emit(c)
        i += 1
        if c.isalnum() or c in "_$":
            word += c
            prev = c
        elif not c.isspace():
            if c == "(":
                parens.append(word)
            elif c == ")":
                after_cond = (parens.pop() if parens else "") in _COND_WORDS
            prev, word = c, ""
    return _tidy("".join(out), touched)


_HTML = re.compile(
    r"(?P<keep><!--\[if\b[^\]]*\]>(?:<!-->|.*?<!\[endif\]-->)|<!--<!\[endif\]-->|<!\[CDATA\[.*?\]\]>)"
    r"|<!--.*?-->"
    r"|\{%-?\s*comment\s*-?%\}.*?\{%-?\s*endcomment\s*-?%\}"
    r"|(?P<so><script\b[^>]*>)(?P<sb>.*?)(?P<sc></script\s*>)"
    r"|(?P<to><style\b[^>]*>)(?P<tb>.*?)(?P<tc></style\s*>)"
    r"|(?P<tag><[A-Za-z][^\s/>]*(?:\"[^\"]*\"|'[^']*'|[^'\">])*>)",
    re.S | re.I,
)


def strip_html(src):
    out = []
    touched = set()
    newlines = 0
    pos = 0

    def emit(text):
        nonlocal newlines
        out.append(text)
        newlines += text.count("\n")

    for match in _HTML.finditer(src):
        emit(src[pos : match.start()])
        pos = match.end()
        if match.group("keep"):
            emit(match.group("keep"))
        elif match.group("tag"):
            emit(match.group("tag"))
        elif match.group("so"):
            head = match.group("so")
            body = match.group("sb")
            if "json" not in head.lower() and "template" not in head.lower():
                body = strip_c_like(body, line_comments=True, regex=True)
            emit(head + body + match.group("sc"))
        elif match.group("to"):
            emit(
                match.group("to")
                + strip_c_like(match.group("tb"), line_comments=False)
                + match.group("tc")
            )
        else:
            touched.add(newlines)
    emit(src[pos:])
    return _tidy("".join(out), touched)


_LITERAL_START = re.compile(r"[:-]\s+[|>][+-]?\d*\s*$")
_SH_HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_]\w*)\2")


def _here_ended(here, body):
    mode, text = here
    if mode == "start":
        return body.startswith(text)
    if mode == "contains":
        return text in body
    if mode == "strip":
        return body.strip() == text
    return body == text


def strip_hash(src, kind="yaml"):
    escape = "`" if kind == "ps1" else "\\"
    out = []
    in_block = False
    literal_indent = None
    here = None
    for number, line in enumerate(io.StringIO(src).readlines(), start=1):
        body = line.rstrip("\r\n")
        eol = line[len(body) :]
        if here is not None:
            out.append(line)
            if _here_ended(here, body):
                here = None
            continue
        if literal_indent is not None:
            indent = len(body) - len(body.lstrip())
            if not body.strip() or indent > literal_indent:
                out.append(line)
                continue
            literal_indent = None
        changed = False
        if in_block:
            end = body.find("#>")
            if end < 0:
                continue
            body = body[end + 2 :]
            in_block = False
            changed = True
        pending = None
        quote = None
        j = 0
        while j < len(body):
            c = body[j]
            if quote:
                if c == escape and quote == '"':
                    j += 1
                elif c == quote:
                    quote = None
            elif (
                kind == "ps1"
                and c == "@"
                and body[j + 1 : j + 2] in ("'", '"')
                and not body[j + 2 :].strip()
            ):
                here = ("start", body[j + 1] + "@")
                break
            elif kind == "toml" and body[j : j + 3] in ('"' * 3, "'" * 3):
                delim = body[j : j + 3]
                end = body.find(delim, j + 3)
                if end < 0:
                    here = ("contains", delim)
                    break
                j = end + 2
            elif kind == "sh" and body[j : j + 2] == "<<" and body[j : j + 3] != "<<<":
                match = _SH_HEREDOC.match(body, j)
                if match:
                    pending = ("strip" if match.group(1) else "exact", match.group(3))
                    j = match.end() - 1
            elif kind == "sh" and c == "\\":
                j += 1
            elif c in "'\"":
                before = body[:j].rstrip()[-1:]
                if kind != "yaml" or before == "" or before in ":-,[{":
                    quote = c
            elif kind == "ps1" and body[j : j + 2] == "<#":
                end = body.find("#>", j + 2)
                changed = True
                if end < 0:
                    in_block = True
                    body = body[:j]
                    break
                body = body[:j] + body[end + 2 :]
                continue
            elif c == "#":
                if number == 1 and body.startswith("#!"):
                    break
                if _DIRECTIVE_HASH.match(body[j:]):
                    break
                if kind == "yaml" and j > 0 and body[j - 1] not in " \t":
                    j += 1
                    continue
                if kind == "sh" and j > 0 and body[j - 1] not in " \t;&|(":
                    j += 1
                    continue
                body = body[:j]
                changed = True
                break
            j += 1
        if pending is not None:
            here = pending
        if changed:
            body = body.rstrip(" \t")
            if not body:
                continue
        if kind == "yaml" and _LITERAL_START.search(body):
            literal_indent = len(body) - len(body.lstrip())
        out.append(body + eol)
    return "".join(out)


_DOLLAR = re.compile(r"\$([A-Za-z_]\w*)?\$")


def strip_sql(src):
    out = []
    touched = set()
    newlines = 0
    i = 0
    n = len(src)

    def emit(text):
        nonlocal newlines
        out.append(text)
        newlines += text.count("\n")

    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if two == "--":
            end = src.find("\n", i)
            end = n if end < 0 else end
            if end > i and src[end - 1] == "\r":
                end -= 1
            touched.add(newlines)
            i = end
        elif two == "/*":
            depth = 1
            j = i + 2
            while j < n and depth:
                if src[j : j + 2] == "/*":
                    depth += 1
                    j += 2
                elif src[j : j + 2] == "*/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            touched.add(newlines)
            i = j
        elif c == "'" or c == '"':
            estring = (
                c == "'"
                and i > 0
                and src[i - 1] in "eE"
                and (i < 2 or not (src[i - 2].isalnum() or src[i - 2] == "_"))
            )
            j = i + 1
            while j < n:
                if estring and src[j] == "\\":
                    j += 2
                    continue
                if src[j] == c:
                    if src[j + 1 : j + 2] == c:
                        j += 2
                        continue
                    break
                j += 1
            emit(src[i : j + 1])
            i = j + 1
        elif (
            c == "$"
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_"))
            and _DOLLAR.match(src, i)
        ):
            tag = _DOLLAR.match(src, i).group(0)
            end = src.find(tag, i + len(tag))
            if end < 0:
                emit(src[i:])
                i = n
            else:
                emit(tag + strip_sql(src[i + len(tag) : end]) + tag)
                i = end + len(tag)
        else:
            emit(c)
            i += 1
    return _tidy("".join(out), touched)


def strip_bat(src):
    out = []
    for line in io.StringIO(src).readlines():
        probe = line.strip().lower()
        if probe == "rem" or probe.startswith("rem ") or probe.startswith("::"):
            continue
        out.append(line)
    return "".join(out)


def strip_text(path, src):
    suffix = PurePath(path).suffix.lower()
    if suffix == ".py":
        return strip_python(src)
    if suffix in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
        return strip_c_like(src, line_comments=True, regex=True)
    if suffix == ".rules":
        return strip_c_like(src, line_comments=True)
    if suffix == ".sql":
        return strip_sql(src)
    if suffix == ".toml":
        return strip_hash(src, kind="toml")
    if suffix in (".sh", ".bash"):
        return strip_hash(src, kind="sh")
    if suffix in (".bat", ".cmd"):
        return strip_bat(src)
    if suffix == ".css":
        return strip_c_like(src, line_comments=False)
    if suffix == ".scss":
        return strip_c_like(src, line_comments=True)
    if suffix in (".html", ".htm", ".xml"):
        return strip_html(src)
    if suffix in (".yml", ".yaml"):
        return strip_hash(src, kind="yaml")
    if suffix == ".ps1":
        return strip_hash(src, kind="ps1")
    return src


def find_text(path, src):
    if PurePath(path).suffix.lower() == ".py":
        return find_python(src)
    before = io.StringIO(src).readlines()
    after = io.StringIO(strip_text(path, src)).readlines()
    found = []
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            found.extend(range(i1 + 1, i2 + 1))
    return found


SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".css",
    ".scss",
    ".html",
    ".htm",
    ".xml",
    ".yml",
    ".yaml",
    ".ps1",
    ".sql",
    ".toml",
    ".sh",
    ".bash",
    ".bat",
    ".cmd",
    ".rules",
}
EXCLUDED_DIRS = {
    ".git",
    "_site",
    ".sass-cache",
    "wp-content",
    "wp-includes",
    "node_modules",
    "vendor",
    "third_party",
    "bower_components",
    "dist",
    ".next",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
}
EXCLUDED_ENDINGS = (
    ".min.js",
    ".min.css",
    ".umd.js",
    ".bundle.js",
    ".chunk.js",
    "~",
    ".scssc",
)


def _ignored(rel, ignores):
    name = PurePath(rel).name
    for pattern in ignores:
        if pattern.endswith("/"):
            if rel.startswith(pattern) or "/" + pattern in "/" + rel:
                return True
        elif fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False


def is_target(rel, ignores=()):
    path = PurePath(rel)
    if EXCLUDED_DIRS.intersection(path.parts[:-1]):
        return False
    if path.name.endswith(EXCLUDED_ENDINGS):
        return False
    if _ignored(rel, ignores):
        return False
    return path.suffix.lower() in SUFFIXES


def load_ignores(root):
    path = Path(root) / ".nocommentsignore"
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tracked_files(root):
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
    )
    return [name for name in result.stdout.split("\0") if name]


def staged_files(root):
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [name for name in result.stdout.split("\0") if name]


def staged_text(root, name):
    result = subprocess.run(
        ["git", "show", f":{name}"], cwd=root, check=True, capture_output=True
    )
    raw = result.stdout
    return raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8")


def main(argv):
    args = list(argv)
    if not args or args[0] not in ("check", "strip"):
        print("usage: nocomments.py check|strip [--root DIR] [paths...]")
        return 2
    command = args.pop(0)
    root = Path.cwd()
    paths = []
    staged = False
    while args:
        item = args.pop(0)
        if item == "--root":
            root = Path(args.pop(0))
        elif item == "--staged":
            staged = True
        else:
            paths.append(item)
    root = root.resolve()
    if staged and command != "check":
        print("--staged works with check only")
        return 2
    ignores = load_ignores(root)
    names = paths or (staged_files(root) if staged else tracked_files(root))
    found = 0
    errors = 0
    for name in sorted(names):
        if not is_target(name, ignores):
            continue
        path = root / name
        try:
            encoding = "utf-8"
            if staged:
                text = staged_text(root, name)
            else:
                raw = path.read_bytes()
                if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                    encoding = "utf-16"
                text = raw.decode(encoding)
            if command == "check":
                for line in find_text(name, text):
                    print(f"{name}:{line}")
                    found += 1
            else:
                new = strip_text(name, text)
                if new != text:
                    path.write_bytes(new.encode(encoding))
        except (
            OSError,
            UnicodeDecodeError,
            ValueError,
            SyntaxError,
            tokenize.TokenError,
            subprocess.CalledProcessError,
        ) as exc:
            print(f"{name}: {exc}")
            errors += 1
    if errors:
        return 2
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
