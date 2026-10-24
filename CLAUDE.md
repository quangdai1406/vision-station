# vision-station: instructions for every session

## Read the code graph first
1. Read `docs/CODEGRAPH.md` before planning, editing or reviewing anything. It is the committed overview of modules, hubs, which tests reach which file, and what the graph cannot see.
2. At the start of the session run `python3 .claude/skills/code-graph/scripts/codegraph.py build` then `... summary`, and follow `.claude/skills/code-graph/SKILL.md` (planning: `impact`/`symbol`; coding: `--rebuild changed`; testing: `tests-for`).
3. After code changes, regenerate and commit the overview: `python3 .claude/skills/code-graph/scripts/gen_codegraph_doc.py` (`--check` exits 1 if it is stale). Never commit `.codegraph/`.
4. The graph guides; it does not replace reading the code or running the tests. List what it cannot see when it matters.

## Project rules
- `station/` is standard library only. `tools/` may use numpy/opencv-python, imported lazily with `ToolDependencyError`.
- `station/contracts.py` is the shared contract: do not change it without the user's say-so.
- Safety rules that tests lock in: REVIEW is never green and never auto-posted; a part is BAD if any of its 8 images scores >= `t_img`; a verdict needs all 8 images; save before lighting green; the MES post uses `session_id` as the Idempotency-Key.
- Never invent thresholds, sample counts or acceptance limits. `t_low`/`t_img` come from labelled data via `tools/thresholds.py` and are approved by Quality; leave `[Quality to define]` where a number is theirs.
- Secrets: the MES token comes only from the environment variable named in the config. Never write a token or password into a file, log or error text.
- Tests: `python3 -m unittest discover -s tests` must pass before you finish.
- Operator guide for the webcam test (Windows/PowerShell): `docs/repeatability_protocol.md`. Capture/report file contract: `tools/FORMAT.md`.
