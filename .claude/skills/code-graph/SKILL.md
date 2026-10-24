---
name: code-graph
description: Build and query a lightweight code graph (files, definitions, imports, references, tests) so planning, coding and testing use the real dependency structure instead of guesses. Use before planning a change, before editing or renaming shared code, when deciding what tests to run, and when reviewing a diff's blast radius in a Python or JS/TS repo. Skip for docs-only or config-only edits, one-file scripts, and repos in other languages.
---

# Code graph

A small static graph of the repo, built by `scripts/codegraph.py` (Python 3, standard library only). It answers three questions fast: **what does this depend on, what depends on it, and which tests reach it.** It is inspired by tools that give agents a code knowledge graph, but it is a simple local script, not a service.

Run from the repo root. `CG` below means `python3 .claude/skills/code-graph/scripts/codegraph.py` (the script is in this repo). The script finds the git root itself; `--root DIR` overrides it.

## Setup

First read `docs/CODEGRAPH.md` (committed overview of modules, hubs, test reach and blind spots). Then:

```
CG build            # writes .codegraph/graph.json (self-ignoring, never commit it)
CG summary          # hubs, entry points, untested code, import cycles, external packages
```

The graph is cheap to rebuild. Rebuild at the start of a task, and again after you edit files (`--rebuild` on any command does it first). If a command warns that the graph is stale, rebuild before trusting it.

## 1. Planning

Before proposing a plan for a change that touches existing code:

1. `CG summary` once per task to see hubs and untested areas.
2. For each file or directory you expect to change: `CG impact <path>`.
3. For each function or class you expect to change: `CG symbol <name>`.
4. Put a **Blast radius** section in the plan: files that depend on the change (certain vs possible), the tests that reach it, and anything with no test coverage.
5. If the radius is larger than the user probably expects, say so before coding, and offer a smaller change.

`CG deps <path>` lists what a file relies on, including external packages. Use it to find where to put new code and which existing helpers to reuse.

## 2. Coding

- Before editing a shared symbol, run `CG symbol <name>` and read its callers. Change callers in the same change, or keep the old signature working.
- Before deleting or renaming anything, confirm with `CG symbol <name>` and then `grep` for the name as text. The graph cannot see dynamic access.
- Prefer files that already sit near the code you depend on. Do not add an import that creates a new cycle; `CG summary` lists current cycles.
- After a batch of edits: `CG --rebuild changed`. Compare its dependents list against your plan. A surprise dependent means an unplanned effect: look at it before moving on.

## 3. Testing

- Start with `CG tests-for <changed file>` or `CG changed` (uses git diff; add `--base origin/main` for a branch). Run those tests first, closest first, using the project's own test command.
- Then run the wider suite once, or at least the tests for the `possible` dependents.
- If the graph finds no test for changed code, say so, and add one or tell the user. Never report "tests pass" when none of them reach the change.
- When adding a test, put it where the graph (and a human) will connect it: import the code under test directly, and name the file after the module.

## How to read the output

- **certain**: reached through an import chain. Treat as real.
- **possible**: matched only by a symbol name used in another file. Check by reading before relying on it; names that are defined in many places, very short, or very common are ignored on purpose.
- `d1`, `d2`, `d3`: how many hops away. Default depth is 3 (`CG impact <path> --depth N` to change).
- Global flags go before the command: `CG --limit 20 --json --rebuild impact <path>` (`--limit` caps lines, `--json` gives machine output).
- A path can be a file, a directory, or a basename (every file with that name is used).

## Limits (state these when they matter)

- Languages: Python (parsed with `ast`) and JS/TS/JSX/TSX/MJS/CJS (regular expressions). Other languages are not indexed.
- Static only. It does not see dynamic imports built from strings, reflection, dependency injection, plugin loading, decorators that register things, generated code, or framework conventions (routes, templates, config-driven wiring).
- Not resolved: path aliases (tsconfig `paths`, webpack aliases), monorepo package names, and star re-exports across packages. These show up as external packages or missing edges.
- Skips `node_modules`, virtualenvs, build output, hidden folders, `*.min.js`, and files over 1 MB. Unparseable files are listed as skipped.
- A graph with no edges for a file means "no evidence", not "safe". Fall back to `grep` and reading the code.

## Rules

- The graph guides; it does not replace reading the code or running the tests.
- Never write graph output into source files or commit `.codegraph/`.
- Do not claim certainty beyond the graph: report certain and possible dependents separately, and list what the graph cannot see.

## Project notes: vision-station

- **Layout:** Python 3, standard library only for `station/` (camera inspection station: contracts, fusion, session state machine, storage, MES client + outbox, cameras/inference/outputs, UI, app wiring). `tools/` is the webcam repeatability and threshold-selection tooling (needs numpy; opencv-python only for capture on the PC, imported lazily). Tests are `tests/test_*.py` (unittest).
- **High-risk hub:** `station/contracts.py` (about 19 importers): shared types and Protocols. Change it only with the orchestrator/user; every module codes to it.
- **Wiring the graph cannot see:** see `.claude/skills/code-graph/blind_spots.txt` (also printed in `docs/CODEGRAPH.md`): config-driven wiring in `station/app.py`, duck-typed session collaborators, the `snapshot()` keys shared with the UI page script, `tools/FORMAT.md` as the capture/report contract, and the MES API in the AISI_MES repo.
- **Tests:** `python3 -m unittest discover -s tests`. No camera, database or network is needed. Use `CG tests-for` / `CG changed` to pick tests first, then run the whole suite before finishing.
- **Keeping the overview current:** after code changes run `python3 .claude/skills/code-graph/scripts/gen_codegraph_doc.py` and commit `docs/CODEGRAPH.md` with the change (`--check` exits 1 when it is stale). `.codegraph/` is never committed.
