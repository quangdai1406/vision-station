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

## Superpowers workflow (mandatory)

For any non-trivial task (feature, refactor, multi-step change), follow these phases **in order**.
Invoke each named skill with the Skill tool. Do not skip or reorder phases. Trivial one-line fixes may skip phases 1-9 but still use 10-16.

Priority: user's direct instructions > this file > skill defaults.

### Planning phase
1. **Brainstorm** — `superpowers:brainstorming`. Clarify intent, requirements, design. No code yet.
2. **Plan** — `superpowers:writing-plans`. Write the plan to `docs/plans/<date>-<topic>.md`.
3. **Pick language/tooling per task** — `python-rust-task-picker`. Annotate each plan task as Python or Rust (the "Python/Rush" picker) and edit the plan accordingly.
4. **JEV One System** — `jev-system-one`. Add the JEV One System section to the plan MD file.
5. **Simplify the plan** — `simplify` (or `simplify-task`) applied to the plan MD. Remove redundant or over-engineered tasks.
6. **Progress tracker** — create `docs/plans/<date>-<topic>-PROGRESS.md` with one checkbox per plan task (status: todo / doing / done / blocked, date, notes). Update it after every task. It is the source of truth for resuming work.

### Setup phase
7. **Worktree** — `superpowers:using-git-worktrees`. Do all implementation in an isolated worktree/branch.
8. **Code map** — `code-graph`. Build the code graph before touching shared code.
9. **Index** — write `docs/CODE_INDEX.md` listing every file and its functions/classes (one line each).

### Build loop (repeat per plan task until the plan is finished)
10. **Multi-agent runner** — `multi-agent-job-runner` (with `superpowers:subagent-driven-development`) to dispatch tasks.
11. **TDD** — `superpowers:test-driven-development`. Failing test first, then minimal code.
12. **Parallelize** — `superpowers:dispatching-parallel-agents` for independent tasks (no shared state).
13. **Debug** — `superpowers:systematic-debugging` on any failure. Find root cause before fixing.
14. **Verify** — `superpowers:verification-before-completion`. Run tests/build; show evidence before claiming done.
15. **Review** — `superpowers:requesting-code-review` (and `superpowers:receiving-code-review` for feedback).
16. **Merge** — `superpowers:finishing-a-development-branch`. Then tick the task in the PROGRESS file.

**Repeat 10-16 for the next task until every item in the PROGRESS file is done.** Then do a final verify + review of the whole plan and report.

### Rules
- **After EVERY change** (any edit, add, rename, move or delete of code, not just per task): update the code graph (`code-graph`) AND `docs/CODE_INDEX.md` in the same commit. A change is not done until both match the code.
- Token-efficient: be terse, read only what is needed, no repeated re-derivation.
- Never claim success without verification output.
- If a named skill is unavailable in the session, say so, then perform that phase manually; do not silently skip it.
- Never skip or delete tests to get green.

### vision-station notes
- The project rules and code-graph rules above win over this workflow where they conflict (for example, never change `station/contracts.py` without the user's say-so).
- "Code graph" here means `docs/CODEGRAPH.md`: after every change run `python3 .claude/skills/code-graph/scripts/gen_codegraph_doc.py`, and also update `docs/CODE_INDEX.md`, in the same commit.
