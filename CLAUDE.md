# Superpowers Workflow (mandatory for every project)

For any non-trivial task (feature, refactor, multi-step change), follow these phases in order. Invoke each named skill with the Skill tool. Do not skip or reorder phases. Trivial one-line fixes may skip phases 1-9 but still use 10-16.

Priority: user's direct instructions > this file > skill defaults.

## Planning phase

1. **Brainstorm** — `superpowers:brainstorming`. Clarify intent, requirements, design. No code yet.
2. **Plan** — `superpowers:writing-plans`. Write the plan to `docs/plans/<date>-<topic>.md`.
3. **Pick language/tooling per task** — `python-rust-task-picker`. Annotate each plan task as Python or Rust (the "Python/Rush" picker) and edit the plan accordingly.
4. **JEV One System** — `jev-system-one`. Add the JEV One System section to the plan MD file.
5. **Simplify the plan** — `simplify` (or `simplify-task`) applied to the plan MD. Remove redundant or over-engineered tasks.
6. **Progress tracker** — create `docs/plans/<date>-<topic>-PROGRESS.md` with one checkbox per plan task (status: todo / doing / done / blocked, date, notes). Update it after every task. It is the source of truth for resuming work.

## Setup phase

7. **Worktree** — `superpowers:using-git-worktrees`. Do all implementation in an isolated worktree/branch.
8. **Code map** — `code-graph`. Build the code graph before touching shared code.
9. **Index** — write `docs/CODE_INDEX.md` listing every file and its functions/classes (one line each).

## Build loop (repeat per plan task until the plan is finished)

10. **Multi-agent runner** — `multi-agent-job-runner` (with `superpowers:subagent-driven-development`) to dispatch tasks.
11. **TDD** — `superpowers:test-driven-development`. Failing test first, then minimal code.
12. **Parallelize** — `superpowers:dispatching-parallel-agents` for independent tasks (no shared state).
13. **Debug** — `superpowers:systematic-debugging` on any failure. Find root cause before fixing.
14. **Verify** — `superpowers:verification-before-completion`. Run tests/build; show evidence before claiming done.
15. **Review** — `superpowers:requesting-code-review` (and `superpowers:receiving-code-review` for feedback).
16. **Merge** — `superpowers:finishing-a-development-branch`. Then tick the task in the PROGRESS file.

**Repeat 10-16 for the next task until every item in the PROGRESS file is done.** Then do a final verify + review of the whole plan and report.

## Rules

- **After EVERY change** (any edit, add, rename, move or delete of code, not just per task): update the code graph (`code-graph`) AND `docs/CODE_INDEX.md` in the same commit. A change is not done until both match the code.
- Token-efficient: be terse, read only what is needed, no repeated re-derivation.
- Never claim success without verification output.
- If a named skill is unavailable in the session, say so, then perform that phase manually; do not silently skip it.
- Never skip or delete tests to get green.
