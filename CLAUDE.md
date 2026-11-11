# Autonomous Workflow

## Autonomy
- Work until every item in `docs/plans/*-PROGRESS.md` is `done` or `blocked`. Do not stop to report between tasks.
- Do not ask questions. Pick the simplest reasonable option and write the assumption in the PROGRESS notes.
- Ask the user only for: missing credentials, irreversible actions, or a task `blocked` 3 times.
- When all items are done or blocked: full test run, one review pass, then the final step below, push the branch, then print exactly `ALL_TASKS_COMPLETE`.

## Final step (once, after all tasks are done)
Build the code graph for the whole project (`code-graph`) and write `docs/CODE_INDEX.md` (every file with its functions/classes, one line each). Check both match the code, commit them, then print the marker.

## Resume (every session start, also after compaction)
- Read the PROGRESS file, the last 10 `git log` lines, and `git status`. Continue the first `todo`/`doing` task. Never redo `done` tasks.
- No PROGRESS file yet: write a plan (`docs/plans/<date>-<topic>.md`, small tasks, each with a test command), create the PROGRESS file (checkbox, status, date, notes), work in a branch, create the branch. Skip the graph and index until the final step.

## Task loop (one task at a time)
1. Set `doing`. Write a failing test, then minimal code.
2. Run the task test and the full suite. Show the output.
3. On failure, find the root cause. After 3 failed tries: mark `blocked`, note why, take the next task.
4. Commit with the PROGRESS update. Set `done`. Start the next task.

## JEV System One (routine decisions)
Decide these by one short JSON answer, not prose: next task, pass/fail, blocked or not, skill needed or not.
`{"decision": "...", "confidence": 0.00-1.00, "escalate": false}`
- 0.85 or higher: act.
- 0.70 to 0.84: act, and add one line to the PROGRESS notes saying what is uncertain.
- Below 0.70: think it through once in full. Still below 0.70: mark `blocked`. Do not wait for the user.
Do not use JSON for design, debugging, or writing code. Use normal reasoning there.

## Rules
- No success claim without verification output. Never skip or delete tests to get green.
- Read only what the task needs. Be terse. Load a skill only when its trigger applies.
