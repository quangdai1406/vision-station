# AISI MES integration for the 4-camera station - Building Plan (Language Assigned)

Summary: 13 tasks. 7 Python, 4 TypeScript, 1 PowerShell, 1 mixed test (Python part scored). 0 Rust. 0 Python + Rust module.

Defaults used: margin 5 points; no team skill or hardware facts were given. The scorecard is the one in the skill (the user's own scores). Weights are mine, 1 to 3, from the task text. TypeScript and PowerShell tasks are fixed by the existing code (the MES is TypeScript; the laptop task scripts are PowerShell), so they are not scored.

Plan: `docs/plans/2026-11-04-mes-integration.md`.

| Task | Language | Python | Rust | Gap |
|---|---|---|---|---|
| M1 Sessions and tickets | TypeScript (fixed) | - | - | - |
| M2 Identity on events | TypeScript (fixed) | - | - | - |
| M3 Criteria with names | TypeScript (fixed) | - | - | - |
| M4 Hub entry | TypeScript (fixed) | - | - | - |
| S1 MES session client | Python | 63 | 50 | 13 |
| S2 Ticket login and roles | Python | 67 | 55 | 12 |
| S3 Remote model client | Python | 76 | 67 | 9 |
| S4 Defect proposal | Python | 63 | 50 | 13 |
| S5 Evidence transport | Python | 69 | 59 | 10 |
| L1 Inference service | Python | 71 | 66 | 5 |
| L2 Load protection and backups | PowerShell (fixed) | - | - | - |
| E1 End-to-end test | Python + TypeScript tests | 65 | 49 | 16 |
| E2 Pi setup and checks | Python | 71 | 57 | 14 |

## M1 Vision stations, sessions and one-time tickets
- Language: TypeScript (fixed: the MES codebase).
- Why: it changes the MES tables and routes; the whole MES is TypeScript.
- Details from the plan: owner none given; dates none given; depends on nothing in this plan.

## M2 Identity on inspection events
- Language: TypeScript (fixed).
- Why: it changes the existing `/inspection-events` route and service.
- Details: owner none; dates none; depends on M1.

## M3 Criteria with names
- Language: TypeScript (fixed).
- Why: it extends the existing `GET /criteria/:partNumber` route.
- Details: owner none; dates none; depends on nothing.

## M4 Hub entry "4-camera inspection"
- Language: TypeScript (fixed).
- Why: it changes the MES hub page and routes.
- Details: owner none; dates none; depends on M1.

## S1 MES session client and posting header
- Language: Python.
- Why: the station is Python and standard library only; HTTP and JSON are in the standard library.
- Scores: Python 63, Rust 50.
- Details: owner none; dates none; depends on M1.

## S2 Ticket login and roles
- Language: Python.
- Why: it extends `station/auth.py` and `station/ui.py`, which are Python; development time weighs most.
- Scores: Python 67, Rust 55.
- Details: owner none; dates none; depends on S1, M1.

## S3 Remote model client
- Language: Python.
- Why: it implements the existing Python `InferenceModel` protocol with the standard library; the error-handling weight (a failure must become a FAULT) does not need Rust, because Python exceptions are enough.
- Scores: Python 76, Rust 67.
- Details: owner none; dates none; depends on L1.

## S4 Defect proposal with pictures
- Language: Python.
- Why: it extends the Python session and UI, and reuses the existing `OperatorLabel`.
- Scores: Python 63, Rust 50.
- Details: owner none; dates none; depends on M3, S2.

## S5 Evidence transport to the laptop
- Language: Python.
- Why: copying with a sha256 check is in the standard library; it must run on the Pi without a build step.
- Scores: Python 69, Rust 59.
- Details: owner none; dates none; depends on nothing in this plan.

## L1 Inference service on the laptop
- Language: Python.
- Why: torch and the model store are Python; the heavy compute already runs in native torch/CUDA code.
- Scores: Python 71, Rust 66. The gap is 5, the margin, and speed has weight 3, so I checked for one speed-critical part we would write ourselves. None qualifies: the speed-critical work is the GPU scoring inside torch, and what we write is HTTP handling, image decode and model loading. No Rust part is named, so the rule picks Python. Revisit if `tools/pi_check` shows the Python overhead, not the GPU, is the bottleneck.
- Details: owner none; dates none; depends on nothing in this plan.

## L2 Load protection and backups
- Language: PowerShell (fixed: the Windows task scripts already in `scripts/windows/`).
- Why: it drives Windows Task Scheduler and `pg_dump`.
- Details: owner none; dates none; depends on L1 (the inference service status check).

## E1 End-to-end test
- Language: Python (station side, scored) plus TypeScript (MES side tests).
- Why: the flow test lives in the Python test suite; the MES side uses the MES test runner.
- Scores: Python 65, Rust 49.
- Details: owner none; dates none; depends on all earlier tasks.

## E2 Pi setup guide and hardware checks
- Language: Python (plus docs).
- Why: the checks use OpenCV, which has Python bindings already in use; the Pi runs the Python station.
- Scores: Python 71, Rust 57.
- Details: owner none; dates none; depends on L1, S3.

## Language boundaries
None. No Python + Rust module task.

## Open questions
- Team Python, Rust and TypeScript skill level: unknown (defaults used).
- Raspberry Pi model and RAM: unknown. It matters for E2 and for decision 1.
- Whether the plant LAN is isolated from guests and Wi-Fi: unknown. It matters for decision 6 (TLS).

Not verified: the weights are my judgment, not measured; nothing has run on the Pi or through the new MES routes.
