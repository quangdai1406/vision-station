# AISI MES integration for the 4-camera station - Building Plan (Language Assigned)

Summary: 11 tasks. 6 Python, 3 TypeScript, 1 PowerShell, 1 mixed test (Python part scored). 0 Rust. 0 Python + Rust module.

Plan: `docs/plans/2026-11-04-mes-integration.md`. Owners and dates: none given. Defaults used: margin 5; no team skill or hardware facts given. Scorecard: the skill's (the user's own scores). Weights are my 1 to 3 judgment from the task text. TypeScript and PowerShell tasks are fixed by the existing code (the MES is TypeScript, the laptop task scripts are PowerShell) and are not scored.

| Task | Language | Python | Rust | Gap | Confidence | Route | Why |
|---|---|---|---|---|---|---|---|
| M1 Sessions, tickets, hub card | TypeScript (fixed) | - | - | - | n/a | fixed | changes the MES |
| M2 Identity on events | TypeScript (fixed) | - | - | - | n/a | fixed | changes the MES route |
| M3 Criteria with names | TypeScript (fixed) | - | - | - | n/a | fixed | changes the MES route |
| S1 Session client and ticket login | Python | 67 | 55 | 12 | 0.90 | execute | extends the stdlib-only Python station |
| S2 Remote model client | Python | 76 | 67 | 9 | 0.80 | ask | implements a Python protocol; exceptions give the FAULT |
| S3 Defect proposal | Python | 63 | 50 | 13 | 0.90 | execute | extends the Python session and UI |
| S4 Evidence transport | Python | 69 | 59 | 10 | 0.85 | execute | stdlib copy with sha256; runs on the Pi without a build |
| L1 Inference service | Python | 71 | 66 | 5 | 0.70 | ask | torch and the model store are Python |
| L2 Load protection, backups | PowerShell (fixed) | - | - | - | n/a | fixed | Windows Task Scheduler and `pg_dump` |
| E1 End-to-end test | Python + TypeScript tests | 65 | 49 | 16 | 0.90 | execute | flow test in the Python suite |
| E2 Pi setup and checks | Python | 71 | 57 | 14 | 0.90 | execute | OpenCV checks on the Pi |

**L1 note:** the gap equals the margin (5) and speed has weight 3, so I looked for one speed-critical part we would write. None qualifies: the speed-critical work is GPU scoring inside torch; what we write is HTTP, image decode and model loading. No Rust part can be named, so the rule picks Python. Revisit if `tools/pi_check` shows Python overhead, not the GPU, is the bottleneck.

JEV routing of the language choices: 0.85 or higher = execute; 0.70 to 0.84 = decided, the owner approves first. Confidence is my judgment; S2 and L1 are the two to confirm (small gaps, and both depend on decision 1 in the plan, where the model runs).

Dependencies (from the plan's interfaces): M2 and S1 need M1; S3 needs M3 and S1; S2 needs L1; E1 needs all; E2 needs L1 and S2.

Open questions: team Python, Rust and TypeScript skill (unknown); Raspberry Pi model and RAM (unknown); whether the plant LAN is isolated from guests and Wi-Fi (unknown).

Not verified: the weights are my judgment, not measured; nothing has run on the Pi or through the new MES routes.
