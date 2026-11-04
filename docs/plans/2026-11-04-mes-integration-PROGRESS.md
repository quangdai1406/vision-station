# Progress: AISI MES integration (2026-11-04)

Plan: `docs/plans/2026-11-04-mes-integration.md`. Language per task: `docs/plans/26_1104_mes_integration_building_plan.md`. Status: todo / doing / done / blocked. Decisions 1 to 8 were made on 2026-11-04 (plan, "Decisions").

| # | Task | Language | Decision gate | Status | Date | Notes |
|---|---|---|---|---|---|---|
| M1 | Sessions, one-time tickets, hub card | TypeScript | 2 execute, 7 ask | done | 2026-11-04 | final checker PASS |
| M2 | Identity on inspection events | TypeScript | 2 execute, 7 ask | done | 2026-11-04 | final checker PASS |
| M3 | Criteria with names | TypeScript | none: start now | done | 2026-11-04 | criteria whitelist added |
| S1 | Session client and ticket login | Python | 2, 8 execute | done | 2026-11-04 | final checker PASS |
| S2 | Remote model client | Python | 1 ask | done | 2026-11-04 | remote model; capturedAt; evidence sync wired |
| S3 | Defect proposal with pictures | Python | 3 execute, 4 ask | done | 2026-11-04 | BAD and REVIEW parts only; stored in part.json |
| S4 | Evidence transport to the laptop | Python | 5 ask | done | 2026-11-04 | verified sync; use the script OR the app thread |
| L1 | Inference service on the laptop | Python | 1 ask | done | 2026-11-04 | health busy flag; /v1/models/<part> |
| L2 | Load protection and backups | PowerShell | none: start now | done | 2026-11-04 | PowerShell not run on Windows |
| E1 | End-to-end test | Python + TypeScript | after all | done | 2026-11-04 | real MES driven by real station code: PASS |
| E2 | Pi setup guide and hardware checks | Python + docs | 1 ask, 6 escalated | done | 2026-11-04 | not run on a Pi |

Decision gates refer to the numbered decisions in the plan. All 11 tasks are built (2026-11-04) and passed the final checker. Nothing has run on a Pi, a GPU, Windows or a real hub. Decision 6 (TLS before use outside the pilot) is still open with the owner.

Later, after the first release: an MES screen for the Admin to approve proposed defect labels; TLS on the plant LAN (required before use outside the pilot).
