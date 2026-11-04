# Progress: AISI MES integration (2026-11-04)

Plan: `docs/plans/2026-11-04-mes-integration.md`. Language per task: `docs/plans/26_1104_mes_integration_building_plan.md`. Status: todo / doing / done / blocked. Decisions 1 to 8 were made on 2026-11-04 (plan, "Decisions").

| # | Task | Language | Decision gate | Status | Date | Notes |
|---|---|---|---|---|---|---|
| M1 | Sessions, one-time tickets, hub card | TypeScript | 2 execute, 7 ask | todo | | |
| M2 | Identity on inspection events | TypeScript | 2 execute, 7 ask | todo | | |
| M3 | Criteria with names | TypeScript | none: start now | todo | | |
| S1 | Session client and ticket login | Python | 2, 8 execute | todo | | |
| S2 | Remote model client | Python | 1 ask | todo | | |
| S3 | Defect proposal with pictures | Python | 3 execute, 4 ask | todo | | |
| S4 | Evidence transport to the laptop | Python | 5 ask | todo | | |
| L1 | Inference service on the laptop | Python | 1 ask | todo | | |
| L2 | Load protection and backups | PowerShell | none: start now | todo | | |
| E1 | End-to-end test | Python + TypeScript | after all | todo | | |
| E2 | Pi setup guide and hardware checks | Python + docs | 1 ask, 6 escalated | todo | | |

Decision gates refer to the numbered decisions in the plan. Tasks that can start now with no approval: M3, L2.

Later, after the first release: an MES screen for the Admin to approve proposed defect labels; TLS on the plant LAN (required before use outside the pilot).
