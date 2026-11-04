# AISI MES integration for the 4-camera station: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An inspector logs in to the AISI MES, opens "4-camera inspection", picks a part number, inspects parts with the cameras, and proposes defects with pictures, all recorded in the MES under their own name.

**Architecture:** The MES (TypeScript/Express) and PostgreSQL stay on the laptop. The 4 cameras and the operator page run on a Raspberry Pi (Python station). The MES hands the logged-in inspector to the Pi with a one-time ticket. The ML model scores pictures on the laptop GPU (an inference service) because the Pi is unlikely to run the model fast enough (unproven). The laptop also keeps training weekly.

**Tech Stack:** TypeScript + Express + PostgreSQL (`aisi_mes`); Python 3 stdlib-only station (`vision-station/station/`); Python + torch inference service on the laptop (new package, not in `station/`); PowerShell for laptop ops.

**Spec:** the owner's request (2026-11-04) and answers: MES has defect criteria per part number; the cameras run on a Raspberry Pi; inspectors log in to the MES and then choose the 4-camera system; the laptop stays the ML trainer and PostgreSQL server. Existing design: `docs/PLAN_multi_part_system.md`, `aisi_mes/docs/vision-station-integration.md`.

Status: plan only. Brainstorming was done through the owner's answers; open questions are in section "Decisions". The `jev-system-one` skill is not available in this session, so its section is written by hand.

## Global Constraints

- `station/` is standard library only. `station/contracts.py` is not changed without the owner's say-so (the new code uses the existing `InferenceModel`, `OperatorLabel`, `Override`).
- Never invent thresholds, defect codes, timeouts, session limits or retention. Use `[Quality to define]` or `[Engineer to define]`.
- MES writes go through `writeAudited`; `audit_trail` stays immutable. Migrations are additive and nullable.
- Tokens only from environment variables; never in a file, URL that is logged, or log line.
- Safety rules stay locked by tests: REVIEW is never green and never auto-posted; any of 8 images at or above `t_img` is BAD; a verdict needs all 8; save before green; the MES post uses `session_id` as the Idempotency-Key.
- A fault (laptop down, MES down, timeout) is never GOOD and never green.
- Follow `CLAUDE.md` (Superpowers workflow): TDD, code graph and `docs/CODE_INDEX.md` regenerated in every commit.

## Review Focus

Failure modes the spec implies that no task's tests would otherwise cover (each is pinned by a test in the owning task):
1. The Pi loses the LAN while a part is mid-inspection: no verdict, FAULT, no green, nothing half-posted (S3, E1).
2. The inspector's MES session expires or they log out while the part is in WAIT_FLIP: the part is finished by a re-login of the same person or marked FAULT; never recorded under a stale identity (S2).
3. The laptop is busy (Sunday training, GPU out of memory): `predict` times out, FAULT with a clear reason (L1, S3).
4. A ticket is replayed or reused: second redeem is rejected (M1).
5. The inspector proposes a defect for a part with no criteria, or an inactive criterion: refused with a message (M3, S4).
6. A post is retried after the session ended: accepted only when the part was captured inside the session window (M2).

## File Structure

| Area | Files | Responsibility |
|---|---|---|
| MES (aisi_mes) | `migrations/finished_good_inspections/0011_vision_sessions.sql`, `src/departments/finished_good_inspections/visionSessionService.ts`, `src/server/machineRoutes.ts`, `src/departments/finished_good_inspections/routes.ts`, `public/hub.html` | stations, sessions, tickets, identity on posts, hub entry |
| Station (Pi) | `station/mes_session.py`, `station/auth.py`, `station/ui.py`, `station/app.py`, `station/remote_model.py`, `station/defect_proposal.py` | session client, ticket login, defect proposal, remote scoring |
| Laptop | `inference_service/server.py`, `inference_service/registry.py`, `scripts/windows/*.ps1` | model serving, load protection, backups |

## Language assignment (Python/Rust picker, defaults used)

No team skill or hardware facts were given, so the picker used its defaults (margin 5). No task has a speed, memory-safety or error-handling need that Python or torch cannot meet, so there is **no Rust**. The MES tasks are TypeScript because the MES codebase is TypeScript (outside the Python/Rust choice). Summary (13 tasks): 7 Python, 4 TypeScript, 1 PowerShell, 1 mixed Python and TypeScript test; 0 Rust; 0 Python + Rust module.

## Tasks

Every task follows the same steps: (1) write the failing tests named below, (2) run them and see them fail, (3) implement the interface, (4) run the tests and the full suite, (5) regenerate the code graph and `docs/CODE_INDEX.md`, then commit. Tasks are in build order.

### Phase A: identity and criteria in the MES (TypeScript, repo `aisi_mes`)

#### M1: Vision stations, sessions and one-time tickets (TypeScript)

**Files:** create `migrations/finished_good_inspections/0011_vision_sessions.sql`, `src/departments/finished_good_inspections/visionSessionService.ts`; modify `src/server/machineRoutes.ts`, `src/departments/finished_good_inspections/routes.ts`; test `tests/core/visionSessions.test.ts`.

**Interfaces (produces):**
- Tables `vision_stations(station_id, display_name, base_url, active)` and `fgi_station_sessions(id, user_id, station_id, ticket_hash, ticket_expires_at, redeemed_at, session_expires_at, ended_at, created_at)`.
- Browser (session auth, role `inspector` or `supervisor`): `POST /api/finished-good-inspections/vision/sessions` body `{stationId}` returns 201 `{redirectUrl}` where the ticket is in the URL fragment (`<base_url>/enter#ticket=<ticket>`), so it never reaches server logs or a Referer header. `POST .../vision/sessions/:id/end`.
- Machine (bearer token): `POST /api/machine/sessions/redeem` `{ticket}` returns 200 `{sessionId, username, roles, expiresAt}`; a second use returns 409. `GET /api/machine/sessions/:id` returns 200 `{active, username, roles, expiresAt}` and extends a sliding expiry, or 401. `POST /api/machine/sessions/:id/end`.
- Ticket lifetime 60 seconds, single use (engineering default). Session expiry slides and matches the MES idle logout (`public/idle-logout.js`, default 600000 ms = 10 minutes).

**Tests:** ticket redeems once (replay returns 409); expired ticket rejected; inactive station rejected; roles outside inspector/supervisor get 403; every create, redeem and end writes `audit_trail` through `writeAudited`; the raw ticket is never stored (only its hash).

#### M2: Identity on inspection events (TypeScript)

**Files:** modify `src/server/machineRoutes.ts`, `src/departments/finished_good_inspections/inspectionService.ts`, migration `0011` (column `inspector_user_id`); test `tests/core/machineInspectionSession.test.ts`.

**Interfaces:** `POST /api/machine/inspection-events` accepts header `Station-Session: <id>`. When `VISION_REQUIRE_SESSION=1` and `source=vision_station`, the header is required (401 without). The event stores `inspector_user_id`. A post is accepted if the part's capture time falls inside the session window (started to ended or expired, plus 5 minutes of clock-skew allowance, an engineering default; the time of the post itself is not limited), so a queued retry after logout still succeeds. An idempotent replay returns 200 without re-checking.

**Tests:** missing header returns 401 when required; unknown session 401; event outside the window rejected; retry after logout inside the window accepted; replay with the same key returns the original event; the audit row names both the machine and the inspector.

#### M3: Criteria with names (TypeScript)

**Files:** modify `src/server/machineRoutes.ts` (`GET /criteria/:partNumber`); test `tests/core/machineCriteria.test.ts`.

**Interfaces:** the response gains `name`, `description` and `active` per criterion, keeping the existing id fields (backward compatible). No cost or supplier data.

**Tests:** old fields unchanged; a part with no criteria returns an empty list; inactive criteria are marked, not dropped; unknown part returns 404.

#### M4: Hub entry "4-camera inspection" (TypeScript)

**Files:** modify `public/hub.html`; test by a route test that the link is shown only to `inspector` and `supervisor`.

**Interfaces:** a hub card posts to the M1 create route and opens `redirectUrl` in a new tab; error text shown on failure. Station base URLs come from `vision_stations`, set by an admin script (`npm run vision-station -- <id> <url>`).

**Tests:** card hidden for other roles; failure shows an error and opens nothing.

### Phase B: the station on the Pi (Python, repo `vision-station`)

#### S1: MES session client and posting header (Python)

**Files:** create `station/mes_session.py`; modify `station/mes_client.py`; test `tests/test_mes_session.py`.

**Interfaces:** `SessionInfo(session_id: str, username: str, roles: tuple[str, ...], expires_utc: str)`; `MesSessionClient(base_url, token, timeout_s).redeem(ticket) -> SessionInfo`, `.validate(session_id) -> SessionInfo | None`, `.end(session_id) -> None`. `MesClient.post_inspection(..., station_session: str | None)` sends `Station-Session`. Tokens never appear in messages or logs.

**Tests:** redeem success and 409; validate 401 returns `None`; network error raises `MesSessionError` without the token in its text; the header is sent on posts.

#### S2: Ticket login and roles in the station page (Python)

**Files:** modify `station/auth.py`, `station/ui.py`, `station/app.py`; test `tests/test_auth_mes.py`, `tests/test_ui_mes.py`.

**Interfaces:** config `auth: {"mode": "mes_ticket", "role_map": {"inspector": "operator", "supervisor": "supervisor"}}`. `GET /enter` serves a tiny page that reads `location.hash` and calls `POST /api/enter {ticket}`, which redeems it and sets an HttpOnly SameSite cookie. Roles outside `role_map` get no access. In this mode local username/password login is off. Logout and idle timeout call `.end`. The session is re-validated on each state poll; if the MES says it ended, the page shows "Log in again from the MES" and a part in progress is finished by the same person's re-login or becomes a FAULT.

**Tests:** ticket in the query string is rejected (fragment only); replayed ticket gives an error; unmapped role gets 403; expired session during WAIT_FLIP leads to the FAULT or re-login path and never to a post under another name; local login is refused in this mode.

#### S3: Remote model client (Python)

**Files:** create `station/remote_model.py`; modify `station/app.py` (`model.remote` config); test `tests/test_remote_model.py`.

**Interfaces:** `RemoteModel(base_url, token_env, timeout_s)` implements the existing `InferenceModel` (`version: str`, `predict(frame) -> ImageResult`) by calling the laptop service. `timeout_s` is `[Engineer to define]` after the measured laptop latency. Any timeout, HTTP error or malformed reply raises `StationError`, so the session goes to FAULT, never GOOD. The token comes from the environment variable only.

**Tests:** success maps to `ImageResult`; timeout, 5xx and bad JSON raise `StationError`; no token in error text; a REVIEW-range score still never turns green (existing safety tests pass unchanged).

#### S4: Defect proposal with pictures (Python)

**Files:** create `station/defect_proposal.py`; modify `station/ui.py`, `station/session.py`, `station/app.py`; test `tests/test_defect_proposal.py`.

**Interfaces:** `DefectProposal(criterion_id: str, camera_id: str, side: Side, note: str)`; `POST /api/part/defects` adds a proposal to the part being shown (VERDICT state). The final post sets `checklistResults[criterion_id]` to fail and attaches `defectPhoto_<criterion_id>` with the chosen picture (ML boxes drawn when present). Rules: criteria come from `GET /criteria/:part` (M3); a proposal on an ML-GOOD part makes the final verdict BAD (moves toward BAD only; turning BAD back to GOOD still needs the supervisor override that exists); the proposal is stored with the existing `OperatorLabel` flow (status `proposed`, approved by an Admin) so approved labels become labelled BAD examples for training (this also answers open decision D3 in the plan). Uses `OperatorLabel` as is; if its fields do not fit, stop and ask before touching `contracts.py`.

**Tests:** inactive or unknown criterion refused; part with no criteria refuses with a clear message; one photo per criterion; ML-GOOD plus a proposal posts BAD with the photo; the label is saved with status `proposed` and never reaches training until approved; REVIEW rules unchanged.

#### S5: Evidence transport to the laptop (Python)

**Files:** modify `station/store.py` (or the store module the code graph names), create `scripts/pi/sync_evidence.sh`; test `tests/test_evidence_sync.py`.

**Interfaces:** the station saves the 8 pictures and `part.json` on the Pi first (save before green), then a retrying sync copies the part folder to the laptop evidence store (the folder the trainer reads), and `archiveRef` is the final path on the laptop. Method: an SMB share on the laptop mounted on the Pi (decided; the laptop is Windows, so no SSH server is needed).

**Tests:** green is shown only after the local save; a failed sync leaves the part queued and retried, never lost and never marked synced; `archiveRef` is set only after a verified copy (sha256).

### Phase C: laptop services

#### L1: Inference service (Python)

**Files:** create `inference_service/__init__.py`, `inference_service/server.py`, `inference_service/registry.py`; test `tests/test_inference_service.py`.

**Interfaces:** `POST /v1/predict` (multipart: `part_number`, `camera_id`, `side`, image) returns `{score, defects[], model_version}`; `GET /v1/health` returns loaded models and GPU state. Bearer token from `INFERENCE_TOKEN`. `registry.get(part_number)` loads only the model approved in `ModelStore` (verifies sha256, 503 on tamper or missing, no fallback to an older version) and reloads when `current.json` changes. Binds to the LAN address `[Engineer to define]`.

**Tests:** unapproved part returns 503; tampered model returns 503; reload after a promote; a wrong token returns 401; a fake extractor returns the expected `ImageResult` fields; GPU out of memory returns 503 with a message, not a crash.

#### L2: One-laptop load protection and backups (PowerShell + docs)

**Files:** create `scripts/windows/backup_postgres.ps1`, update `scripts/windows/run_weekly_training.ps1` (training at below-normal priority) and `docs/weekly_training.md`; test `tests/test_scripts_weekly.py` additions.

**Interfaces:** nightly `pg_dump` of the MES database to `[Engineer to define: backup folder]`, with a restore check line in the runbook; the Sunday 06:00 training runs at below-normal priority and refuses to start when the inference service reports a request in progress; MES, PostgreSQL and the inference service start at boot as Windows scheduled tasks or services (the engineer picks per service).

**Tests:** scripts contain no password literal and no purge or promote call; priority flag present; the backup script fails loudly when `pg_dump` fails.

### Phase D: bring-up and proof

#### E1: End-to-end test (Python + TypeScript)

**Files:** create `tests/test_e2e_mes_flow.py`; add `tests/core/visionFlow.test.ts`.

**Tests:** fake MES with sessions and criteria, fake laptop service: ticket login, part search, inspect (8 images), propose a defect, post with `Station-Session` and the failed-criterion photo, idempotent retry; laptop down gives FAULT and no post; session ended mid-part never posts under another name.

#### E2: Pi setup guide and real-hardware checks (Python + docs)

**Files:** create `docs/pi_setup.md`, `tools/pi_check.py`; test `tests/test_pi_check.py` (fake cv2).

**Interfaces:** `docs/pi_setup.md` covers Raspberry Pi OS, Python, powered USB hub for 4 cameras, camera device paths (Linux uses V4L2, not DirectShow), systemd service for the station, and the LAN address and time sync. `python -m tools.pi_check` lists cameras, runs the hub test, measures the round trip to the inference service (per image and for 8) and prints the numbers; it decides nothing.

**Tests:** output lists each camera, a dead camera is reported by index, an unreachable laptop is reported as such.

## Decisions (JEV One System, decision first)

The owner asked me to decide for them from what I know about their job (quality work in an IATF 16949 plant, a Windows laptop that hosts the MES, one engineer running the project, Quality owning limits). Each row is decided; confidence says how sure I am; "revisit" says what would change it.

| # | Decision | Decided | Confidence | Revisit when |
|---|---|---|---|---|
| 1 | Where the model runs | Laptop GPU inference service, called from the Pi | 0.75 | `tools/pi_check` shows the Pi can score 8 images inside the operator wait limit `[Quality to define]` |
| 2 | Inspector identity | One-time ticket and MES session, no second login on the Pi | 0.90 | never expected |
| 3 | Human-proposed defect on an ML-GOOD part | Final verdict becomes BAD; only a supervisor can reverse it | 0.85 | Quality objects (the conservative direction for a quality plant) |
| 4 | Who approves proposed defect labels | The Admin, in the MES. Built after the first release (task M5, below); until then labels stay "proposed" and training ignores them | 0.75 | the Admin wants it earlier |
| 5 | Evidence storage | Laptop disk is the store the trainer reads; the Pi saves locally first, then copies over an SMB share. The owner deletes used GOOD pictures by hand after promotion (earlier decision) | 0.80 | disk fills faster than expected |
| 6 | LAN security | Pilot on the plant LAN without TLS; TLS (task H1) is required before the station is used outside the pilot. The MES session cookie is not marked secure today | 0.65 | the LAN is shared with guests or Wi-Fi: do H1 first |
| 7 | Session length, ticket life, clock skew | 10 minutes sliding (the MES idle logout), 60 seconds single use, 5 minutes skew. Engineering defaults, not quality limits | 0.80 | operators complain about relogins |
| 8 | Roles allowed on the Pi | `inspector` becomes operator, `supervisor` becomes supervisor; no other role | 0.85 | the plant wants Quality Engineers on the line |

Still `[Quality to define]`: the operator wait limit, `t_img`, `t_low`, `min_good`, `min_holdout`, `holdout_fraction`, defect codes. Still `[Engineer to define]` after measuring: the `RemoteModel` timeout, the laptop LAN address, the backup folder.

## Later (not in the first release)

- **M5 (TypeScript):** MES screen where the Admin approves or rejects proposed defect labels with a reason.
- **H1 (PowerShell/config):** TLS in front of the MES, the Pi page and the inference service on the plant LAN.

## Unproven

Everything here is a plan. The Pi has never run the station, the laptop inference service does not exist, and the ticket flow is untested. Real 4-camera capture on the Pi, USB power with 4 cameras, and the latency from the Pi to the laptop are all unmeasured.

## Self-review

- Spec coverage: login in MES (M1, M4, S2), part number choice (existing search; S2 role mapping), inspect (S3, L1), propose defect with pictures (M3, S4, M2), MES defect criteria per part (M3), Pi cameras (E2, S5), laptop as trainer and database server (L2, S5).
- Types match across tasks: `SessionInfo`, `Station-Session`, `criterion_id`, `DefectProposal`.
- Proportion: tasks give files, signatures and test assertions, not code.
