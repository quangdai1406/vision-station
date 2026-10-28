# Plan: bring the station to the company and connect to the AISI MES (2026-10-29)

Builds on `docs/26_1026_PLAN_office_day.md` and `docs/office_checklist.md`. It adds what changed since then:
Taking-data scans, label approval, and a promoted HOME TRIAL model. Numbers that belong to Quality are
`[Quality to define]`. Nothing here has been run against the real MES.

## Ground rules
1. **Start in collect mode.** Nothing is posted to the MES in collect mode (REVIEW is never auto-posted).
   Move to model mode (`parts.mode: "mes"` with a promoted model) only after talking to the MES administrator:
   model mode sends real inspection events to a live production MES.
2. **Do not carry the HomeToy model or home data into the company.** It is a 1-toy HOME TRIAL model with
   placeholder thresholds.
3. **Make a new company config:** new site, line and station IDs, the real part number, new logins. Do not
   reuse the home users or the `demo_*` logins.
4. **The MES token goes only in an environment variable** (`mes.token_env`). Never in a file, log or screenshot.
5. **Keep the UI on 127.0.0.1** (no TLS, no login rate limit).
6. **Do not move the cameras or the light.** If they move, repeat the 20-part collect check.

## Before leaving (at home, about 15 minutes)
| # | Task | Done when |
|---|---|---|
| 1 | `git status` clean on `main`; decide whether to push (nothing pushed since `115c2a4`) | Clean tree, push decision made |
| 2 | Full tests and `gen_codegraph_doc.py --check` | Both pass |
| 3 | `python -m tools.capture_speed --devices 0 2 4 6 --width 1280 --height 720 --rounds 10 --pause 1.0 --mjpg --threaded --flush 1` | 0 failures |
| 4 | Pack: Pi, power, 4 cameras and hub, light, LAN cable, USB drive, toy parts | Packed |
| 5 | Make `config.company.json` from `docs/home_trial/config.home.example.json`, placeholders still marked | File ready (not committed) |

## At the company
| # | Step | Owner | Verify |
|---|---|---|---|
| 1 | Network: get the MES address and port; `ping` and `curl` from the Pi; ask IT about firewall and proxy | You, IT | Pi gets an HTTP code from the MES |
| 2 | Token: ask the MES administrator for a machine token; export it in the shell that starts the station | MES admin | Part search returns real parts; token in no file |
| 3 | Test target: ask whether the MES has a test line, test part or sandbox. If not, stay in collect mode and post nothing | You, MES admin | Written answer |
| 4 | Config: `mes.base_url`, `mes.token_env`, `parts.mode: "mes"`; create real logins per role with `users add` | Claude | Page loads; each role sees the right actions |
| 5 | Part and criteria: search one real part number; run `tools.criteria`; fill `mes.criteria_ids` and `mes.class_to_criterion` | Claude, Quality | Mapping complete for that part |
| 6 | Collect REAL GOOD parts in collect mode: many different parts, not one repeated | You | 8 images each; counts as Quality sets (`min_good`, `min_holdout`, `holdout_fraction`: `[Quality to define]`) |
| 7 | Collect BAD parts per defect type; operator proposes the label, admin makes it official | You, admin | Official labels matching MES criteria |
| 8 | Taking-data scans (supervisor), approved by the admin | You | Scans are `official` |
| 9 | Unplug and replug one camera during a run | You, Claude watching | The station names the camera and recovers |
| 10 | Export: `tools.export_evidence` dry run first, then for real; read `export_report.json` | You | Counts match, exit code 0 |
| 11 | Laptop training setup per `docs/weekly_training.md`, then `selftest`; first `trainer weekly` only if there is enough data | You, Claude | GPU shown; candidate built |

## Not for this day
- Model mode against the live MES, until Quality approves a model and `t_low`/`t_img`. Thresholds come from
  labelled held-out scores through `tools/thresholds.py` and Quality's written approval `[Quality to define]`.
- Promotion of any model: needs a named person and Quality.
- SharePoint transfer (D-C): postponed until the merge into the MES.
- Alarm and light wiring (`serial_relay` channel map, acknowledge method).
- Quality's acceptance of MJPG pictures as evidence.

## Risks to watch
- Network or firewall blocks the MES. Fallback: collect mode works offline.
- The real MES may differ from the fake MES. Part search and criteria formats mirror the integration doc
  (`docs/vision-station-integration.md` in the AISI_MES repo); nothing is tested against the real MES.
- A saved part whose MES enqueue failed is not re-queued automatically. Look for `mes_enqueue_failed` events.
- After a station restart the part's own MES criteria are not reloaded until the part is selected again.
- The 60 s flip window; an override works only before the alarm is acknowledged (found in the home demo).
- Small-defect detection at 360x640 is unmeasured. Check it on the real labelled pictures from steps 6 to 8.

## Open questions for the owner
1. Does the MES have a test line, test part or sandbox, or is it production only? This decides whether anything
   can be posted.
2. Push the commits to GitHub before leaving?
