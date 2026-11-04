# Plan: safe manual deletion after weekly training (2026-11-04)

Decisions from the owner: they delete used pictures by hand; GOOD pictures may be deleted; the
Admin approves (promotes) models.

Problem: the weekly run starts from the approved model and reads only pictures newer than it. A
picture deleted before its candidate is promoted is lost for good.

Tasks (all Python):
1. `trainer/purge.py`: refuse unless the candidate is on the approved history. Test first.
2. Docs: runbook section 7 (manual deletion after promotion), plan decisions D4 and section 11.
3. Code index and code graph regenerated.

JEV One System: choice made by the owner; no automatic deletion; refusal is the safe default.
