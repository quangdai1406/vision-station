"""Tests for human defect proposals (plan S3): rules, session, MES post, storage, UI routes, page.

Fakes only (see tests/test_session.py and tests/test_ui.py); no network, no sleeping.
"""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from station import ui as ui_module
from station.contracts import InvalidTransition, Side, State, Verdict
from station.defect_proposal import (DefectProposal, ProposalRecord, add_proposal, check_proposal,
                                     remove_proposal)
from station.mes_client import MappingError, MesConfig, build_request
from station.storage import Store
from station.ui import _PAGE
from tests import test_evidence_sync as tes
from tests import test_ui as tui
from tests.test_session import Harness
from tests.test_ui import ADM_PW, ADM_USER, OP_PW, OP_USER, SUP_PW, SUP_USER, FakeSession

TS = "2026-10-23T08:00:00Z"
CRITERIA = ["C-1", "C-2"]


def _row(criterion="C-1", camera="C1", side=Side.TOP, by="anna", note=""):
    return ProposalRecord(criterion_id=criterion, camera_id=camera, side=side, note=note,
                          proposed_by=by, proposed_at=TS)


def _with_proposal_store(h):
    """The test FakeStore predates defect_proposals: accept the keyword like station.storage does."""
    orig = h.store.save_part

    def save(record, defect_proposals=None):
        h.store.proposal_saves = getattr(h.store, "proposal_saves", []) + [defect_proposals]
        return orig(record)

    h.store.save_part = save
    return h


def _bad_harness():
    """A part the model judges BAD, shown in VERDICT with its 8 pictures."""
    h = _with_proposal_store(Harness(fuse_verdict=Verdict.BAD))
    h.complete_part(0.0)
    assert h.session.state is State.VERDICT
    return h


class ProposalRuleTests(unittest.TestCase):
    def test_active_criterion_and_a_picture_of_the_part_are_accepted(self):
        p = DefectProposal("C-1", "C2", Side.BOTTOM, "small chip")
        self.assertIs(check_proposal(p, criteria_ids=CRITERIA, image_keys=[("C2", Side.BOTTOM)]), p)

    def test_unknown_or_inactive_criterion_is_refused(self):
        # inactive criteria are not in criteria_ids (build_part_config leaves them out)
        with self.assertRaisesRegex(ValueError, "not an active criterion"):
            check_proposal(DefectProposal("C-9", "C1", Side.TOP), criteria_ids=CRITERIA,
                           image_keys=[("C1", Side.TOP)])

    def test_part_without_criteria_refuses_with_a_clear_message(self):
        with self.assertRaisesRegex(ValueError, "no inspection criteria"):
            check_proposal(DefectProposal("C-1", "C1", Side.TOP), criteria_ids=[],
                           image_keys=[("C1", Side.TOP)])

    def test_picture_must_be_one_of_the_parts_images(self):
        with self.assertRaisesRegex(ValueError, "8 images"):
            check_proposal(DefectProposal("C-1", "C9", Side.TOP), criteria_ids=CRITERIA,
                           image_keys=[("C1", Side.TOP)])

    def test_note_is_at_most_500_characters(self):
        check_proposal(DefectProposal("C-1", "C1", Side.TOP, "x" * 500), criteria_ids=CRITERIA,
                       image_keys=[("C1", Side.TOP)])
        with self.assertRaisesRegex(ValueError, "500"):
            check_proposal(DefectProposal("C-1", "C1", Side.TOP, "x" * 501), criteria_ids=CRITERIA,
                           image_keys=[("C1", Side.TOP)])

    def test_one_proposal_per_criterion_second_replaces_first(self):
        rows, replaced = add_proposal([], _row("C-1", "C1"))
        self.assertFalse(replaced)
        rows, replaced = add_proposal(rows, _row("C-2", "C3"))
        self.assertFalse(replaced)
        rows, replaced = add_proposal(rows, _row("C-1", "C4", Side.BOTTOM))
        self.assertTrue(replaced)
        self.assertEqual([(r.criterion_id, r.camera_id) for r in rows], [("C-2", "C3"), ("C-1", "C4")])

    def test_remove_proposal(self):
        rows, removed = remove_proposal([_row("C-1")], "C-1")
        self.assertEqual(rows, [])
        self.assertTrue(removed)
        rows, removed = remove_proposal([_row("C-1")], "C-2")
        self.assertFalse(removed)

    def test_as_dict_is_json_safe(self):
        data = json.loads(json.dumps(_row("C-1", "C2", Side.BOTTOM, note="n").as_dict()))
        self.assertEqual(data, {"criterion_id": "C-1", "camera_id": "C2", "side": "BOTTOM", "note": "n",
                                "proposed_by": "anna", "proposed_at": TS})


class SessionProposalTests(unittest.TestCase):
    def test_proposal_is_refused_outside_verdict_and_review(self):
        h = Harness()
        with self.assertRaises(InvalidTransition):
            h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)
        h.press_at(0.0)  # WAIT_FLIP
        with self.assertRaises(InvalidTransition):
            h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)

    def test_session_checks_the_criterion_and_the_inspector(self):
        h = _bad_harness()
        with self.assertRaisesRegex(ValueError, "not an active criterion"):
            h.session.propose_defect(DefectProposal("C-9", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)
        with self.assertRaisesRegex(ValueError, "no inspection criteria"):
            h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=[])
        with self.assertRaisesRegex(ValueError, "by is required"):
            h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by=" ", criteria_ids=CRITERIA)

    def test_proposal_is_stored_with_the_session_inspector_and_logged(self):
        h = _bad_harness()
        replaced = h.session.propose_defect(DefectProposal("C-1", "C2", Side.BOTTOM, "chip"),
                                            by="anna", criteria_ids=CRITERIA)
        self.assertFalse(replaced)
        (row,) = h.session.snapshot()["defect_proposals"]
        self.assertEqual(row, {"criterion_id": "C-1", "camera_id": "C2", "side": "BOTTOM", "note": "chip",
                               "proposed_by": "anna", "proposed_at": TS})
        kinds = h.store.kinds()
        self.assertIn("defect_proposed", kinds)

    def test_second_proposal_for_a_criterion_replaces_the_first(self):
        h = _bad_harness()
        h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)
        replaced = h.session.propose_defect(DefectProposal("C-1", "C4", Side.BOTTOM), by="anna",
                                            criteria_ids=CRITERIA)
        self.assertTrue(replaced)
        rows = h.session.snapshot()["defect_proposals"]
        self.assertEqual([(r["criterion_id"], r["camera_id"]) for r in rows], [("C-1", "C4")])

    def test_remove_proposal(self):
        h = _bad_harness()
        h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)
        self.assertTrue(h.session.remove_defect("C-1"))
        self.assertFalse(h.session.remove_defect("C-1"))
        self.assertEqual(h.session.snapshot()["defect_proposals"], [])

    def test_bad_part_with_a_proposal_saves_with_the_proposal_and_stays_bad(self):
        h = _bad_harness()
        h.session.propose_defect(DefectProposal("C-2", "C3", Side.BOTTOM), by="anna", criteria_ids=CRITERIA)
        h.session.ack_alarm()
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.store.saved[-1].final_verdict, Verdict.BAD)
        saved_rows = h.store.proposal_saves[-1]
        self.assertEqual(saved_rows[0]["criterion_id"], "C-2")
        self.assertEqual(saved_rows[0]["proposed_by"], "anna")
        rows = h.session.proposals_for(h.store.saved[-1].part_id)
        self.assertEqual([r.criterion_id for r in rows], ["C-2"])
        self.assertEqual(h.session.proposals_for(h.store.saved[-1].part_id), [])  # handed over once

    def test_a_bad_part_cannot_be_flipped_to_good_by_a_proposal_path(self):
        h = _bad_harness()
        h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)
        h.session.override(by="sup", new_verdict=Verdict.GOOD, reason="looks fine")
        self.assertEqual(h.store.saved[-1].final_verdict, Verdict.BAD)
        self.assertIsNone(h.store.saved[-1].override)
        self.assertEqual(h.store.saved[-1].disposition, "quarantine")
        self.assertEqual(h.outputs.calls[-2:], ["show_bad", "alarm_off"])
        self.assertIn("defect_proposal_forces_bad", h.store.kinds())
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})

    def test_review_part_resolved_good_with_a_proposal_is_bad(self):
        h = _with_proposal_store(Harness(fuse_verdict=Verdict.REVIEW))
        h.complete_part(0.0)
        self.assertEqual(h.session.state, State.SUPERVISOR_REVIEW)
        h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)
        h.session.resolve_review(by="sup", new_verdict=Verdict.GOOD, reason="ok")
        self.assertEqual(h.store.saved[-1].final_verdict, Verdict.BAD)
        self.assertEqual(h.store.saved[-1].machine_verdict, Verdict.REVIEW)

    def test_review_part_without_proposals_keeps_the_review_rules(self):
        h = Harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        h.session.resolve_review(by="sup", new_verdict=Verdict.GOOD, reason="ok")
        self.assertEqual(h.store.saved[-1].final_verdict, Verdict.GOOD)
        self.assertEqual(h.outputs.calls[-1], "show_good")

    def test_proposals_are_cleared_when_the_next_part_starts(self):
        h = _bad_harness()
        h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)
        h.session.ack_alarm()
        self.assertEqual(h.session.snapshot()["defect_proposals"], [])
        h.fuse.verdict = Verdict.GOOD  # the next part is judged GOOD
        h.press_at(5.0)  # the next part's TOP press
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        self.assertEqual(h.session.snapshot()["defect_proposals"], [])
        h.press_at(6.0)  # its BOTTOM press: GOOD, saved without proposals
        self.assertEqual(h.store.saved[-1].final_verdict, Verdict.GOOD)
        self.assertEqual(h.session.proposals_for(h.store.saved[-1].part_id), [])
        self.assertEqual(h.session.snapshot()["defect_proposals"], [])

    def test_good_part_has_no_proposal_window(self):
        # Machine GOOD parts are saved and lit at once (save before green), so a proposal is refused.
        h = Harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        self.assertEqual(h.session.state, State.IDLE)
        with self.assertRaises(InvalidTransition):
            h.session.propose_defect(DefectProposal("C-1", "C1", Side.TOP), by="anna", criteria_ids=CRITERIA)


class MesPostTests(unittest.TestCase):
    def _record(self):
        h = _bad_harness()
        h.session.ack_alarm()
        return h.store.saved[-1]

    def _cfg(self):
        return MesConfig(base_url="http://mes.test", token="t", criteria_ids=CRITERIA,
                         class_to_criterion={"scratch": "C-1"})

    def test_proposal_fails_its_criterion_and_attaches_the_chosen_picture(self):
        record = self._record()
        rows = [_row("C-2", "C3", Side.BOTTOM, by="anna")]
        fields, files = build_request(record, self._cfg(), archive_ref="arch", proposals=rows)
        self.assertEqual(json.loads(fields["checklistResults"]), {"C-1": "fail", "C-2": "fail"})
        self.assertEqual(fields["verdict"], "BAD")
        self.assertIn("C-2", fields["rejectionReason"])
        photos = {name: data for name, _fn, data in files}
        self.assertEqual(photos["defectPhoto_C-2"], b"BOTTOM-C3")
        self.assertIn("defectPhoto_C-1", photos)  # the model's own photo for C-1 stays

    def test_proposal_picture_replaces_the_model_picture_for_the_same_criterion(self):
        record = self._record()
        rows = [_row("C-1", "C2", Side.TOP, by="anna")]
        _fields, files = build_request(record, self._cfg(), archive_ref="arch", proposals=rows)
        photos = {name: (fn, data) for name, fn, data in files}
        self.assertEqual(photos["defectPhoto_C-1"], ("C2_TOP.png", b"TOP-C2"))

    def test_proposal_on_an_unknown_criterion_is_a_mapping_error(self):
        record = self._record()
        with self.assertRaises(MappingError):
            build_request(record, self._cfg(), archive_ref="arch", proposals=[_row("C-9")])

    def test_a_good_post_never_carries_a_proposal(self):
        record = self._record()
        record.final_verdict = Verdict.GOOD
        with self.assertRaisesRegex(ValueError, "must be posted as BAD"):
            build_request(record, self._cfg(), archive_ref="arch", proposals=[_row("C-2")])

    def test_no_proposals_means_the_post_is_unchanged(self):
        record = self._record()
        plain = build_request(record, self._cfg(), archive_ref="arch")
        self.assertEqual(plain, build_request(record, self._cfg(), archive_ref="arch", proposals=()))


class StorageProposalTests(unittest.TestCase):
    def test_proposals_are_in_part_json_and_never_an_operator_label(self):
        h = _bad_harness()
        h.session.ack_alarm()
        record = h.store.saved[-1]
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(pathlib.Path(tmp), site="SITE", line="L1")
            rows = [_row("C-2", "C3", Side.BOTTOM, note="chip").as_dict()]
            part_dir = store.save_part(record, defect_proposals=rows)
            doc = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["defect_proposals"], rows)
        self.assertIsNone(doc["operator_label"])  # training reads operator_label only: proposals are not labels
        self.assertEqual(doc["final_verdict"], "BAD")

    def test_no_proposals_key_when_there_are_none(self):
        h = _bad_harness()
        h.session.ack_alarm()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(pathlib.Path(tmp), site="SITE", line="L1")
            part_dir = store.save_part(h.store.saved[-1])
            doc = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
        self.assertNotIn("defect_proposals", doc)


class EvidencePlannedDestinationTests(tes.EvidenceSyncTestCase):
    def test_planned_destination_is_the_path_sync_part_returns(self):
        from station import evidence_sync as es
        planned = es.planned_destination(self.part, self.dest)
        self.assertEqual(planned, str(self.final_path()))
        self.assertEqual(es.sync_part(self.part, self.dest), planned)
        self.assertTrue(self.final_path().is_dir())

    def test_planned_destination_writes_nothing(self):
        from station import evidence_sync as es
        before = tes.snapshot(self.dest)
        es.planned_destination(self.part, self.dest)
        self.assertEqual(tes.snapshot(self.dest), before)

    def test_planned_destination_refuses_a_folder_outside_the_store_layout(self):
        from station import evidence_sync as es
        with self.assertRaises(es.SyncError):
            es.planned_destination(pathlib.Path(self.dest) / "x", self.dest)


class DefectSession(FakeSession):
    """FakeSession with the proposal calls the UI routes use."""

    def __init__(self):
        super().__init__()
        self.proposed = []
        self.removed = []

    def part_criteria(self):
        return [{"id": "C-1", "name": "Scratch", "description": "Long mark", "criteria_ids": ["x"]}]

    def propose_defect(self, by, criterion_id, camera_id, side, note=""):
        self.proposed.append({"by": by, "criterion_id": criterion_id, "camera_id": camera_id,
                              "side": side, "note": note})
        return {"replaced": False}

    def remove_defect(self, criterion_id):
        self.removed.append(criterion_id)
        return {"removed": True}


class DefectRouteTests(tui.UiTestCase):
    def setUp(self):
        from station.auth import SessionManager, UserStore, hash_password
        from station.ui import make_server
        import threading
        self.session = DefectSession()
        self.users = UserStore({
            OP_USER: {"password_hash": hash_password(OP_PW), "role": "operator"},
            SUP_USER: {"password_hash": hash_password(SUP_PW), "role": "supervisor"},
            ADM_USER: {"password_hash": hash_password(ADM_PW), "role": "admin"},
        })
        self.sessions = SessionManager(timeout_s=600)
        self.server = make_server(self.session, self.users, self.sessions, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def test_operator_proposes_and_the_inspector_is_the_login_not_the_body(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("POST", "/api/part/defects", {
            "criterion_id": "C-1", "camera_id": "C2", "side": "BOTTOM", "note": "chip",
            "by": "someone-else"}, token=token)
        self.assertEqual(reply.status, 200, reply.body)
        self.assertEqual(self.js(reply), {"ok": True, "replaced": False})
        self.assertEqual(self.session.proposed, [{"by": OP_USER, "criterion_id": "C-1", "camera_id": "C2",
                                                  "side": "BOTTOM", "note": "chip"}])

    def test_bad_input_is_refused(self):
        token = self.login(OP_USER, OP_PW)
        cases = [
            {"criterion_id": "", "camera_id": "C1", "side": "TOP"},
            {"criterion_id": "C-1", "camera_id": "C9", "side": "TOP"},
            {"criterion_id": "C-1", "camera_id": "C1", "side": "LEFT"},
            {"criterion_id": "C-1", "camera_id": "C1", "side": "TOP", "note": "x" * 501},
        ]
        for body in cases:
            with self.subTest(body=body):
                reply = self.call("POST", "/api/part/defects", body, token=token)
                self.assertEqual(reply.status, 400, reply.body)
        self.assertEqual(self.session.proposed, [])

    def test_no_login_is_401(self):
        reply = self.call("POST", "/api/part/defects", {"criterion_id": "C-1", "camera_id": "C1",
                                                       "side": "TOP"})
        self.assertEqual(reply.status, 401)

    def test_role_without_propose_defect_gets_403(self):
        token = self.login(SUP_USER, SUP_PW)
        with mock.patch.object(ui_module, "role_allows", lambda role, action: False):
            reply = self.call("POST", "/api/part/defects", {"criterion_id": "C-1", "camera_id": "C1",
                                                           "side": "TOP"}, token=token)
        self.assertEqual(reply.status, 403)
        self.assertEqual(self.session.proposed, [])

    def test_remove_route(self):
        token = self.login(SUP_USER, SUP_PW)
        reply = self.call("POST", "/api/part/defects/remove", {"criterion_id": "C-1"}, token=token)
        self.assertEqual(reply.status, 200, reply.body)
        self.assertEqual(self.js(reply), {"ok": True, "removed": True})
        self.assertEqual(self.session.removed, ["C-1"])

    def test_criteria_route_returns_only_name_and_description(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("GET", "/api/part/criteria", token=token)
        self.assertEqual(reply.status, 200)
        self.assertEqual(self.js(reply), {"criteria": [{"id": "C-1", "name": "Scratch",
                                                        "description": "Long mark"}]})

    def test_operator_can_see_the_auth_action_for_proposals(self):
        token = self.login(OP_USER, OP_PW)
        state = self.js(self.call("GET", "/api/state", token=token))
        self.assertIn("propose_defect", state["auth"]["actions"])


class PagePropositionScanTests(unittest.TestCase):
    def _block(self):
        start = _PAGE.index("defect proposals (shown while")
        end = _PAGE.index("part picker (only when parts_mode")
        return _PAGE[start:end]

    def test_defect_script_uses_textcontent_only(self):
        block = self._block()
        self.assertIn("textContent", block)
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            self.assertNotIn(bad, block)

    def test_page_has_the_defect_panel_and_hooks(self):
        for needle in ('id="defect-panel"', 'id="defect-criteria"', 'id="defect-pictures"',
                       'id="defect-pending"', 'id="defect-add"', "renderDefects(s);"):
            self.assertIn(needle, _PAGE)


if __name__ == "__main__":
    unittest.main()
