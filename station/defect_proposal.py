from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from station.contracts import Side

MAX_NOTE_CHARS = 500


@dataclass(frozen=True)
class DefectProposal:
    criterion_id: str
    camera_id: str
    side: Side
    note: str = ""


@dataclass(frozen=True)
class ProposalRecord:
    criterion_id: str
    camera_id: str
    side: Side
    note: str
    proposed_by: str
    proposed_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "camera_id": self.camera_id,
            "side": self.side.value,
            "note": self.note,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at,
        }


def check_proposal(proposal: DefectProposal, *, criteria_ids: Iterable[str],
                   image_keys: Iterable[tuple[str, Side]]) -> DefectProposal:
    criteria = list(criteria_ids)
    if not criteria:
        raise ValueError("this part has no inspection criteria, so a defect cannot be proposed for it")
    if not isinstance(proposal.criterion_id, str) or proposal.criterion_id not in criteria:
        raise ValueError("the criterion is not an active criterion of this part")
    if not isinstance(proposal.side, Side):
        raise ValueError("side must be TOP or BOTTOM")
    if (proposal.camera_id, proposal.side) not in set(image_keys):
        raise ValueError("the picture must be one of the part's 8 images")
    if not isinstance(proposal.note, str) or len(proposal.note) > MAX_NOTE_CHARS:
        raise ValueError(f"the note must be {MAX_NOTE_CHARS} characters or fewer")
    return proposal


def add_proposal(rows: Sequence[ProposalRecord], row: ProposalRecord) -> tuple[list[ProposalRecord], bool]:
    replaced = any(r.criterion_id == row.criterion_id for r in rows)
    kept = [r for r in rows if r.criterion_id != row.criterion_id]
    return kept + [row], replaced


def remove_proposal(rows: Sequence[ProposalRecord], criterion_id: str) -> tuple[list[ProposalRecord], bool]:
    kept = [r for r in rows if r.criterion_id != criterion_id]
    return kept, len(kept) != len(rows)
