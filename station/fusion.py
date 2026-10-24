"""Per-image classification and per-part fusion.

Thresholds come in as an argument everywhere; nothing here hardcodes a value.
Standard library only. Types come from station/contracts.py.
"""
from __future__ import annotations

from typing import List, Sequence

from station.contracts import (
    IMAGES_PER_PART,
    Defect,
    ImageResult,
    PartVerdict,
    Thresholds,
    Verdict,
)


def classify_image(result: ImageResult, th: Thresholds) -> Verdict:
    """score >= t_img -> BAD; score >= t_low -> REVIEW; otherwise GOOD."""
    if result.score >= th.t_img:
        return Verdict.BAD
    if result.score >= th.t_low:
        return Verdict.REVIEW
    return Verdict.GOOD


def fuse(results: Sequence[ImageResult], th: Thresholds) -> PartVerdict:
    """Combine the 8 image results of one part into one verdict.

    Any BAD image makes the part BAD. Otherwise any REVIEW image makes the
    part REVIEW. Only a part with no BAD and no REVIEW images is GOOD.
    """
    items = tuple(results)
    if len(items) != IMAGES_PER_PART:
        raise ValueError(
            f"fuse needs exactly {IMAGES_PER_PART} results, got {len(items)}"
        )

    classes = [classify_image(r, th) for r in items]

    if Verdict.BAD in classes:
        verdict = Verdict.BAD
    elif Verdict.REVIEW in classes:
        verdict = Verdict.REVIEW
    else:
        verdict = Verdict.GOOD

    if verdict is Verdict.GOOD:
        reason = "no image above T_low"
    else:
        deciding = [r for r, c in zip(items, classes) if c is verdict]
        # max() returns the first image with the highest score, so ties
        # resolve to capture order.
        worst = max(deciding, key=lambda r: r.score)
        reason = (
            f"{verdict.value}: worst {worst.camera_id} {worst.side.value} "
            f"score {worst.score:.3f}"
        )
        found: List[str] = []
        for r in deciding:
            for d in r.defects:
                if d.defect_class not in found:
                    found.append(d.defect_class)
        if found:
            reason += f"; defects: {', '.join(found)}"

    return PartVerdict(verdict=verdict, results=items, reason=reason)


def worst_defects(results: Sequence[ImageResult]) -> List[Defect]:
    """Every defect from every result, sorted by score descending."""
    all_defects: List[Defect] = [d for r in results for d in r.defects]
    all_defects.sort(key=lambda d: d.score, reverse=True)
    return all_defects
