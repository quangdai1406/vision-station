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
    if result.score >= th.t_img:
        return Verdict.BAD
    if result.score >= th.t_low:
        return Verdict.REVIEW
    return Verdict.GOOD


def fuse(
    results: Sequence[ImageResult],
    th: Thresholds,
    expected: int = IMAGES_PER_PART,
) -> PartVerdict:
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise ValueError(f"expected must be an int, got {expected!r}")
    if expected < 2 or expected % 2 != 0:
        raise ValueError(f"expected must be an even number >= 2, got {expected}")
    items = tuple(results)
    if len(items) != expected:
        raise ValueError(
            f"fuse needs exactly {expected} results, got {len(items)}"
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
    all_defects: List[Defect] = [d for r in results for d in r.defects]
    all_defects.sort(key=lambda d: d.score, reverse=True)
    return all_defects
