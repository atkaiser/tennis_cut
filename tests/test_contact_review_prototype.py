import signal
from collections import Counter
from pathlib import Path

import pytest

from label_videos import prototype_contact_review


def test_finalize_reviewed_splits_scales_past_small_pilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = ("single_state", "between_state", "unresolved")
    annotations = {}
    videos = []
    for family_index in range(11):
        annotation = Path(f"family-{family_index}.json")
        annotations[annotation] = {
            "contact_review_metadata": {"side_view_suitability": "in_scope"},
            "shots": [
                {"contact_review": {"status": status}} for status in statuses
            ],
        }
        videos.append(
            {
                "annotation": str(annotation),
                "family": f"family-{family_index}",
                "shots": [{"shot_index": index} for index in range(len(statuses))],
            }
        )
    monkeypatch.setattr(
        prototype_contact_review,
        "load_annotation",
        lambda path: annotations[path],
    )

    def timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("split assignment exceeded one second")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1)
    try:
        assignments = prototype_contact_review.finalize_reviewed_splits(
            {"videos": videos}
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

    assert set(assignments) == {video["family"] for video in videos}
    assert Counter(assignments.values()) == {
        "train": 8,
        "validation": 1,
        "test": 2,
    }
