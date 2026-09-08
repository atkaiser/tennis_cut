from fractions import Fraction
from pathlib import Path

import pytest

from tennis_cut.contact_review import (
    ReviewedContact,
    ReviewedFrame,
    ReviewedVisualState,
)
from train_contact_detector.prototype_contact_bakeoff import (
    classify_selection,
    choose_review_threshold,
    contact_targets,
    honor_reviewed_states,
    listwise_contact_loss,
    select_training_indices,
    stable_player_crop,
)


def _frame(ordinal: int) -> ReviewedFrame:
    return ReviewedFrame(0, ordinal, ordinal, Fraction(1, 120))


def _contact(
    status: str, state_ordinals: tuple[tuple[int, ...], ...]
) -> ReviewedContact:
    states = tuple(
        ReviewedVisualState(tuple(_frame(ordinal) for ordinal in ordinals))
        for ordinals in state_ordinals
    )
    return ReviewedContact(
        video=Path("video.mp4"),
        annotation=Path("video.json"),
        family="family",
        split="train",
        suitability="in_scope",
        shot_index=3,
        labeled_timestamp=Fraction(1),
        shot_type="forehand",
        status=status,
        canonical_frame=None if not states else states[0].frames[0],
        visual_states=states,
    )


def test_single_state_targets_include_neighboring_visual_states() -> None:
    contact = _contact("single_state", ((12,),))

    targets = contact_targets(contact, ((10, 11), (12,), (13, 14)))

    assert targets == pytest.approx(
        {10: 0.075, 11: 0.075, 12: 0.7, 13: 0.075, 14: 0.075}
    )


def test_between_state_targets_split_mass_between_contact_states() -> None:
    contact = _contact("between_state", ((12, 13), (14,)))

    targets = contact_targets(contact, ((11,), (12, 13), (14,), (15,)))

    assert targets == pytest.approx({12: 0.25, 13: 0.25, 14: 0.5})


def test_unresolved_contact_has_no_training_targets() -> None:
    contact = _contact("unresolved", ())

    assert contact_targets(contact, ((10,), (11,), (12,))) == {}


def test_human_review_boundaries_override_automatic_duplicate_grouping() -> None:
    contact = _contact("single_state", ((11,),))

    states = honor_reviewed_states(contact, ((10, 11), (12, 13)))

    assert states == ((10,), (11,), (12, 13))


def test_stable_player_crop_uses_one_square_for_all_frames() -> None:
    boxes = [(400, 200, 600, 800), (410, 190, 610, 810), (390, 210, 590, 790)]

    crop = stable_player_crop(boxes, image_width=1920, image_height=1080)

    assert crop == (50, 50, 950, 950)


def test_stable_player_crop_clamps_without_changing_shape() -> None:
    boxes = [(20, 100, 220, 700)]

    crop = stable_player_crop(boxes, image_width=1280, image_height=720)

    assert crop == (0, 0, 900, 720)


def test_training_sample_keeps_targets_and_spans_negative_window() -> None:
    records = [
        {"target": 0.0, "visual_state": state}
        for state in range(10)
    ]
    records[4]["target"] = 0.15
    records[5]["target"] = 0.7
    records[6]["target"] = 0.15

    selected = select_training_indices(records, negative_count=4, seed=12)

    assert {4, 5, 6}.issubset(selected)
    assert len(selected) == 7
    assert min(selected) == 0
    assert max(selected) == 9


def test_listwise_loss_uses_soft_target_distribution() -> None:
    import torch

    logits = torch.tensor([0.0, 1.0, 2.0])
    targets = torch.tensor([0.15, 0.7, 0.15])

    loss = listwise_contact_loss(logits, targets)

    expected = -(targets * torch.log_softmax(logits, dim=0)).sum()
    assert float(loss) == pytest.approx(float(expected))


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"acceptable": True, "target": 0.7}, "exact"),
        ({"acceptable": False, "target": 0.15}, "tolerated"),
        ({"acceptable": False, "target": 0.0}, "failed"),
    ],
)
def test_report_classifies_selected_frame(record: dict, expected: str) -> None:
    assert classify_selection(record) == expected


def test_confidence_threshold_prefers_recall_without_accepting_unresolved() -> None:
    entries = [
        {"confidence": 0.8, "correct": True},
        {"confidence": 0.5, "correct": True},
        {"confidence": 0.3, "correct": False},
    ]

    assert choose_review_threshold(entries) == pytest.approx(0.5)
