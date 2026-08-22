from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
import unittest

from tennis_cut.comparison.planning import (
    COMPARISON_POLICY,
    ArtifactRequest,
    ComparisonSource,
    PlayerObservation,
    Rectangle,
    SelectedComparisonWindows,
    UnrepresentableTimeline,
    build_render_plan,
    prepare_source_window,
    select_comparison_windows,
)
from tennis_cut.comparison.pro_selection import (
    DecodedFrame,
    InspectedMedia,
    ProSelection,
)
from tennis_cut.swing_detection import DetectedSwing


def source(
    name: str,
    *,
    first_pts: int,
    last_pts: int,
    time_base: Fraction,
    width: int = 1920,
    height: int = 1080,
) -> ComparisonSource:
    return ComparisonSource(
        path=Path(name),
        width=width,
        height=height,
        inspected_media=InspectedMedia(
            frames=tuple(
                DecodedFrame(0, ordinal, pts, time_base)
                for ordinal, pts in enumerate(range(first_pts, last_pts + 1))
            )
        ),
    )


class ComparisonPolicyTests(unittest.TestCase):
    def test_owns_all_fixed_comparison_decisions_as_immutable_data(self) -> None:
        self.assertEqual(COMPARISON_POLICY.pre_contact, Fraction(6, 5))
        self.assertEqual(COMPARISON_POLICY.post_contact, Fraction(7, 10))
        self.assertEqual(COMPARISON_POLICY.crop_margin, Fraction(1, 4))
        self.assertEqual(COMPARISON_POLICY.panel_aspect, (8, 9))
        self.assertEqual(COMPARISON_POLICY.output_size, (1280, 720))
        self.assertEqual(COMPARISON_POLICY.user_panel_origin, (0, 0))
        self.assertEqual(COMPARISON_POLICY.pro_panel_origin, (640, 0))
        self.assertEqual(
            COMPARISON_POLICY.supported_shot_types,
            frozenset({"forehand", "backhand", "volley", "serve"}),
        )
        with self.assertRaises(FrozenInstanceError):
            COMPARISON_POLICY.pre_contact = Fraction(1)  # ty: ignore[invalid-assignment]


class SelectComparisonWindowsTests(unittest.TestCase):
    def test_selects_exact_normalized_windows_in_accepted_swing_order(self) -> None:
        user = source(
            "user.mov", first_pts=100, last_pts=160, time_base=Fraction(1, 10)
        )
        pro = source(
            "pro.mov", first_pts=800, last_pts=1120, time_base=Fraction(1, 40)
        )
        pro_contact = pro.inspected_media.frames[200]
        selection = ProSelection(pro.path, pro_contact, "forehand")
        swings = (
            DetectedSwing(0, Fraction(23, 2), "backhand"),
            DetectedSwing(1, Fraction(12), "forehand"),
            DetectedSwing(2, Fraction(29, 2), "overhead"),
            DetectedSwing(3, Fraction(15), "forehand"),
        )

        result = select_comparison_windows(
            user_source=user,
            user_swings=swings,
            pro_source=pro,
            pro_selection=selection,
            pro_speed=Fraction(1, 4),
        )

        self.assertIsInstance(result, SelectedComparisonWindows)
        self.assertEqual([window.swing_ordinal for window in result.user], [1, 3])
        self.assertEqual(result.pro.contact_frame, pro_contact)
        self.assertEqual(result.pro.normalized_frames[0].offset, Fraction(-6, 5))
        self.assertEqual(result.pro.normalized_frames[-1].offset, Fraction(7, 10))
        # Pro source offsets are quarter-speed normalized; user offsets stay real-time.
        self.assertEqual(
            result.pro.normalized_frames[1].offset
            - result.pro.normalized_frames[0].offset,
            Fraction(1, 160),
        )
        self.assertEqual(
            result.user[0].normalized_frames[1].offset
            - result.user[0].normalized_frames[0].offset,
            Fraction(1, 10),
        )

    def test_retains_frame_active_at_boundaries_and_explicit_contact_frame(self) -> None:
        user = ComparisonSource(
            path=Path("user.mov"),
            width=640,
            height=360,
            inspected_media=InspectedMedia(
                frames=tuple(
                    DecodedFrame(0, ordinal, pts, Fraction(1, 10))
                    for ordinal, pts in enumerate((80, 89, 91, 100, 103, 107, 108))
                )
            ),
        )
        pro = source(
            "pro.mov", first_pts=80, last_pts=120, time_base=Fraction(1, 10)
        )
        selection = ProSelection(pro.path, pro.inspected_media.frames[20], "serve")

        result = select_comparison_windows(
            user_source=user,
            user_swings=(DetectedSwing(4, Fraction(10), "serve"),),
            pro_source=pro,
            pro_selection=selection,
            pro_speed=Fraction(1),
        )

        window = result.user[0]
        self.assertEqual(
            [(item.frame.pts, item.offset) for item in window.normalized_frames],
            [
                (80, Fraction(-6, 5)),
                (89, Fraction(-11, 10)),
                (91, Fraction(-9, 10)),
                (100, Fraction(0)),
                (103, Fraction(3, 10)),
                (107, Fraction(7, 10)),
            ],
        )
        self.assertEqual(window.contact_frame.pts, 100)

    def test_silently_omits_matching_swings_without_complete_window(self) -> None:
        user = source(
            "user.mov", first_pts=0, last_pts=30, time_base=Fraction(1, 10)
        )
        pro = source(
            "pro.mov", first_pts=0, last_pts=30, time_base=Fraction(1, 10)
        )
        selection = ProSelection(pro.path, pro.inspected_media.frames[15], "volley")

        result = select_comparison_windows(
            user_source=user,
            user_swings=(
                DetectedSwing(0, Fraction(1), "volley"),
                DetectedSwing(1, Fraction(3, 2), "volley"),
                DetectedSwing(2, Fraction(5, 2), "volley"),
            ),
            pro_source=pro,
            pro_selection=selection,
            pro_speed=Fraction(1),
        )

        self.assertEqual([window.swing_ordinal for window in result.user], [1])


class RenderPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user = source(
            "user.mov", first_pts=80, last_pts=120, time_base=Fraction(1, 10)
        )
        self.pro = source(
            "pro.mov", first_pts=40, last_pts=60, time_base=Fraction(1, 5)
        )
        self.windows = select_comparison_windows(
            user_source=self.user,
            user_swings=(DetectedSwing(7, Fraction(10), "forehand"),),
            pro_source=self.pro,
            pro_selection=ProSelection(
                self.pro.path, self.pro.inspected_media.frames[10], "forehand"
            ),
            pro_speed=Fraction(1),
        )

    def test_unions_observations_into_fixed_8_by_9_crops_and_equal_panels(self) -> None:
        observations = (
            PlayerObservation(0, Rectangle(200, 100, 160, 180)),
            PlayerObservation(1, Rectangle(220, 110, 120, 140)),
        )

        prepared_user = prepare_source_window(
            self.windows.user[0], observations
        )
        prepared_pro = prepare_source_window(
            self.windows.pro,
            (
                PlayerObservation(0, Rectangle(0, 0, 200, 500)),
                PlayerObservation(1, Rectangle(100, 100, 100, 400)),
            ),
        )
        plan = build_render_plan(
            user=prepared_user,
            pro=prepared_pro,
            slow_motion=Fraction(1, 2),
            artifact=ArtifactRequest(Path("comparison.mp4")),
        )

        self.assertEqual(prepared_user.crop, Rectangle(180, 77, 200, 225))
        self.assertEqual(prepared_pro.crop, Rectangle(0, 0, 560, 630))
        self.assertEqual(plan.layout.output, Rectangle(0, 0, 1280, 720))
        self.assertEqual(plan.layout.user_panel, Rectangle(0, 0, 640, 720))
        self.assertEqual(plan.layout.pro_panel, Rectangle(640, 0, 640, 720))
        with self.assertRaises(FrozenInstanceError):
            plan.slow_motion = Fraction(1)  # ty: ignore[invalid-assignment]

    def test_builds_exact_event_union_with_shared_contact_and_source_holds(self) -> None:
        user = prepare_source_window(
            self.windows.user[0],
            (PlayerObservation(0, Rectangle(200, 100, 160, 180)),),
        )
        pro = prepare_source_window(
            self.windows.pro,
            (PlayerObservation(0, Rectangle(200, 100, 160, 180)),),
        )

        plan = build_render_plan(
            user=user,
            pro=pro,
            slow_motion=Fraction(1, 2),
            artifact=ArtifactRequest(Path("comparison.mp4")),
        )

        self.assertEqual(plan.clip_bounds.normalized_start, Fraction(-6, 5))
        self.assertEqual(plan.clip_bounds.normalized_end, Fraction(7, 10))
        self.assertEqual(plan.output_time_base, Fraction(1, 5))
        self.assertEqual(plan.events[0].normalized_time, Fraction(-6, 5))
        self.assertEqual(plan.events[-1].normalized_time, Fraction(7, 10))
        contact = next(
            event for event in plan.events if event.normalized_time == Fraction(0)
        )
        self.assertEqual(contact.output_tick, 12)
        self.assertEqual(contact.user_frame, user.window.contact_frame)
        self.assertEqual(contact.pro_frame, pro.window.contact_frame)
        self.assertEqual(
            {event.user_frame.ordinal for event in plan.events},
            {item.frame.ordinal for item in user.window.normalized_frames},
        )
        self.assertEqual(
            {event.pro_frame.ordinal for event in plan.events},
            {item.frame.ordinal for item in pro.window.normalized_frames},
        )
        pro_holds = sum(
            current.pro_frame == previous.pro_frame
            for previous, current in zip(plan.events, plan.events[1:])
        )
        self.assertGreater(pro_holds, 0)

    def test_fails_when_distinct_events_exceed_target_timescale(self) -> None:
        user = prepare_source_window(
            self.windows.user[0],
            (PlayerObservation(0, Rectangle(200, 100, 160, 180)),),
        )
        pro = prepare_source_window(
            self.windows.pro,
            (PlayerObservation(0, Rectangle(200, 100, 160, 180)),),
        )
        first = user.window.normalized_frames[0]
        impossible_offset = Fraction(-6, 5) + Fraction(1, 2_147_483_648)
        impossible_window = type(user.window)(
            source=user.window.source,
            swing_ordinal=user.window.swing_ordinal,
            contact_timestamp=user.window.contact_timestamp,
            contact_frame=user.window.contact_frame,
            normalized_frames=(
                first,
                type(first)(user.window.normalized_frames[1].frame, impossible_offset),
                *user.window.normalized_frames[1:],
            ),
        )
        impossible_user = type(user)(
            window=impossible_window,
            observations=user.observations,
            crop=user.crop,
        )

        with self.assertRaises(UnrepresentableTimeline):
            build_render_plan(
                user=impossible_user,
                pro=pro,
                slow_motion=Fraction(1),
                artifact=ArtifactRequest(Path("comparison.mp4")),
            )


if __name__ == "__main__":
    unittest.main()
