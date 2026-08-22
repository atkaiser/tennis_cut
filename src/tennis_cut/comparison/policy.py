"""The single immutable policy for comparison timing and composition."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ComparisonPolicy:
    """Non-configurable visual and timing decisions for comparisons."""

    pre_contact: Fraction = Fraction(6, 5)
    post_contact: Fraction = Fraction(7, 10)
    crop_margin: Fraction = Fraction(1, 4)
    panel_aspect: tuple[int, int] = (8, 9)
    output_size: tuple[int, int] = (1280, 720)
    user_panel_origin: tuple[int, int] = (0, 0)
    pro_panel_origin: tuple[int, int] = (640, 0)
    supported_shot_types: frozenset[str] = frozenset(
        {"forehand", "backhand", "volley", "serve"}
    )


COMPARISON_POLICY = ComparisonPolicy()
