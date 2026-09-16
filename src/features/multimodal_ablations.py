from __future__ import annotations

from typing import Final

from .multimodal_columns import (
    PROFILE_COLUMNS,
    GROWTH_COLUMNS,
    SUN_COLUMNS,
    NUTRITION_COLUMNS,
    BREASTFEEDING_COLUMNS,
    SUPPLEMENT_COLUMNS,
    OPTICAL_COLUMNS,
)


PROFILE_GROWTH_COLUMNS: Final[tuple[str, ...]] = (
    *PROFILE_COLUMNS,
    *GROWTH_COLUMNS,
)

BEHAVIOR_NUTRITION_COLUMNS: Final[tuple[str, ...]] = (
    *SUN_COLUMNS,
    *NUTRITION_COLUMNS,
    *BREASTFEEDING_COLUMNS,
    *SUPPLEMENT_COLUMNS,
)

OPTICAL_ONLY_COLUMNS: Final[tuple[str, ...]] = (
    *OPTICAL_COLUMNS,
)


ABLATION_FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "profile_growth": PROFILE_GROWTH_COLUMNS,

    "behavior_nutrition": BEHAVIOR_NUTRITION_COLUMNS,

    "optical_only": OPTICAL_ONLY_COLUMNS,

    "profile_behavior": (
        *PROFILE_GROWTH_COLUMNS,
        *BEHAVIOR_NUTRITION_COLUMNS,
    ),

    "profile_optical": (
        *PROFILE_GROWTH_COLUMNS,
        *OPTICAL_ONLY_COLUMNS,
    ),

    "behavior_optical": (
        *BEHAVIOR_NUTRITION_COLUMNS,
        *OPTICAL_ONLY_COLUMNS,
    ),

    "all_modalities": (
        *PROFILE_GROWTH_COLUMNS,
        *BEHAVIOR_NUTRITION_COLUMNS,
        *OPTICAL_ONLY_COLUMNS,
    ),
}


def get_ablation_columns(
    experiment_name: str,
) -> tuple[str, ...]:
    """
    Return the frozen feature columns for an ablation experiment.
    """

    if experiment_name not in ABLATION_FEATURE_GROUPS:
        available = ", ".join(ABLATION_FEATURE_GROUPS)

        raise ValueError(
            f"Unknown ablation experiment: {experiment_name!r}. "
            f"Available experiments: {available}"
        )

    return ABLATION_FEATURE_GROUPS[experiment_name]


def ablation_experiment_names() -> tuple[str, ...]:
    """Return all available ablation experiment names."""

    return tuple(ABLATION_FEATURE_GROUPS.keys())