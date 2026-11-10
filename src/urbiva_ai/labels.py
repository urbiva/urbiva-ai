"""Label taxonomy shared with urbiva-app.

The order of CATEGORIES defines the class index used by the model. It must
match the categories accepted by the application's /api/vision contract.
"""

from __future__ import annotations

CATEGORIES: tuple[str, ...] = (
    "stairs",
    "blocked_path",
    "broken_surface",
    "curb",
    "obstruction",
    "other",
    "none",
)
NONE_CATEGORY = "none"
BARRIER_CATEGORIES: tuple[str, ...] = tuple(c for c in CATEGORIES if c != NONE_CATEGORY)
NUM_CLASSES = len(CATEGORIES)

CONFIDENCE_LEVELS: tuple[str, ...] = ("low", "medium", "high")


def index_of(category: str) -> int:
    """Return the class index of a category, raising ValueError for unknown labels."""
    try:
        return CATEGORIES.index(category)
    except ValueError as exc:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}") from exc


def is_barrier(category: str) -> bool:
    """True for every category except 'none'."""
    index_of(category)
    return category != NONE_CATEGORY
