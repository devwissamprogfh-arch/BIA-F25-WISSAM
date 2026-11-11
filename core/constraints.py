"""Catalogue of constraints, penalty weights and small time helpers.

HARD constraints  -> a final schedule MUST have zero violations of these.
SOFT constraints  -> quality criteria that we try to minimise.

Every constraint has a precise *counting unit* so that the fitness function
(``fitness.py``) and the independent validator (``validator.py``) report the
same numbers for the same schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ConstraintSpec:
    name: str            # key used in all reports and in the weights dictionary
    kind: str            # 'hard' or 'soft'
    unit: str            # what one violation counts
    description: str
    default_weight: float


CONSTRAINTS: Tuple[ConstraintSpec, ...] = (
    # ------------------------------------------------------------------ HARD
    ConstraintSpec("room_conflicts", "hard", "pair of events",
                   "Two events use the same room in overlapping time.", 1000.0),
    ConstraintSpec("teacher_conflicts", "hard", "pair of events",
                   "A teacher is assigned to two events that overlap in time.", 1000.0),
    ConstraintSpec("student_conflicts", "hard", "student x overlapping pair",
                   "A student must attend two events that overlap in time "
                   "(counted once per affected student).", 1000.0),
    ConstraintSpec("capacity_violations", "hard", "event",
                   "The room has fewer seats than the number of attendees.", 1000.0),
    ConstraintSpec("room_type_violations", "hard", "event",
                   "The room type differs from the type the event requires (e.g. lab).", 1000.0),
    ConstraintSpec("availability_violations", "hard", "event",
                   "The event overlaps a slot in which its teacher is unavailable.", 1000.0),
    ConstraintSpec("bounds_violations", "hard", "event",
                   "The event is unassigned, has an invalid slot/room, or runs past "
                   "the end of its day.", 1000.0),
    # ------------------------------------------------------------------ SOFT
    ConstraintSpec("preference_violations", "soft", "event",
                   "The event is (partly) outside the teacher's preferred slots.", 2.0),
    ConstraintSpec("spread_violations", "soft", "pair of events",
                   "Two lecture sessions of the same course fall on the same day.", 3.0),
    ConstraintSpec("gap_violations", "soft", "student x close pair",
                   "A student has two events on one day with less than "
                   "'min_gap_slots' of rest between them (exams by default).", 2.0),
    ConstraintSpec("room_waste", "soft", "fraction of seats",
                   "Sum over events of (capacity - attendees) / capacity: "
                   "discourages using huge rooms for small groups.", 1.0),
)

HARD_NAMES: Tuple[str, ...] = tuple(c.name for c in CONSTRAINTS if c.kind == "hard")
SOFT_NAMES: Tuple[str, ...] = tuple(c.name for c in CONSTRAINTS if c.kind == "soft")
DEFAULT_WEIGHTS: Dict[str, float] = {c.name: c.default_weight for c in CONSTRAINTS}


@dataclass
class FitnessConfig:
    """Penalty weights (w1..w11 of the objective function).

    Design rule: every hard weight (1000) is larger than the whole soft penalty
    of any reasonable schedule (the largest soft total measured on the planted
    solutions is about 450), so removing one hard violation always pays off more
    than any soft improvement. Weights are documented here and can be
    overridden per experiment (``with_overrides``).
    """

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown constraint names in weights: {sorted(unknown)}")
        for name, default in DEFAULT_WEIGHTS.items():   # fill anything missing
            self.weights.setdefault(name, default)

    def with_overrides(self, **overrides: float) -> "FitnessConfig":
        """Return a copy with some weights replaced."""
        merged = dict(self.weights)
        merged.update(overrides)
        return FitnessConfig(merged)


def intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """True iff the half-open intervals [start_a, end_a) and [start_b, end_b) intersect."""
    return start_a < end_b and start_b < end_a
