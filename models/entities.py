"""Core domain entities of the Smart Academic Scheduler.

All entities are small immutable dataclasses so that they can be shared safely
between the three optimisation algorithms (GA / PSO / ACO) without any risk of
one algorithm silently modifying the problem data of another one.

Time model
----------
Time is discretised into *slots*. A global slot index ``g`` identifies one slot
of the whole planning period:

    g = day_index * slots_per_day + slot_in_day

An event that starts at global slot ``g`` and lasts ``d`` slots occupies the
half-open interval ``[g, g + d)``. Two events overlap iff their intervals
intersect, so an event ending at slot 3 (exclusive) and another one starting at
slot 3 do NOT overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class TimeGrid:
    """Discrete planning period: ``len(day_names)`` days x ``slots_per_day`` slots."""

    day_names: Tuple[str, ...]
    slots_per_day: int
    first_slot_hour: int = 8      # wall-clock hour of the first slot of a day
    slot_minutes: int = 60        # length of one slot in minutes

    @property
    def n_days(self) -> int:
        return len(self.day_names)

    @property
    def n_slots(self) -> int:
        """Total number of slots in the planning period."""
        return self.n_days * self.slots_per_day

    def day_of(self, g: int) -> int:
        """Day index of a global slot."""
        return g // self.slots_per_day

    def slot_in_day(self, g: int) -> int:
        """Position of a global slot inside its day (0 .. slots_per_day-1)."""
        return g % self.slots_per_day

    def label(self, g: int) -> str:
        """Human readable label such as 'Mon 10:00' for a global slot."""
        if not 0 <= g < self.n_slots:
            raise IndexError(f"slot {g} is outside the grid (0..{self.n_slots - 1})")
        minutes = self.first_slot_hour * 60 + self.slot_in_day(g) * self.slot_minutes
        return f"{self.day_names[self.day_of(g)]} {minutes // 60:02d}:{minutes % 60:02d}"


@dataclass(frozen=True)
class Teacher:
    """A teacher (or exam supervisor). A teacher can run only one event at a time."""

    teacher_id: str
    name: str
    unavailable_slots: FrozenSet[int] = frozenset()   # HARD: events must avoid these slots
    preferred_slots: FrozenSet[int] = frozenset()     # SOFT: empty set = no preference


@dataclass(frozen=True)
class Room:
    """A room with a seat capacity and a type ('standard' or 'lab')."""

    room_id: str
    name: str
    capacity: int
    room_type: str = "standard"


@dataclass(frozen=True)
class Student:
    student_id: str
    name: str


@dataclass(frozen=True)
class Course:
    """A course. Each course yields several lecture events and/or one exam event."""

    course_id: str
    name: str
    teacher_id: str
    sessions_per_week: int = 1        # number of weekly lecture events
    duration_slots: int = 1           # length of one lecture, in slots
    room_type: str = "standard"       # required room type for lectures
    has_exam: bool = True
    exam_duration_slots: int = 2      # length of the exam, in slots


@dataclass(frozen=True)
class Event:
    """One schedulable unit: a single lecture session or a single exam.

    The optimisation algorithms assign every event a (start slot, room) pair.
    ``student_ids`` is the set of students who must attend the event, which is
    what makes student clashes detectable.
    """

    event_id: str
    course_id: str
    teacher_id: str
    duration: int
    room_type: str
    student_ids: FrozenSet[str]
    session_index: int = 0
    kind: str = "event"

    @property
    def size(self) -> int:
        """Number of attendees; the assigned room must have at least this capacity."""
        return len(self.student_ids)
