"""Lecture events: a course with ``sessions_per_week = k`` yields k lecture events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from models.entities import Course, Event


@dataclass(frozen=True)
class Lecture(Event):
    """A weekly lecture session of a course."""

    kind: str = "lecture"


def build_lecture_events(course: Course, student_ids: Iterable[str]) -> List[Lecture]:
    """Create the weekly lecture events of ``course``.

    Every session is a separate event (ids ``L_<course>_<n>``) that must be
    placed on its own; all of them share the same teacher and student set.
    """
    students = frozenset(student_ids)
    return [
        Lecture(
            event_id=f"L_{course.course_id}_{k + 1}",
            course_id=course.course_id,
            teacher_id=course.teacher_id,
            duration=course.duration_slots,
            room_type=course.room_type,
            student_ids=students,
            session_index=k,
        )
        for k in range(course.sessions_per_week)
    ]
