"""Exam events: a course with an exam yields exactly one exam event."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from models.entities import Course, Event

# Exams are always held in ordinary exam rooms (never in labs).
EXAM_ROOM_TYPE = "standard"


@dataclass(frozen=True)
class Exam(Event):
    """The final exam of a course."""

    kind: str = "exam"


def build_exam_events(course: Course, student_ids: Iterable[str]) -> List[Exam]:
    """Create the exam event of ``course`` (empty list if the course has no exam)."""
    if not course.has_exam:
        return []
    return [
        Exam(
            event_id=f"E_{course.course_id}",
            course_id=course.course_id,
            teacher_id=course.teacher_id,     # the teacher acts as the exam supervisor
            duration=course.exam_duration_slots,
            room_type=EXAM_ROOM_TYPE,
            student_ids=frozenset(student_ids),
            session_index=0,
        )
    ]
