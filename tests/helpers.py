"""Shared helpers for the tests: a tiny hand-made instance with known answers."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.constraints import HARD_NAMES  # noqa: E402
from core.fitness import evaluate  # noqa: E402
from core.problem import Problem  # noqa: E402
from core.validator import validate_schedule  # noqa: E402
from models.entities import Course, Room, Student, Teacher, TimeGrid  # noqa: E402
from models.schedule import Schedule  # noqa: E402


def make_problem(min_gap_slots=None, mode="lecture") -> Problem:
    """Tiny instance.

    Grid : 2 days (Sun, Mon) x 4 slots  -> global slots 0..3 (Sun) and 4..7 (Mon)
    Rooms: R1 (30 seats), R2 (20 seats), LAB (30 seats, lab)
    T1   : unavailable at slot 3 ;  T2 : prefers slots {0, 1, 4, 5}
    C1 (T1): 2 sessions x 2 slots, 25 students S01..S25     -> L_C1_1, L_C1_2
    C2 (T2): 1 session  x 1 slot , 5 students S21..S25      -> L_C2_1
    C3 (T2): 1 session  x 2 slots, 3 students S01..S03, lab -> L_C3_1
    Students S21..S25 attend both C1 and C2 (5 shared students).
    """
    grid = TimeGrid(("Sun", "Mon"), 4, 8, 60)
    rooms = [Room("R1", "Room 1", 30, "standard"), Room("R2", "Room 2", 20, "standard"),
             Room("LAB", "Lab", 30, "lab")]
    teachers = [Teacher("T1", "Teacher 1", frozenset({3})),
                Teacher("T2", "Teacher 2", frozenset(), frozenset({0, 1, 4, 5}))]
    courses = [Course("C1", "Course 1", "T1", 2, 2, "standard", False, 2),
               Course("C2", "Course 2", "T2", 1, 1, "standard", False, 2),
               Course("C3", "Course 3", "T2", 1, 2, "lab", False, 2)]
    students = [Student(f"S{k:02d}", f"Student {k}") for k in range(1, 26)]
    enrollments = [(f"S{k:02d}", "C1") for k in range(1, 26)]
    enrollments += [(f"S{k:02d}", "C2") for k in range(21, 26)]
    enrollments += [(f"S{k:02d}", "C3") for k in range(1, 4)]
    return Problem("tiny", mode, grid, rooms, teachers, courses, students, enrollments, min_gap_slots)


def sched(problem: Problem, mapping) -> Schedule:
    """Build a schedule from {event_id: (start_slot, room_id)}."""
    return Schedule.from_mapping(problem.event_ids, problem.room_ids, mapping)


# A fully valid schedule of the tiny instance (zero hard violations).
CLEAN = {"L_C1_1": (0, "R1"),      # Sun 08-10, R1
         "L_C1_2": (4, "R1"),      # Mon 08-10, R1
         "L_C2_1": (2, "R2"),      # Sun 10-11, R2
         "L_C3_1": (6, "LAB")}     # Mon 10-12, LAB


def both(problem: Problem, mapping):
    """Evaluate with the fitness function AND the validator; assert they agree."""
    s = sched(problem, mapping)
    fit, val = evaluate(problem, s), validate_schedule(problem, s)
    for name in HARD_NAMES:
        assert fit.counts[name] == val.counts[name], (name, fit.counts[name], val.counts[name])
    return fit, val
