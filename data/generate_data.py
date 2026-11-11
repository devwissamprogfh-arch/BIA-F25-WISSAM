"""Synthetic data generator with a PLANTED feasible solution.

Why synthetic data? The project plan states that we do not claim to use real
university data. We create data ourselves so that cases with a known answer
exist (small / medium / large, as required by the plan: 20 / 60 / 150 events).

Key idea - planting
-------------------
1. Build rooms, teachers and courses.
2. Build a *planted schedule* that satisfies every hard constraint by
   construction (random placement with explicit conflict checks).
3. Choose teacher unavailability only among slots the planted schedule does
   not use, and enrol every student only in courses that do not overlap in the
   planted schedule and whose planted rooms are large enough.

Hence every generated instance is guaranteed to be FEASIBLE: a schedule with
zero hard violations exists (it is saved as ``planted_schedule.csv``). If an
algorithm fails to find a feasible schedule, the algorithm is to blame, not the
data. The planted schedule is NOT optimal for the soft constraints, so
algorithms can (and should) still improve on it.

Usage (from the project root):
    python -m data.generate_data            # (re)generate all six data sets
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from core.problem import Problem, save_problem
from core.validator import diagnose_instance, validate_schedule
from models.entities import Course, Room, Student, Teacher, TimeGrid
from models.exam import build_exam_events
from models.lecture import build_lecture_events
from models.schedule import Schedule, save_schedule_csv

# Sizes required by the plan: 20 / 60 / 150 events.
SIZES: Dict[str, Dict[str, int]] = {
    "small":  dict(n_events=20,  n_students=60,  n_teachers=6,  n_rooms=4,  seed=11),
    "medium": dict(n_events=60,  n_students=180, n_teachers=14, n_rooms=7,  seed=22),
    "large":  dict(n_events=150, n_students=420, n_teachers=30, n_rooms=12, seed=33),
}

LECTURE_DAYS = ("Sun", "Mon", "Tue", "Wed", "Thu")            # one teaching week
EXAM_DAYS = tuple(f"W{w}-{d}" for w in (1, 2) for d in ("Sun", "Mon", "Tue", "Wed", "Thu"))
SLOTS_PER_DAY = 8                                               # 08:00 .. 16:00, 1 hour slots


def _make_grid(mode: str) -> TimeGrid:
    return TimeGrid(LECTURE_DAYS if mode == "lecture" else EXAM_DAYS, SLOTS_PER_DAY, 8, 60)


def _make_rooms(rng: random.Random, mode: str, n_rooms: int) -> List[Room]:
    n_labs = max(1, round(0.2 * n_rooms)) if mode == "lecture" else 0
    rooms: List[Room] = []
    for k in range(n_rooms - n_labs):
        rooms.append(Room(f"R{k + 1:02d}", f"Room {k + 1}", rng.choice([30, 40, 50, 60, 80, 100]), "standard"))
    for k in range(n_labs):
        rooms.append(Room(f"LAB{k + 1:02d}", f"Lab {k + 1}", rng.choice([24, 30, 36]), "lab"))
    return rooms


def _make_courses(rng: random.Random, mode: str, n_events: int,
                  teacher_ids: List[str]) -> List[Course]:
    """Create courses so that the number of EVENTS equals ``n_events`` exactly."""
    courses: List[Course] = []
    remaining, k = n_events, 0
    while remaining > 0:
        k += 1
        if mode == "lecture":
            sessions = min(rng.choice([1, 2, 2, 3]), remaining)
            remaining -= sessions
            room_type = "lab" if rng.random() < 0.15 else "standard"
            duration = rng.choices([1, 2], [0.4, 0.6])[0]
            exam_duration = 2
        else:                                   # one exam per course
            sessions, room_type, duration = 1, "standard", 1
            exam_duration = rng.choice([2, 2, 3])
            remaining -= 1
        courses.append(Course(f"C{k:03d}", f"Course {k}", "", sessions, duration, room_type,
                              True, exam_duration))
    # Spread teaching load evenly: shuffle the courses over the teachers round-robin.
    order = list(range(len(courses)))
    rng.shuffle(order)
    assigned = {}
    for pos, idx in enumerate(order):
        assigned[idx] = teacher_ids[pos % len(teacher_ids)]
    return [Course(c.course_id, c.name, assigned[i], c.sessions_per_week, c.duration_slots,
                   c.room_type, c.has_exam, c.exam_duration_slots) for i, c in enumerate(courses)]


def _plant_schedule(rng: random.Random, mode: str, grid: TimeGrid, rooms: List[Room],
                    courses: List[Course]) -> Dict[str, Tuple[int, int]]:
    """Return {event_id: (start_slot, room_index)} satisfying teacher/room constraints."""
    builder = build_lecture_events if mode == "lecture" else build_exam_events
    events = [e for c in courses for e in builder(c, ())]
    # Hardest first: lab events and long events have the fewest options.
    events.sort(key=lambda e: (e.room_type != "lab", -e.duration, rng.random()))
    room_busy: Set[Tuple[int, int]] = set()
    teacher_busy: Dict[str, Set[int]] = defaultdict(set)
    course_days: Dict[str, Set[int]] = defaultdict(set)
    planted: Dict[str, Tuple[int, int]] = {}
    spd = grid.slots_per_day

    def free(e, g: int, r: int) -> bool:
        span = range(g, g + e.duration)
        return (rooms[r].room_type == e.room_type
                and all((r, x) not in room_busy for x in span)
                and all(x not in teacher_busy[e.teacher_id] for x in span))

    for e in events:
        starts = [g for g in range(grid.n_slots) if g % spd + e.duration <= spd]
        room_idx = [k for k, r in enumerate(rooms) if r.room_type == e.room_type]
        choice = None
        for _ in range(400):                    # random tries, preferring a new day per course
            g, r = rng.choice(starts), rng.choice(room_idx)
            if free(e, g, r) and (grid.day_of(g) not in course_days[e.course_id] or _ >= 200):
                choice = (g, r)
                break
        if choice is None:                      # exhaustive fallback
            cands = [(g, r) for g in starts for r in room_idx if free(e, g, r)]
            if not cands:
                raise RuntimeError(f"could not plant {e.event_id}; use a bigger grid/more rooms")
            choice = rng.choice(cands)
        g, r = choice
        planted[e.event_id] = (g, r)
        course_days[e.course_id].add(grid.day_of(g))
        for x in range(g, g + e.duration):
            room_busy.add((r, x))
            teacher_busy[e.teacher_id].add(x)
    return planted


def generate_instance(mode: str, size: str) -> Tuple[Problem, Schedule]:
    """Generate one feasible instance and its planted solution."""
    cfg = SIZES[size]
    rng = random.Random(cfg["seed"] + (0 if mode == "lecture" else 1000))
    grid = _make_grid(mode)
    teacher_ids = [f"T{k + 1:02d}" for k in range(cfg["n_teachers"])]
    rooms = _make_rooms(rng, mode, cfg["n_rooms"])
    courses = _make_courses(rng, mode, cfg["n_events"], teacher_ids)
    planted = _plant_schedule(rng, mode, grid, rooms, courses)

    # --- per-course planted time spans and the smallest planted room capacity -----
    builder = build_lecture_events if mode == "lecture" else build_exam_events
    spans: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    cap: Dict[str, int] = {}
    teacher_used: Dict[str, Set[int]] = defaultdict(set)
    for c in courses:
        for e in builder(c, ()):
            g, r = planted[e.event_id]
            spans[c.course_id].append((g, g + e.duration))
            cap[c.course_id] = min(cap.get(c.course_id, 10 ** 9), rooms[r].capacity)
            teacher_used[c.teacher_id].update(range(g, g + e.duration))
    limit = {cid: int(0.9 * v) for cid, v in cap.items()}        # keep a little head-room

    def clashes(cid_a: str, taken: List[Tuple[int, int]]) -> bool:
        return any(sa < eb and sb < ea for sa, ea in spans[cid_a] for sb, eb in taken)

    # --- students: only courses that do not overlap in the planted schedule ------
    students = [Student(f"S{k + 1:03d}", f"Student {k + 1}") for k in range(cfg["n_students"])]
    enrolled: Dict[str, List[str]] = defaultdict(list)           # student -> courses
    size_of: Dict[str, int] = defaultdict(int)                   # course -> #students
    course_ids = [c.course_id for c in courses]
    for s in students:
        wanted = rng.randint(4, 6)
        taken: List[Tuple[int, int]] = []
        pool = course_ids[:]
        rng.shuffle(pool)
        for cid in pool:
            if len(enrolled[s.student_id]) >= wanted:
                break
            if size_of[cid] < limit[cid] and not clashes(cid, taken):
                enrolled[s.student_id].append(cid)
                size_of[cid] += 1
                taken += spans[cid]
    # second pass: make sure no course is nearly empty
    for cid in course_ids:
        for s in students:
            if size_of[cid] >= 8:
                break
            taken = [sp for other in enrolled[s.student_id] for sp in spans[other]]
            if (cid not in enrolled[s.student_id] and size_of[cid] < limit[cid]
                    and len(enrolled[s.student_id]) < 7 and not clashes(cid, taken)):
                enrolled[s.student_id].append(cid)
                size_of[cid] += 1
    enrollments = [(sid, cid) for sid, lst in enrolled.items() for cid in lst]

    # --- teachers: unavailable slots avoid the planted usage; some have preferences -
    teachers: List[Teacher] = []
    morning = frozenset(g for g in range(grid.n_slots) if g % SLOTS_PER_DAY < SLOTS_PER_DAY // 2)
    afternoon = frozenset(range(grid.n_slots)) - morning
    for tid in teacher_ids:
        free_slots = sorted(set(range(grid.n_slots)) - teacher_used[tid])
        k = rng.randint(3, 8) if mode == "lecture" else rng.randint(4, 10)
        unavailable = frozenset(rng.sample(free_slots, min(k, len(free_slots))))
        preferred = rng.choice([frozenset(), morning, afternoon])
        teachers.append(Teacher(tid, f"Teacher {tid[1:]}", unavailable, preferred))

    problem = Problem(f"{mode}_{size}", mode, grid, rooms, teachers, courses, students, enrollments)
    sched = Schedule.from_mapping(problem.event_ids, problem.room_ids,
                                  {eid: (g, rooms[r].room_id) for eid, (g, r) in planted.items()})
    return problem, sched


def generate_all(out_dir: str) -> None:
    """Write the six data sets (lecture/exam x small/medium/large)."""
    for mode in ("lecture", "exam"):
        for size in SIZES:
            problem, planted = generate_instance(mode, size)
            report = validate_schedule(problem, planted)
            diag = diagnose_instance(problem)
            assert report.is_valid, f"planted schedule of {mode}/{size} is invalid:\n{report.text()}"
            assert diag.possibly_feasible, diag.text()
            path = os.path.join(out_dir, mode, size)
            save_problem(problem, path, {"seed": SIZES[size]["seed"], "size": size,
                                         "synthetic": True, "has_planted_solution": True})
            save_schedule_csv(problem, planted, os.path.join(path, "planted_schedule.csv"))
            s = problem.summary()
            print(f"{mode}/{size:<7} events={s['events']:>4} rooms={s['rooms']:>3} "
                  f"students={s['students']:>4} utilisation={s['room_utilisation_pct']}%")


if __name__ == "__main__":
    generate_all(os.path.dirname(os.path.abspath(__file__)))
