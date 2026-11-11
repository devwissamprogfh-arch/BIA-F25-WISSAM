"""Problem instance: all static data plus pre-computed lookup tables.

The :class:`Problem` object is the single source of truth shared by the
validator, the fitness function and the three algorithms. It is built once and
never modified afterwards. Everything an algorithm may need to search
efficiently (feasible rooms of an event, valid start slots, which events share
students, ...) is pre-computed here so that the (very frequently called)
fitness function stays cheap.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from models.entities import Course, Room, Student, Teacher, TimeGrid
from models.exam import build_exam_events
from models.lecture import build_lecture_events

MODES = ("lecture", "exam")


class DataError(ValueError):
    """Raised when input data is malformed or internally inconsistent."""


class Problem:
    """A complete scheduling instance.

    Parameters
    ----------
    mode : 'lecture' builds ``sessions_per_week`` events per course;
           'exam' builds one exam event per course that has an exam.
    min_gap_slots : minimum desired rest (in slots) between two events of the
           same student on the same day. It only drives the SOFT 'gap'
           constraint. Defaults: 0 for lectures (disabled), 1 for exams.
    """

    def __init__(self, name: str, mode: str, grid: TimeGrid,
                 rooms: Sequence[Room], teachers: Iterable[Teacher],
                 courses: Iterable[Course], students: Sequence[Student],
                 enrollments: Iterable[Tuple[str, str]],
                 min_gap_slots: Optional[int] = None):
        if mode not in MODES:
            raise DataError(f"mode must be one of {MODES}, got {mode!r}")
        self.name = name
        self.mode = mode
        self.grid = grid
        self.rooms: List[Room] = list(rooms)
        self.teachers: Dict[str, Teacher] = {t.teacher_id: t for t in teachers}
        self.courses: Dict[str, Course] = {c.course_id: c for c in courses}
        self.students: List[Student] = list(students)
        self.enrollments: List[Tuple[str, str]] = list(enrollments)
        self.min_gap_slots = (1 if mode == "exam" else 0) if min_gap_slots is None else min_gap_slots

        self._check_references()
        self.room_ids: List[str] = [r.room_id for r in self.rooms]
        self.room_index: Dict[str, int] = {rid: k for k, rid in enumerate(self.room_ids)}
        self._build_events()
        self._build_lookup_tables()

    # ------------------------------------------------------------------ validation
    def _check_references(self) -> None:
        """Reject data whose ids do not refer to existing entities."""
        if self.grid.slots_per_day <= 0 or self.grid.n_days <= 0:
            raise DataError("time grid must have at least one day and one slot per day")
        if len({r.room_id for r in self.rooms}) != len(self.rooms):
            raise DataError("duplicate room ids")
        student_ids = {s.student_id for s in self.students}
        if len(student_ids) != len(self.students):
            raise DataError("duplicate student ids")
        for r in self.rooms:
            if r.capacity <= 0:
                raise DataError(f"room {r.room_id}: capacity must be positive")
        for c in self.courses.values():
            if c.teacher_id not in self.teachers:
                raise DataError(f"course {c.course_id}: unknown teacher {c.teacher_id!r}")
            if c.sessions_per_week < 1 or c.duration_slots < 1 or c.exam_duration_slots < 1:
                raise DataError(f"course {c.course_id}: sessions and durations must be >= 1")
        for sid, cid in self.enrollments:
            if sid not in student_ids:
                raise DataError(f"enrollment refers to unknown student {sid!r}")
            if cid not in self.courses:
                raise DataError(f"enrollment refers to unknown course {cid!r}")
        for t in self.teachers.values():
            for g in set(t.unavailable_slots) | set(t.preferred_slots):
                if not 0 <= g < self.grid.n_slots:
                    raise DataError(f"teacher {t.teacher_id}: slot {g} outside the time grid")

    # ------------------------------------------------------------------ events
    def _build_events(self) -> None:
        course_students: Dict[str, Set[str]] = defaultdict(set)
        for sid, cid in self.enrollments:
            course_students[cid].add(sid)
        builder = build_lecture_events if self.mode == "lecture" else build_exam_events
        self.events = []
        for cid, course in self.courses.items():
            self.events.extend(builder(course, course_students.get(cid, ())))
        self.event_ids: List[str] = [e.event_id for e in self.events]
        self.event_index: Dict[str, int] = {eid: i for i, eid in enumerate(self.event_ids)}

    # ------------------------------------------------------------------ look-ups
    def _build_lookup_tables(self) -> None:
        spd, n_slots = self.grid.slots_per_day, self.grid.n_slots
        self.durations: List[int] = [e.duration for e in self.events]
        self.sizes: List[int] = [e.size for e in self.events]
        self.teacher_unavailable: List[FrozenSet[int]] = [
            self.teachers[e.teacher_id].unavailable_slots for e in self.events]
        self.teacher_preferred: List[FrozenSet[int]] = [
            self.teachers[e.teacher_id].preferred_slots for e in self.events]

        # Static search-space reduction used by the algorithms.
        self.feasible_rooms: List[List[int]] = [
            [k for k, r in enumerate(self.rooms)
             if r.room_type == e.room_type and r.capacity >= e.size]
            for e in self.events]
        self.valid_starts: List[List[int]] = [
            [g for g in range(n_slots) if g % spd + e.duration <= spd]
            for e in self.events]

        # Event groups.
        self.teacher_events: Dict[str, List[int]] = defaultdict(list)
        self.course_events: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(self.events):
            self.teacher_events[e.teacher_id].append(i)
            self.course_events[e.course_id].append(i)

        # Student sharing: shared_pairs[(i, j)] (i < j) = number of common students.
        self.student_events: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(self.events):
            for sid in e.student_ids:
                self.student_events[sid].append(i)
        self.shared_pairs: Dict[Tuple[int, int], int] = defaultdict(int)
        for lst in self.student_events.values():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    self.shared_pairs[(lst[a], lst[b])] += 1
        self.shared_pairs = dict(self.shared_pairs)
        # Adjacency view: neighbors[i] = [(j, number_of_common_students), ...]
        self.neighbors: List[List[Tuple[int, int]]] = [[] for _ in self.events]
        for (i, j), n in self.shared_pairs.items():
            self.neighbors[i].append((j, n))
            self.neighbors[j].append((i, n))
        # Sort so results do not depend on the (hash-randomised) order of student ids:
        # the same seed must give the same run in every Python process.
        for nb in self.neighbors:
            nb.sort()

    # ------------------------------------------------------------------ shortcuts
    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_rooms(self) -> int:
        return len(self.rooms)

    @property
    def n_slots(self) -> int:
        return self.grid.n_slots

    def summary(self) -> Dict[str, float]:
        """Key figures describing the size and tightness of the instance."""
        units = sum(self.durations)
        return {
            "name": self.name,
            "mode": self.mode,
            "events": self.n_events,
            "courses": len(self.courses),
            "teachers": len(self.teachers),
            "rooms": self.n_rooms,
            "students": len(self.students),
            "enrollments": len(self.enrollments),
            "slots": self.n_slots,
            "slot_units_to_place": units,
            "room_slot_capacity": self.n_rooms * self.n_slots,
            "room_utilisation_pct": round(100.0 * units / (self.n_rooms * self.n_slots), 1),
            "avg_event_size": round(sum(self.sizes) / max(1, self.n_events), 1),
            "event_pairs_sharing_students": len(self.shared_pairs),
        }


# ====================================================================== CSV I/O
def _parse_slots(text: str) -> FrozenSet[int]:
    """Parse '3;4;12' into a frozenset of ints ('' -> empty set)."""
    text = (text or "").strip()
    return frozenset(int(x) for x in text.split(";") if x.strip() != "")


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise DataError(f"missing data file: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_int(row: Dict[str, str], key: str, fname: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError, TypeError):
        raise DataError(f"{fname}: column {key!r} must be an integer (row {row})") from None


def load_problem(directory: str, mode: Optional[str] = None) -> Problem:
    """Load an instance from a directory holding the five CSV files + meta.json.

    Expected files: courses.csv, rooms.csv, teachers.csv, students.csv,
    enrollments.csv (and optionally meta.json with the time grid and mode).
    """
    meta_path = os.path.join(directory, "meta.json")
    meta: Dict = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    mode = mode or meta.get("mode", "lecture")
    grid = TimeGrid(
        day_names=tuple(meta.get("day_names", ["Sun", "Mon", "Tue", "Wed", "Thu"])),
        slots_per_day=int(meta.get("slots_per_day", 8)),
        first_slot_hour=int(meta.get("first_slot_hour", 8)),
        slot_minutes=int(meta.get("slot_minutes", 60)),
    )
    rooms = [Room(r["room_id"], r["name"], _to_int(r, "capacity", "rooms.csv"), r["room_type"])
             for r in _read_csv(os.path.join(directory, "rooms.csv"))]
    teachers = [Teacher(t["teacher_id"], t["name"], _parse_slots(t.get("unavailable_slots", "")),
                        _parse_slots(t.get("preferred_slots", "")))
                for t in _read_csv(os.path.join(directory, "teachers.csv"))]
    courses = [Course(c["course_id"], c["name"], c["teacher_id"],
                      _to_int(c, "sessions_per_week", "courses.csv"),
                      _to_int(c, "duration_slots", "courses.csv"), c["room_type"],
                      c["has_exam"].strip().lower() in ("1", "true", "yes"),
                      _to_int(c, "exam_duration_slots", "courses.csv"))
               for c in _read_csv(os.path.join(directory, "courses.csv"))]
    students = [Student(s["student_id"], s["name"])
                for s in _read_csv(os.path.join(directory, "students.csv"))]
    enrollments = [(e["student_id"], e["course_id"])
                   for e in _read_csv(os.path.join(directory, "enrollments.csv"))]
    return Problem(meta.get("name", os.path.basename(os.path.normpath(directory))), mode, grid,
                   rooms, teachers, courses, students, enrollments,
                   meta.get("min_gap_slots"))


def save_problem(problem: Problem, directory: str, extra_meta: Optional[Dict] = None) -> None:
    """Write an instance in the CSV layout understood by :func:`load_problem`."""
    os.makedirs(directory, exist_ok=True)

    def write(fname: str, header: List[str], rows: List[List]) -> None:
        with open(os.path.join(directory, fname), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    def slots_text(s: FrozenSet[int]) -> str:
        return ";".join(str(g) for g in sorted(s))

    write("rooms.csv", ["room_id", "name", "capacity", "room_type"],
          [[r.room_id, r.name, r.capacity, r.room_type] for r in problem.rooms])
    write("teachers.csv", ["teacher_id", "name", "unavailable_slots", "preferred_slots"],
          [[t.teacher_id, t.name, slots_text(t.unavailable_slots), slots_text(t.preferred_slots)]
           for t in problem.teachers.values()])
    write("courses.csv", ["course_id", "name", "teacher_id", "sessions_per_week", "duration_slots",
                          "room_type", "has_exam", "exam_duration_slots"],
          [[c.course_id, c.name, c.teacher_id, c.sessions_per_week, c.duration_slots, c.room_type,
            int(c.has_exam), c.exam_duration_slots] for c in problem.courses.values()])
    write("students.csv", ["student_id", "name"], [[s.student_id, s.name] for s in problem.students])
    write("enrollments.csv", ["student_id", "course_id"], [list(e) for e in problem.enrollments])
    meta = {"name": problem.name, "mode": problem.mode, "day_names": list(problem.grid.day_names),
            "slots_per_day": problem.grid.slots_per_day,
            "first_slot_hour": problem.grid.first_slot_hour,
            "slot_minutes": problem.grid.slot_minutes, "min_gap_slots": problem.min_gap_slots}
    meta.update(extra_meta or {})
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
