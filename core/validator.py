"""Independent validator.

Two services, both deliberately implemented WITHOUT reusing the code of
``fitness.py`` so that a bug in one of them cannot hide a bug in the other:

1. :func:`validate_schedule` - checks a concrete schedule against every HARD
   constraint and lists each violation with a readable message. It uses a
   brute-force, easy-to-audit method: explicit slot sets and a double loop over
   all event pairs.
2. :func:`diagnose_instance` - checks the *data* before any algorithm runs and
   detects instances that are impossible (or nearly impossible) to satisfy, e.g.
   not enough rooms or slots. This prevents blaming an algorithm for a failure
   that is caused by the data.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.constraints import HARD_NAMES
from core.problem import Problem
from models.schedule import Schedule


# ====================================================================== schedule
@dataclass
class Violation:
    """One detected hard-constraint violation."""

    code: str                  # one of core.constraints.HARD_NAMES
    events: Tuple[str, ...]    # ids of the events involved
    amount: int                # counting unit (1, or number of affected students)
    message: str


@dataclass
class ValidationReport:
    violations: List[Violation] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        """Total amount per hard constraint (same units as the fitness function)."""
        totals = {name: 0 for name in HARD_NAMES}
        for v in self.violations:
            totals[v.code] += v.amount
        return totals

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def is_valid(self) -> bool:
        """True iff the schedule satisfies every hard constraint."""
        return not self.violations

    def text(self, limit: int = 20) -> str:
        """Short human-readable summary (first ``limit`` violations)."""
        if self.is_valid:
            return "VALID: no hard-constraint violation."
        lines = [f"INVALID: {self.total} hard violation unit(s) in {len(self.violations)} finding(s)"]
        lines += [f"  - [{v.code}] {v.message}" for v in self.violations[:limit]]
        if len(self.violations) > limit:
            lines.append(f"  ... and {len(self.violations) - limit} more")
        return "\n".join(lines)


def validate_schedule(problem: Problem, schedule: Schedule) -> ValidationReport:
    """Check ``schedule`` against all hard constraints (brute-force reference)."""
    if schedule.n_events != problem.n_events:
        raise ValueError("schedule length does not match the number of events")
    spd = problem.grid.slots_per_day
    report = ValidationReport()
    add = report.violations.append

    # ---- decode: which events are placed and which slots they cover ------------
    placed: Dict[int, Tuple[int, int, int]] = {}      # i -> (start, duration, room)
    for i, ev in enumerate(problem.events):
        s, r = schedule.slots[i], schedule.rooms[i]
        if not (0 <= s < problem.n_slots and 0 <= r < problem.n_rooms):
            add(Violation("bounds_violations", (ev.event_id,), 1,
                          f"{ev.event_id} is unassigned or has an invalid slot/room ({s}, {r})"))
            continue
        if s % spd + ev.duration > spd:
            add(Violation("bounds_violations", (ev.event_id,), 1,
                          f"{ev.event_id} starting at {problem.grid.label(s)} runs past the end of the day"))
        placed[i] = (s, ev.duration, r)
    covered = {i: set(range(s, s + d)) for i, (s, d, _) in placed.items()}

    # ---- per-event checks ------------------------------------------------------
    for i, (s, d, r) in placed.items():
        ev, room = problem.events[i], problem.rooms[r]
        if room.capacity < ev.size:
            add(Violation("capacity_violations", (ev.event_id,), 1,
                          f"{ev.event_id} has {ev.size} attendees but room {room.room_id} "
                          f"has only {room.capacity} seats"))
        if room.room_type != ev.room_type:
            add(Violation("room_type_violations", (ev.event_id,), 1,
                          f"{ev.event_id} needs a '{ev.room_type}' room but got "
                          f"{room.room_id} ('{room.room_type}')"))
        blocked = covered[i] & set(problem.teachers[ev.teacher_id].unavailable_slots)
        if blocked:
            add(Violation("availability_violations", (ev.event_id,), 1,
                          f"{ev.event_id}: teacher {ev.teacher_id} is unavailable at "
                          f"{', '.join(problem.grid.label(g) for g in sorted(blocked))}"))

    # ---- pairwise checks (explicit double loop) --------------------------------
    ids = sorted(placed)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if not (covered[i] & covered[j]):
                continue                                   # no common slot -> no conflict
            ei, ej = problem.events[i], problem.events[j]
            when = problem.grid.label(max(placed[i][0], placed[j][0]))
            if placed[i][2] == placed[j][2]:
                add(Violation("room_conflicts", (ei.event_id, ej.event_id), 1,
                              f"{ei.event_id} and {ej.event_id} share room "
                              f"{problem.room_ids[placed[i][2]]} around {when}"))
            if ei.teacher_id == ej.teacher_id:
                add(Violation("teacher_conflicts", (ei.event_id, ej.event_id), 1,
                              f"teacher {ei.teacher_id} must run {ei.event_id} and "
                              f"{ej.event_id} at the same time (around {when})"))
            common = ei.student_ids & ej.student_ids
            if common:
                add(Violation("student_conflicts", (ei.event_id, ej.event_id), len(common),
                              f"{len(common)} student(s) attend both {ei.event_id} and "
                              f"{ej.event_id} which overlap around {when}"))
    return report


# ====================================================================== instance
@dataclass
class InstanceReport:
    """Result of the data sanity checks."""

    errors: List[str] = field(default_factory=list)     # instance cannot be feasible
    warnings: List[str] = field(default_factory=list)   # instance is very tight

    @property
    def possibly_feasible(self) -> bool:
        """False when a necessary condition for feasibility is violated."""
        return not self.errors

    def text(self) -> str:
        lines = ["Instance check: " + ("no obvious infeasibility" if self.possibly_feasible
                                       else "INFEASIBLE DATA")]
        lines += [f"  ERROR   {m}" for m in self.errors]
        lines += [f"  WARNING {m}" for m in self.warnings]
        return "\n".join(lines)


def diagnose_instance(problem: Problem, tight_ratio: float = 0.8) -> InstanceReport:
    """Detect data that makes the hard constraints unsatisfiable (necessary conditions)."""
    rep = InstanceReport()
    n_slots = problem.n_slots

    # 1. every event needs a suitable room and a start slot that fits inside a day
    for i, ev in enumerate(problem.events):
        if not problem.feasible_rooms[i]:
            rep.errors.append(f"{ev.event_id}: no '{ev.room_type}' room has >= {ev.size} seats")
        if not problem.valid_starts[i]:
            rep.errors.append(f"{ev.event_id}: duration {ev.duration} does not fit in a day "
                              f"of {problem.grid.slots_per_day} slots")
            continue
        unavailable = problem.teacher_unavailable[i]
        if not any(not (set(range(g, g + ev.duration)) & set(unavailable))
                   for g in problem.valid_starts[i]):
            rep.errors.append(f"{ev.event_id}: teacher {ev.teacher_id} is unavailable at "
                              f"every possible start slot")

    # 2. total room capacity, per room type
    for rtype in sorted({e.room_type for e in problem.events}):
        units = sum(e.duration for e in problem.events if e.room_type == rtype)
        n_rooms = sum(1 for r in problem.rooms if r.room_type == rtype)
        cap = n_rooms * n_slots
        if units > cap:
            rep.errors.append(f"'{rtype}' events need {units} room-slots but only {cap} exist")
        elif units > tight_ratio * cap:
            rep.warnings.append(f"'{rtype}' rooms are {100 * units / cap:.0f}% utilised (very tight)")

    # 3. teacher load versus their available slots
    for tid, members in problem.teacher_events.items():
        units = sum(problem.durations[i] for i in members)
        free = n_slots - len(problem.teachers[tid].unavailable_slots)
        if units > free:
            rep.errors.append(f"teacher {tid} must teach {units} slots but has only {free} available")
        elif units > tight_ratio * free:
            rep.warnings.append(f"teacher {tid} is {100 * units / free:.0f}% loaded (very tight)")

    # 4. student load versus the length of the planning period
    for sid, members in problem.student_events.items():
        units = sum(problem.durations[i] for i in members)
        if units > n_slots:
            rep.errors.append(f"student {sid} must attend {units} slots but the period has {n_slots}")
        elif units > tight_ratio * n_slots:
            rep.warnings.append(f"student {sid} is {100 * units / n_slots:.0f}% loaded (very tight)")
    return rep
