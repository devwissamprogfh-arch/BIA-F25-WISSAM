"""Shared fitness / objective function (to be MINIMISED).

    penalty = sum_c  w_c * violations_c          (c over all constraints)

The hard weights dominate, so the algorithms first learn to remove conflicts and
then to improve the soft quality criteria. All three algorithms must call
:func:`evaluate` / :func:`fitness` - they are not allowed to define their own
objective (this is what makes the comparison fair).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from core.constraints import HARD_NAMES, SOFT_NAMES, FitnessConfig
from core.problem import Problem
from models.schedule import Schedule

# An unplaced event gets a zero-length interval far away from the real timeline.
# A zero-length interval overlaps nothing, so unplaced events never create
# (double-counted) conflicts; they are only reported as bounds violations.
_NOWHERE = -10 ** 6

_DEFAULT_CONFIG = FitnessConfig()


@dataclass
class FitnessResult:
    """Detailed evaluation of one schedule."""

    counts: Dict[str, float]     # violations per constraint (unweighted)
    hard_penalty: float          # weighted sum over hard constraints
    soft_penalty: float          # weighted sum over soft constraints

    @property
    def total(self) -> float:
        """The scalar objective value (lower is better)."""
        return self.hard_penalty + self.soft_penalty

    @property
    def hard_violations(self) -> int:
        """Unweighted number of hard violations."""
        return int(sum(self.counts[n] for n in HARD_NAMES))

    @property
    def feasible(self) -> bool:
        """True iff the schedule respects every hard constraint."""
        return self.hard_violations == 0

    def as_dict(self) -> Dict[str, float]:
        out = dict(self.counts)
        out.update(hard_penalty=self.hard_penalty, soft_penalty=self.soft_penalty,
                   total=self.total, hard_violations=self.hard_violations,
                   feasible=self.feasible)
        return out


def _count_overlapping_pairs(indices: List[int], start: List[int], end: List[int]) -> int:
    """Number of overlapping pairs among ``indices`` (sweep over start times)."""
    if len(indices) < 2:
        return 0
    order = sorted(indices, key=start.__getitem__)
    count = 0
    for a_pos, a in enumerate(order):
        end_a = end[a]
        for b in order[a_pos + 1:]:
            if start[b] >= end_a:      # sorted by start: no later event can overlap a
                break
            count += 1
    return count


def evaluate(problem: Problem, schedule: Schedule,
             config: Optional[FitnessConfig] = None) -> FitnessResult:
    """Evaluate ``schedule`` and return per-constraint counts and weighted totals."""
    cfg = config or _DEFAULT_CONFIG
    n = problem.n_events
    if schedule.n_events != n:
        raise ValueError(f"schedule has {schedule.n_events} events, problem has {n}")
    spd, n_slots, n_rooms = problem.grid.slots_per_day, problem.n_slots, problem.n_rooms
    counts: Dict[str, float] = {name: 0 for name in HARD_NAMES + SOFT_NAMES}

    # ---- 1. decode the schedule into start/end/room arrays -------------------
    start = [_NOWHERE] * n
    end = [_NOWHERE] * n
    room = [-1] * n
    placed: List[int] = []
    for i in range(n):
        s, r = schedule.slots[i], schedule.rooms[i]
        if not (0 <= s < n_slots and 0 <= r < n_rooms):
            counts["bounds_violations"] += 1          # unassigned / out of range
            continue
        d = problem.durations[i]
        if s % spd + d > spd:
            counts["bounds_violations"] += 1          # runs past the end of the day
        start[i], end[i], room[i] = s, s + d, r       # still placed: other checks apply
        placed.append(i)

    # ---- 2. per-event checks ---------------------------------------------------
    rooms = problem.rooms
    events = problem.events
    for i in placed:
        r_obj = rooms[room[i]]
        size = problem.sizes[i]
        if r_obj.capacity < size:
            counts["capacity_violations"] += 1
        else:
            counts["room_waste"] += (r_obj.capacity - size) / r_obj.capacity
        if r_obj.room_type != events[i].room_type:
            counts["room_type_violations"] += 1
        span = range(start[i], end[i])
        unavailable = problem.teacher_unavailable[i]
        if unavailable and any(g in unavailable for g in span):
            counts["availability_violations"] += 1
        preferred = problem.teacher_preferred[i]
        if preferred and not all(g in preferred for g in span):
            counts["preference_violations"] += 1

    # ---- 3. room and teacher conflicts (pairs of overlapping events) ----------
    by_room: Dict[int, List[int]] = {}
    for i in placed:
        by_room.setdefault(room[i], []).append(i)
    for members in by_room.values():
        counts["room_conflicts"] += _count_overlapping_pairs(members, start, end)
    for members in problem.teacher_events.values():
        counts["teacher_conflicts"] += _count_overlapping_pairs(
            [i for i in members if end[i] != _NOWHERE], start, end)

    # ---- 4. student conflicts and close events (pre-computed shared pairs) -----
    min_gap = problem.min_gap_slots
    for (i, j), shared in problem.shared_pairs.items():
        si, ei, sj, ej = start[i], end[i], start[j], end[j]
        if si < ej and sj < ei:
            counts["student_conflicts"] += shared
        elif min_gap > 0 and ei != _NOWHERE and ej != _NOWHERE and si // spd == sj // spd:
            gap = sj - ei if sj >= ei else si - ej       # rest between the two events
            if gap < min_gap:
                counts["gap_violations"] += shared

    # ---- 5. spread of the sessions of one course over the days -----------------
    for members in problem.course_events.values():
        if len(members) > 1:
            per_day: Dict[int, int] = {}
            for i in members:
                if end[i] != _NOWHERE:
                    d = start[i] // spd
                    per_day[d] = per_day.get(d, 0) + 1
            counts["spread_violations"] += sum(k * (k - 1) // 2 for k in per_day.values())

    # ---- 6. weighted totals ---------------------------------------------------
    w = cfg.weights
    hard = sum(w[name] * counts[name] for name in HARD_NAMES)
    soft = sum(w[name] * counts[name] for name in SOFT_NAMES)
    return FitnessResult(counts, hard, soft)


def fitness(problem: Problem, schedule: Schedule,
            config: Optional[FitnessConfig] = None) -> float:
    """Scalar objective value (lower is better) - the function algorithms optimise."""
    return evaluate(problem, schedule, config).total


def is_feasible(problem: Problem, schedule: Schedule) -> bool:
    """True iff the schedule has no hard violation according to the fitness function."""
    return evaluate(problem, schedule).feasible
