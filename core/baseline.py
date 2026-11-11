"""Simple reference schedulers (NOT part of GA/PSO/ACO).

* ``random_schedule`` - uniformly random placement; the weakest baseline and a
  convenient source of random individuals/particles for the algorithms.
* ``greedy_schedule`` - places events one by one (hardest first) at the position
  with the smallest immediate penalty. Fast, and a useful sanity reference: a
  smart algorithm that cannot beat it points to a bug or poor parameters.

Run ``python -m core.baseline`` to print the baseline results for all data sets.
"""
from __future__ import annotations

import os
import random
import time
from typing import List, Optional

from core.fitness import evaluate
from core.problem import Problem, load_problem
from models.schedule import Schedule


def random_schedule(problem: Problem, rng: Optional[random.Random] = None) -> Schedule:
    """Place every event at a random valid start slot in a random suitable room."""
    rng = rng or random.Random()
    sched = Schedule.empty(problem.n_events)
    for i in range(problem.n_events):
        starts = problem.valid_starts[i] or list(range(problem.n_slots))
        rooms = problem.feasible_rooms[i] or list(range(problem.n_rooms))
        sched.assign(i, rng.choice(starts), rng.choice(rooms))
    return sched


def greedy_schedule(problem: Problem, rng: Optional[random.Random] = None) -> Schedule:
    """Greedy constructive heuristic (hardest events first, least-penalty position)."""
    rng = rng or random.Random(0)
    HARD = 1000.0
    n = problem.n_events
    # Difficulty = how constrained an event is: many shared students, long, few rooms.
    difficulty = [sum(k for _, k in problem.neighbors[i]) + 5 * problem.durations[i]
                  - 2 * len(problem.feasible_rooms[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -difficulty[i])

    sched = Schedule.empty(n)
    start = [None] * n
    room_busy = set()                       # (room, slot)
    teacher_busy = {}                       # teacher -> set(slot)
    course_days = {}                        # course -> list of days used
    spd = problem.grid.slots_per_day

    for i in order:
        ev, d = problem.events[i], problem.durations[i]
        starts = list(problem.valid_starts[i] or range(problem.n_slots))
        rng.shuffle(starts)                 # random tie-breaking between equal positions
        rooms = list(problem.feasible_rooms[i] or range(problem.n_rooms))
        busy_t = teacher_busy.setdefault(ev.teacher_id, set())
        best = None
        for g in starts:
            span = range(g, g + d)
            # cost of the start slot alone (teacher, students, preferences, spread)
            cost_s = 0.0
            cost_s += HARD * sum(1 for x in span if x in busy_t)
            cost_s += HARD * sum(1 for x in span if x in problem.teacher_unavailable[i])
            for j, shared in problem.neighbors[i]:
                if start[j] is not None and start[j] < g + d and g < start[j] + problem.durations[j]:
                    cost_s += HARD * shared
            pref = problem.teacher_preferred[i]
            if pref and not all(x in pref for x in span):
                cost_s += 2.0
            cost_s += 3.0 * sum(1 for dd in course_days.get(ev.course_id, ()) if dd == g // spd)
            if best is not None and cost_s >= best[0]:
                continue
            for r in rooms:
                cost = cost_s
                if any((r, x) in room_busy for x in span):
                    cost += HARD
                cap = problem.rooms[r].capacity
                cost += (cap - problem.sizes[i]) / cap if cap >= problem.sizes[i] else HARD
                if best is None or cost < best[0]:
                    best = (cost, g, r)
        _, g, r = best
        sched.assign(i, g, r)
        start[i] = g
        for x in range(g, g + d):
            room_busy.add((r, x))
            busy_t.add(x)
        course_days.setdefault(ev.course_id, []).append(g // spd)
    return sched


def _main() -> None:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    print(f"{'dataset':<16}{'method':<8}{'hard':>6}{'soft':>9}{'time(s)':>9}")
    for mode in ("lecture", "exam"):
        for size in ("small", "medium", "large"):
            path = os.path.join(root, mode, size)
            if not os.path.isdir(path):
                continue
            problem = load_problem(path)
            for label, fn in (("random", lambda p: random_schedule(p, random.Random(1))),
                              ("greedy", greedy_schedule)):
                t0 = time.perf_counter()
                sched = fn(problem)
                dt = time.perf_counter() - t0
                res = evaluate(problem, sched)
                print(f"{mode + '/' + size:<16}{label:<8}{res.hard_violations:>6}"
                      f"{res.soft_penalty:>9.1f}{dt:>9.3f}")


if __name__ == "__main__":
    _main()
