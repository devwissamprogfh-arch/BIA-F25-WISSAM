"""Shared incremental search machinery for GA, PSO and ACO.

Why this module exists
----------------------
``core.fitness.evaluate`` is the *official* objective, but it re-scans the whole
schedule on every call (about 1 ms for the large data sets).  Moving ONE event
only changes the penalty of the pairs that involve that event, so the
algorithms can ask a much cheaper question: "what does event ``i`` cost if it is
put at (slot, room), given where all other events currently are?".

:class:`SearchState` answers that question in O(neighbours) time.  The weights
and counting units are exactly those of ``core.constraints`` / ``core.fitness``;
``tests/test_search_state.py`` checks that the incremental numbers always agree
with ``evaluate``.  Final results are still ranked with ``evaluate`` and
confirmed with ``validate_schedule``.

Provided tools
--------------
* :class:`SearchState`  - incremental cost of one event, best-move search.
* ``construct``         - randomised greedy construction (used by GA/PSO to start
                          from reasonable individuals, like ACO ants do).
* ``repair``            - min-conflicts repair: moves only conflicted events until
                          no hard constraint is violated (or the step budget ends).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from core.constraints import FitnessConfig
from core.problem import Problem
from models.schedule import Schedule

UNASSIGNED = -1


class _Static:
    """Per-problem tables that never change (built once, cached on the problem)."""

    def __init__(self, problem: Problem, config: FitnessConfig):
        p = problem
        n = p.n_events
        w = config.weights
        self.w = w
        self.spd = p.grid.slots_per_day
        self.dur = list(p.durations)

        # Search domains: starts inside one day that avoid teacher unavailability.
        self.starts: List[List[int]] = []
        self.rooms: List[List[int]] = []
        for i in range(n):
            d, unav = p.durations[i], p.teacher_unavailable[i]
            starts = [g for g in p.valid_starts[i]
                      if not unav or not any(x in unav for x in range(g, g + d))]
            if not starts:                       # data problem: keep something placeable
                starts = list(p.valid_starts[i]) or list(range(p.n_slots))
            rooms = list(p.feasible_rooms[i])
            if not rooms:
                rooms = [k for k, r in enumerate(p.rooms)
                         if r.room_type == p.events[i].room_type] or list(range(p.n_rooms))
            self.starts.append(starts)
            self.rooms.append(rooms)

        # Unary soft terms.
        self.pref: List[Dict[int, float]] = []
        for i in range(n):
            pref = p.teacher_preferred[i]
            d = p.durations[i]
            if pref:
                self.pref.append({g: (w["preference_violations"]
                                      if not all(x in pref for x in range(g, g + d)) else 0.0)
                                  for g in self.starts[i]})
            else:
                self.pref.append({})
        self.waste: List[Dict[int, float]] = []
        for i in range(n):
            size = p.sizes[i]
            tab = {}
            for r in self.rooms[i]:
                cap = p.rooms[r].capacity
                tab[r] = w["room_waste"] * (cap - size) / cap if cap >= size else 0.0
            self.waste.append(tab)

        teacher_mates = [[] for _ in range(n)]
        for members in p.teacher_events.values():
            for i in members:
                teacher_mates[i] = [j for j in members if j != i]
        course_mates = [[] for _ in range(n)]
        for members in p.course_events.values():
            for i in members:
                course_mates[i] = [j for j in members if j != i]
        self.teacher_mates = teacher_mates
        self.course_mates = course_mates
        self.neighbors = [list(nb) for nb in p.neighbors]
        self.min_gap = p.min_gap_slots

        # Difficulty used to order constructive placement (hardest first).
        self.difficulty = [
            sum(k for _, k in p.neighbors[i]) + len(teacher_mates[i]) * 4
            + 5 * p.durations[i] - 2 * len(self.rooms[i]) - len(self.starts[i]) / 8.0
            for i in range(n)]


def _get_static(problem: Problem, config: Optional[FitnessConfig] = None) -> _Static:
    cfg = config or FitnessConfig()
    cached = getattr(problem, "_search_static", None)
    if cached is None or cached[0] != cfg.weights:
        cached = (dict(cfg.weights), _Static(problem, cfg))
        problem._search_static = cached
    return cached[1]


class SearchState:
    """A (possibly partial) schedule with occupancy tables for fast move costs."""

    def __init__(self, problem: Problem, schedule: Optional[Schedule] = None,
                 config: Optional[FitnessConfig] = None):
        self.p = problem
        self.st = _get_static(problem, config)
        n = problem.n_events
        self.n = n
        self.start: List[int] = [UNASSIGNED] * n
        self.room: List[int] = [UNASSIGNED] * n
        self.occ: List[List[List[int]]] = [
            [[] for _ in range(problem.n_slots)] for _ in range(problem.n_rooms)]
        if schedule is not None:
            for i in range(n):
                s, r = schedule.slots[i], schedule.rooms[i]
                if 0 <= s < problem.n_slots and 0 <= r < problem.n_rooms:
                    self.place(i, s, r)

    # ---------------------------------------------------------------- bookkeeping
    def place(self, i: int, g: int, r: int) -> None:
        """Put event ``i`` at (g, r); it must currently be unplaced."""
        self.start[i], self.room[i] = g, r
        occ_r = self.occ[r]
        for x in range(g, min(g + self.st.dur[i], len(occ_r))):
            occ_r[x].append(i)

    def unplace(self, i: int) -> None:
        g, r = self.start[i], self.room[i]
        if g == UNASSIGNED:
            return
        occ_r = self.occ[r]
        for x in range(g, min(g + self.st.dur[i], len(occ_r))):
            occ_r[x].remove(i)
        self.start[i] = self.room[i] = UNASSIGNED

    def move(self, i: int, g: int, r: int) -> Set[int]:
        """Move event ``i``; return the set of events whose conflict status may change."""
        affected: Set[int] = {i}
        affected.update(self.st.teacher_mates[i])
        affected.update(j for j, _ in self.st.neighbors[i])
        d = self.st.dur[i]
        old_g, old_r = self.start[i], self.room[i]
        if old_g != UNASSIGNED:
            for x in range(old_g, old_g + d):
                affected.update(self.occ[old_r][x])
        self.unplace(i)
        self.place(i, g, r)
        for x in range(g, g + d):
            affected.update(self.occ[r][x])
        return affected

    def to_schedule(self) -> Schedule:
        return Schedule(self.start, self.room)

    # ---------------------------------------------------------------- cost pieces
    def start_parts(self, i: int, g: int) -> Tuple[float, float]:
        """(hard, soft) cost of event ``i`` starting at ``g`` that does not depend on its room."""
        st = self.st
        w = st.w
        end = g + st.dur[i]
        start = self.start
        dur = st.dur
        hard = 0.0
        soft = st.pref[i].get(g, 0.0)

        hits = 0
        for j in st.teacher_mates[i]:
            sj = start[j]
            if sj != UNASSIGNED and sj < end and g < sj + dur[j]:
                hits += 1
        hard += w["teacher_conflicts"] * hits

        spd = st.spd
        day = g // spd
        min_gap = st.min_gap
        wsc, wgap = w["student_conflicts"], w["gap_violations"]
        for j, shared in st.neighbors[i]:
            sj = start[j]
            if sj == UNASSIGNED:
                continue
            ej = sj + dur[j]
            if g < ej and sj < end:
                hard += wsc * shared
            elif min_gap > 0 and sj // spd == day:
                gap = sj - end if sj >= end else g - ej
                if gap < min_gap:
                    soft += wgap * shared

        same_day = 0
        for j in st.course_mates[i]:
            sj = start[j]
            if sj != UNASSIGNED and sj // spd == day:
                same_day += 1
        soft += w["spread_violations"] * same_day
        return hard, soft

    def room_parts(self, i: int, g: int, r: int) -> Tuple[float, float]:
        """(hard, soft) cost that depends on the room: room clashes and wasted seats."""
        occ_r = self.occ[r]
        seen: Set[int] = set()
        for x in range(g, g + self.st.dur[i]):
            for j in occ_r[x]:
                if j != i:
                    seen.add(j)
        return self.st.w["room_conflicts"] * len(seen), self.st.waste[i].get(r, 0.0)

    def cost(self, i: int, g: int, r: int) -> Tuple[float, float]:
        h1, s1 = self.start_parts(i, g)
        h2, s2 = self.room_parts(i, g, r)
        return h1 + h2, s1 + s2

    def hard_of(self, i: int) -> float:
        """Hard cost currently caused by event ``i`` (0 when it is conflict-free)."""
        g, r = self.start[i], self.room[i]
        if g == UNASSIGNED:
            return 0.0
        return self.start_parts(i, g)[0] + self.room_parts(i, g, r)[0]

    def pair_hard(self, i: int, g: int, r: int) -> Dict[int, float]:
        """Hard cost of putting event ``i`` at (g, r), split per partner event.

        Returns ``{j: weight}``; the sum of the values equals ``cost(i, g, r)[0]``.
        Used for O(neighbours) incremental conflict bookkeeping in ``repair``.
        """
        st = self.st
        w = st.w
        end = g + st.dur[i]
        start, dur = self.start, st.dur
        out: Dict[int, float] = {}
        wt = w["teacher_conflicts"]
        for j in st.teacher_mates[i]:
            sj = start[j]
            if sj != UNASSIGNED and sj < end and g < sj + dur[j]:
                out[j] = out.get(j, 0.0) + wt
        wsc = w["student_conflicts"]
        for j, shared in st.neighbors[i]:
            sj = start[j]
            if sj != UNASSIGNED and sj < end and g < sj + dur[j]:
                out[j] = out.get(j, 0.0) + wsc * shared
        wr = w["room_conflicts"]
        occ_r = self.occ[r]
        seen: Set[int] = set()
        for x in range(g, min(end, len(occ_r))):
            for j in occ_r[x]:
                if j != i and j not in seen:
                    seen.add(j)
                    out[j] = out.get(j, 0.0) + wr
        return out

    def conflicted_events(self) -> Set[int]:
        return {i for i in range(self.n) if self.hard_of(i) > 0}

    # ---------------------------------------------------------------- move search
    def best_move(self, i: int, rng: random.Random, *, exclude_current: bool = False,
                  sample: float = 1.0, forbidden: Optional[Set[Tuple[int, int]]] = None
                  ) -> Optional[Tuple[int, int, float]]:
        """Best (slot, room) for event ``i`` given all other events.

        ``sample`` < 1 looks only at a random fraction of the start slots (cheaper and
        more diverse).  Ties are broken at random.  Returns ``(g, r, cost)``.
        """
        st = self.st
        starts = st.starts[i]
        if sample < 1.0:
            k = max(1, int(round(len(starts) * sample)))
            starts = rng.sample(starts, k)
        cur = (self.start[i], self.room[i]) if exclude_current else None
        best = float("inf")
        ties: List[Tuple[int, int]] = []
        rooms = st.rooms[i]
        for g in starts:
            h, s = self.start_parts(i, g)
            base = h + s
            if base > best:
                continue
            for r in rooms:
                if cur is not None and (g, r) == cur:
                    continue
                if forbidden and (g, r) in forbidden:
                    continue
                h2, s2 = self.room_parts(i, g, r)
                t = base + h2 + s2
                if t < best - 1e-12:
                    best, ties = t, [(g, r)]
                elif t <= best + 1e-12:
                    ties.append((g, r))
        if not ties:
            return None
        g, r = rng.choice(ties)
        return g, r, best


def construct(problem: Problem, rng: random.Random, *, sample: float = 0.5,
              jitter: float = 3.0, config: Optional[FitnessConfig] = None) -> SearchState:
    """Randomised greedy construction: hardest events first, best (sampled) position.

    ``jitter`` perturbs the difficulty order, ``sample`` limits the start slots that
    are examined for each event; both make successive calls produce different schedules.
    """
    state = SearchState(problem, None, config)
    st = state.st
    order = sorted(range(problem.n_events),
                   key=lambda i: -(st.difficulty[i] + rng.uniform(0, jitter)))
    for i in order:
        move = state.best_move(i, rng, sample=sample)
        if move is None:                         # cannot happen with non-empty domains
            move = (st.starts[i][0], st.rooms[i][0], 0.0)
        state.place(i, move[0], move[1])
    return state


def repair(problem: Problem, schedule: Schedule, rng: random.Random, *,
           max_steps: int = 200, noise: float = 0.02, tabu_tenure: int = 0,
           patience: int = 0, config: Optional[FitnessConfig] = None) -> Schedule:
    """Breakout (weighted min-conflicts) repair of the hard constraints.

    Only events that take part in a hard conflict are touched.  A chosen event moves
    to the (slot, room) with the lowest *weighted* conflict cost.  When it cannot
    improve (a local minimum) the weights of the pairs it clashes with are increased
    ("breakout"), which reshapes the landscape and lets the search escape plateaus
    that plain min-conflicts / tabu search cannot leave.  Conflict bookkeeping is
    incremental, so a step costs O(neighbours).  The best schedule seen is returned
    (a feasible one as soon as it is found).  ``tabu_tenure``/``patience`` are kept
    only for backward compatibility.
    """
    state = SearchState(problem, schedule, config)
    st = state.st
    for i in range(problem.n_events):                  # place anything still unplaced
        if state.start[i] == UNASSIGNED:
            mv = state.best_move(i, rng)
            state.place(i, mv[0], mv[1])

    n = problem.n_events
    w = st.w
    wt, wsc, wr = w["teacher_conflicts"], w["student_conflicts"], w["room_conflicts"]
    cnt = [state.hard_of(i) for i in range(n)]
    conflicted = {i for i in range(n) if cnt[i] > 0}
    if not conflicted:
        return state.to_schedule()
    total = sum(cnt) / 2.0
    best_total, best_sched = total, state.to_schedule()
    mult: Dict[int, float] = {}                         # pair (i*n+j, i<j) -> weight factor
    start, dur = state.start, st.dur

    def pair_key(a: int, b: int) -> int:
        return a * n + b if a < b else b * n + a

    for _ in range(max_steps):
        i = rng.choice(tuple(conflicted))
        og, orr = start[i], state.room[i]
        d_i = dur[i]
        # -- weighted cost of every start slot (student + teacher clashes) --------------
        nbrs, mates = st.neighbors[i], st.teacher_mates[i]
        best_cost, ties = float("inf"), []
        cur_cost = None
        cand_starts = st.starts[i]
        for g in cand_starts:
            end = g + d_i
            base = 0.0
            for j, shared in nbrs:
                sj = start[j]
                if sj < end and g < sj + dur[j]:
                    base += wsc * shared * mult.get(pair_key(i, j), 1.0)
            for j in mates:
                sj = start[j]
                if sj < end and g < sj + dur[j]:
                    base += wt * mult.get(pair_key(i, j), 1.0)
            if base > best_cost and g != og:
                continue
            pref = st.pref[i].get(g, 0.0) * 1e-3
            for r in st.rooms[i]:
                rc = 0.0
                occ_r = state.occ[r]
                seen = set()
                for x in range(g, end):
                    for j in occ_r[x]:
                        if j != i and j not in seen:
                            seen.add(j)
                            rc += wr * mult.get(pair_key(i, j), 1.0)
                c = base + rc + pref + st.waste[i].get(r, 0.0) * 1e-3
                if g == og and r == orr:
                    cur_cost = c
                    continue
                if c < best_cost - 1e-9:
                    best_cost, ties = c, [(g, r)]
                elif c <= best_cost + 1e-9:
                    ties.append((g, r))
        if cur_cost is None:
            cur_cost = float("inf")
        if rng.random() < noise:
            mv = (rng.choice(cand_starts), rng.choice(st.rooms[i]))
        elif ties and best_cost < cur_cost - 1e-9:
            mv = rng.choice(ties)
        else:
            # local minimum for this event: raise the weight of its current clashes
            for j in state.pair_hard(i, og, orr):
                k = pair_key(i, j)
                mult[k] = mult.get(k, 1.0) + 1.0
            if ties and rng.random() < 0.5:             # sideways move on plateaus
                mv = rng.choice(ties)
            else:
                continue
        g, r = mv
        old_pairs = state.pair_hard(i, og, orr)
        state.unplace(i)
        state.place(i, g, r)
        new_pairs = state.pair_hard(i, g, r)
        for j, wgt in old_pairs.items():
            cnt[j] -= wgt
            if cnt[j] <= 1e-9:
                cnt[j] = 0.0
                conflicted.discard(j)
        for j, wgt in new_pairs.items():
            cnt[j] += wgt
            conflicted.add(j)
        total += sum(new_pairs.values()) - sum(old_pairs.values())
        cnt[i] = sum(new_pairs.values())
        if cnt[i] > 0:
            conflicted.add(i)
        else:
            conflicted.discard(i)
        if total < best_total - 1e-9:
            best_total, best_sched = total, state.to_schedule()
            if not conflicted:
                return best_sched
    return best_sched
