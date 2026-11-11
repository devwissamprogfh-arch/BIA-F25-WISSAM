"""Ant Colony Optimization for Smart Academic Scheduler.

Construction graph : for every event, the choices are (start slot, room) pairs of its
                     feasible domain; pheromone tau[event][(slot, room)] is learned.
Ant                : places the events one by one (most constrained first).  The
                     probability of a choice is proportional to
                     tau^alpha * eta^beta, where eta = 1 / (1 + extra cost) and the cost
                     is the exact incremental cost given the events already placed
                     (``common.SearchState``; same weights as ``core.fitness``).
Local search       : an ant whose schedule still violates hard constraints is passed
                     through a short ``common.repair`` (ACO + local search, the usual
                     recipe for timetabling); the repaired schedule is what is evaluated
                     and reinforced.
Update             : evaporation, then deposit on the iteration-best and the best-so-far
                     solutions; pheromone is kept inside [tau_min, tau_max] (MAX-MIN rule).
"""

from dataclasses import dataclass
import random

from core.problem import Problem
from core.fitness import evaluate
from models.schedule import Schedule
from algorithms.common import SearchState, _get_static, repair


@dataclass
class ACOConfig:
    ants: int = 12
    iterations: int = 30
    alpha: float = 1.0
    beta: float = 2.0
    evaporation_rate: float = 0.3
    deposit_weight: float = 1.0
    seed: int = 42
    repair_steps: int = 150        # local-search budget per infeasible ant (0 = no local search)
    final_repair_steps: int = 20000  # one last, stronger repair of the best solution if infeasible
    candidate_sample: float = 0.6  # fraction of start slots an ant examines per event


@dataclass
class ACOResult:
    best_schedule: Schedule
    best_fitness: float
    best_hard_violations: int
    best_soft_penalty: float
    history: list


def _key(result):
    return (result.hard_violations, result.total)


def _build_solution(problem, st, order, pheromones, config, rng):
    """One ant: place every event by a pheromone/heuristic roulette wheel."""
    state = SearchState(problem)
    alpha, beta = config.alpha, config.beta
    sample = config.candidate_sample
    for i in order:
        starts = st.starts[i]
        if sample < 1.0 and len(starts) > 4:
            starts = rng.sample(starts, max(4, round(len(starts) * sample)))
        rooms = st.rooms[i]
        tau_i = pheromones[i]
        choices, costs = [], []
        for g in starts:
            h1, s1 = state.start_parts(i, g)
            base = h1 + s1
            for r in rooms:
                h2, s2 = state.room_parts(i, g, r)
                choices.append((g, r))
                costs.append(base + h2 + s2)
        lowest = min(costs)
        weights = []
        for choice, c in zip(choices, costs):
            eta = 1.0 / (1.0 + (c - lowest))
            weights.append(max((tau_i[choice] ** alpha) * (eta ** beta), 1e-300))
        g, r = rng.choices(choices, weights=weights, k=1)[0]
        state.place(i, g, r)
    return state.to_schedule()


def ant_colony_optimization(problem: Problem, config=None):
    """Build schedules with ants and minimize the shared fitness objective."""

    config = config or ACOConfig()
    rng = random.Random(config.seed)

    if config.ants < 1:
        raise ValueError("ants must be at least 1")
    if config.iterations < 1:
        raise ValueError("iterations must be at least 1")
    if config.alpha < 0 or config.beta < 0:
        raise ValueError("alpha and beta cannot be negative")
    if not 0.0 <= config.evaporation_rate < 1.0:
        raise ValueError("evaporation_rate must be in [0, 1)")
    if config.deposit_weight <= 0:
        raise ValueError("deposit_weight must be positive")
    if not 0.0 < config.candidate_sample <= 1.0:
        raise ValueError("candidate_sample must be in (0, 1]")

    st = _get_static(problem)

    # Place the most constrained events first.
    order = sorted(range(problem.n_events), key=lambda i: -st.difficulty[i])

    tau_max = config.deposit_weight / max(config.evaporation_rate, 0.05)
    tau_min = tau_max / 50.0
    pheromones = [
        {(g, r): 1.0 for g in st.starts[i] for r in st.rooms[i]}
        for i in range(problem.n_events)
    ]

    best_schedule, best_result = None, None
    history = []

    for iteration in range(config.iterations):
        iteration_best_schedule, iteration_best_result = None, None

        for _ in range(config.ants):
            schedule = _build_solution(problem, st, order, pheromones, config, rng)
            result = evaluate(problem, schedule)
            if config.repair_steps > 0 and result.hard_violations:
                schedule = repair(problem, schedule, rng, max_steps=config.repair_steps)
                result = evaluate(problem, schedule)

            if iteration_best_result is None or _key(result) < _key(iteration_best_result):
                iteration_best_schedule, iteration_best_result = schedule.copy(), result
            if best_result is None or _key(result) < _key(best_result):
                best_schedule, best_result = schedule.copy(), result

        # Evaporation (kept above tau_min).
        keep = 1.0 - config.evaporation_rate
        for event_pheromones in pheromones:
            for choice in event_pheromones:
                event_pheromones[choice] = max(event_pheromones[choice] * keep, tau_min)

        # Reinforcement: iteration-best and best-so-far.
        for sched, res, share in ((iteration_best_schedule, iteration_best_result, 1.0),
                                  (best_schedule, best_result, 0.5)):
            deposit = share * config.deposit_weight / (1.0 + max(0.0, res.total) / 1000.0)
            for i in range(problem.n_events):
                choice = (sched.slots[i], sched.rooms[i])
                if choice in pheromones[i]:
                    pheromones[i][choice] = min(pheromones[i][choice] + deposit, tau_max)

        history.append({
            "iteration": iteration + 1,
            "best_fitness": best_result.total,
            "best_hard_violations": best_result.hard_violations,
            "best_soft_penalty": best_result.soft_penalty,
            "iteration_best_fitness": iteration_best_result.total,
        })

    # Last attempt: stronger repair of the best-so-far solution if it is still infeasible.
    if best_result.hard_violations and config.final_repair_steps > 0:
        schedule = repair(problem, best_schedule, rng, max_steps=config.final_repair_steps)
        result = evaluate(problem, schedule)
        if _key(result) < _key(best_result):
            best_schedule, best_result = schedule, result
            history[-1]["best_fitness"] = result.total
            history[-1]["best_hard_violations"] = result.hard_violations
            history[-1]["best_soft_penalty"] = result.soft_penalty

    return ACOResult(
        best_schedule=best_schedule,
        best_fitness=best_result.total,
        best_hard_violations=best_result.hard_violations,
        best_soft_penalty=best_result.soft_penalty,
        history=history,
    )
