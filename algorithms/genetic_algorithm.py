"""Genetic Algorithm (memetic) for Smart Academic Scheduler.

Chromosome  : one gene per event, gene = (start slot, room index)  -> ``Schedule``.
Fitness     : ``core.fitness.evaluate`` (hard violations x 1000 + soft penalties), lower is better.
Initial pop : randomised greedy construction (``common.construct``) for part of the
              population, plain random individuals for the rest (diversity).
Selection   : tournament.            Crossover: uniform, gene by gene.
Mutation    : re-draw (slot, room) of an event inside its *feasible* domain
              (teacher availability, room type and capacity already respected).
Local search: offspring that still violate hard constraints are passed through a short
              ``common.repair`` (Lamarckian memetic step).  Without it a plain GA cannot
              reach feasibility on the medium/large instances (hundreds of violations).
Elitism     : the best ``elite_size`` individuals are copied unchanged.
"""

from dataclasses import dataclass
import random

from core.problem import Problem
from core.fitness import evaluate
from models.schedule import Schedule
from algorithms.common import _get_static, construct, repair


@dataclass
class GAConfig:
    population_size: int = 40
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    tournament_size: int = 3
    elite_size: int = 1
    seed: int = 42
    # Memetic part (set repair_rate=0 and construct_fraction=0 for a plain GA).
    construct_fraction: float = 0.5   # share of the initial population built greedily
    repair_rate: float = 0.5          # probability that an infeasible child is repaired
    repair_steps: int = 150           # step budget of one repair call
    initial_repair_steps: int = 1500  # budget used once on the best initial individuals
    initial_repair_count: int = 3     # how many initial individuals get that stronger repair
    final_repair_steps: int = 20000   # last polish of the best individual if it is still infeasible


@dataclass
class GAResult:
    best_schedule: Schedule
    best_fitness: float
    best_hard_violations: int
    best_soft_penalty: float
    history: list


def random_schedule(problem: Problem, rng: random.Random) -> Schedule:
    """Create a complete random schedule using the feasible domains."""
    st = _get_static(problem)
    slots, rooms = [], []
    for i in range(problem.n_events):
        slots.append(rng.choice(st.starts[i]))
        rooms.append(rng.choice(st.rooms[i]))
    return Schedule(slots, rooms)


def tournament_selection(population, scores, rng, size):
    """Choose the candidate with the lowest objective value."""
    contestants = [rng.randrange(len(population)) for _ in range(min(size, len(population)))]
    winner = min(contestants, key=lambda i: scores[i])
    return population[winner]


def crossover(parent1, parent2, rng):
    """Uniform crossover: every event inherits its (slot, room) gene from one parent."""
    n = parent1.n_events
    if n < 2:
        return parent1.copy(), parent2.copy()
    c1, c2 = parent1.copy(), parent2.copy()
    for i in range(n):
        if rng.random() < 0.5:
            c1.slots[i], c2.slots[i] = parent2.slots[i], parent1.slots[i]
            c1.rooms[i], c2.rooms[i] = parent2.rooms[i], parent1.rooms[i]
    return c1, c2


def mutate(schedule, problem, rng, mutation_rate):
    """Reassign an event's time and/or room inside its feasible domain."""
    st = _get_static(problem)
    child = schedule.copy()
    for i in range(problem.n_events):
        if rng.random() < mutation_rate:
            mode = rng.random()
            if mode < 0.4:
                child.slots[i] = rng.choice(st.starts[i])
            elif mode < 0.6:
                child.rooms[i] = rng.choice(st.rooms[i])
            else:
                child.slots[i] = rng.choice(st.starts[i])
                child.rooms[i] = rng.choice(st.rooms[i])
    return child


def _key(result):
    return (result.hard_violations, result.total)


def genetic_algorithm(problem: Problem, config: GAConfig = None) -> GAResult:
    """Run the memetic GA. Lower fitness is better."""
    config = config or GAConfig()
    rng = random.Random(config.seed)

    if config.population_size < 2:
        raise ValueError("population_size must be at least 2")
    if config.generations < 0:
        raise ValueError("generations cannot be negative")
    if not 0.0 <= config.crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0.0 <= config.mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    if config.tournament_size < 1:
        raise ValueError("tournament_size must be positive")
    if not 1 <= config.elite_size < config.population_size:
        raise ValueError("elite_size must be between 1 and population_size-1")
    if not 0.0 <= config.construct_fraction <= 1.0:
        raise ValueError("construct_fraction must be between 0 and 1")
    if not 0.0 <= config.repair_rate <= 1.0:
        raise ValueError("repair_rate must be between 0 and 1")

    n_built = round(config.population_size * config.construct_fraction)
    population = []
    for k in range(config.population_size):
        if k < n_built:
            sched = construct(problem, rng, sample=rng.uniform(0.4, 1.0)).to_schedule()
        else:
            sched = random_schedule(problem, rng)
        population.append(sched)

    results = [evaluate(problem, s) for s in population]
    # Give the most promising initial individuals one stronger repair.
    if config.initial_repair_steps > 0 and config.initial_repair_count > 0:
        order0 = sorted(range(len(population)), key=lambda i: _key(results[i]))
        for i in order0[:config.initial_repair_count]:
            if results[i].hard_violations:
                population[i] = repair(problem, population[i], rng,
                                       max_steps=config.initial_repair_steps)
                results[i] = evaluate(problem, population[i])

    history = []
    best_ever, best_result = None, None

    for generation in range(config.generations + 1):
        if generation > 0:
            results = [evaluate(problem, s) for s in population]
        scores = [_key(r) for r in results]
        order = sorted(range(len(population)), key=lambda i: scores[i])
        current_best = order[0]

        if best_result is None or scores[current_best] < _key(best_result):
            best_ever = population[current_best].copy()
            best_result = results[current_best]

        history.append({
            "generation": generation,
            "best_fitness": best_result.total,
            "best_hard_violations": best_result.hard_violations,
            "best_soft_penalty": best_result.soft_penalty,
            "generation_best_fitness": results[current_best].total,
        })

        if generation == config.generations:
            break

        next_population = [population[i].copy() for i in order[:config.elite_size]]
        while len(next_population) < config.population_size:
            p1 = tournament_selection(population, scores, rng, config.tournament_size)
            p2 = tournament_selection(population, scores, rng, config.tournament_size)
            if rng.random() < config.crossover_rate:
                child1, child2 = crossover(p1, p2, rng)
            else:
                child1, child2 = p1.copy(), p2.copy()
            for child in (child1, child2):
                child = mutate(child, problem, rng, config.mutation_rate)
                if config.repair_rate > 0 and rng.random() < config.repair_rate:
                    if evaluate(problem, child).hard_violations:
                        child = repair(problem, child, rng, max_steps=config.repair_steps)
                if len(next_population) < config.population_size:
                    next_population.append(child)
        population = next_population

    if best_result.hard_violations and config.final_repair_steps > 0:
        polished = repair(problem, best_ever, rng, max_steps=config.final_repair_steps)
        polished_result = evaluate(problem, polished)
        if _key(polished_result) < _key(best_result):
            best_ever, best_result = polished, polished_result
            history[-1]["best_fitness"] = best_result.total
            history[-1]["best_hard_violations"] = best_result.hard_violations
            history[-1]["best_soft_penalty"] = best_result.soft_penalty

    return GAResult(
        best_schedule=best_ever,
        best_fitness=best_result.total,
        best_hard_violations=best_result.hard_violations,
        best_soft_penalty=best_result.soft_penalty,
        history=history,
    )
