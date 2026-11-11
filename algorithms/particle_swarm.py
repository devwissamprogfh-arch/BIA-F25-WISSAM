"""Hybrid Particle Swarm Optimization for Smart Academic Scheduler.

Each particle is a flat real vector [slot0, room0, slot1, room1, ...] holding an
*index* into the event's feasible start-slot / room domain.  Velocities follow the
standard PSO update; positions are rounded to indices when decoded.  Because a pure
PSO cannot satisfy hundreds of coupled hard constraints, every decoded particle that
still violates a hard constraint is passed through ``common.repair`` (bounded
min-conflicts / breakout local search) and its position is replaced by the repaired
schedule (Lamarckian hybrid).  All solutions are evaluated with the shared
core.fitness.evaluate() objective.

Lower fitness is better, but candidates are compared lexicographically:
1. Fewer hard violations are always preferred.
2. When hard-violation counts are equal, lower total fitness is preferred.
"""

from dataclasses import dataclass
import random

from core.problem import Problem
from core.fitness import evaluate
from models.schedule import Schedule
from algorithms.common import _get_static, construct, repair


@dataclass
class PSOConfig:
    swarm_size: int = 30
    iterations: int = 100
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    seed: int = 42

    # Hybrid PSO: bounded local repair (min-conflicts / breakout, see common.repair).
    repair_enabled: bool = True
    repair_steps: int = 150            # step budget per repaired particle
    initial_repair_steps: int = 1500   # stronger repair for the best initial particles
    initial_repair_count: int = 3
    final_repair_steps: int = 20000    # last polish of the global best if it is still infeasible
    construct_fraction: float = 0.5    # share of the swarm initialised by greedy construction
    # Kept for backward compatibility (unused now).
    repair_passes: int = 1
    repair_max_events: int = 3
    repair_candidates_per_event: int = 4


@dataclass
class PSOResult:
    best_schedule: Schedule
    best_fitness: float
    best_hard_violations: int
    best_soft_penalty: float
    history: list


def _result_key(result) -> tuple:
    """Prioritize fewer hard violations, then minimize total fitness."""
    return (result.hard_violations, result.total)


def get_slot_domain(problem: Problem, event_index: int) -> list:
    """Return valid starts that do not overlap teacher unavailability."""
    return _get_static(problem).starts[event_index]


def get_room_domain(problem: Problem, event_index: int) -> list:
    """Return rooms of the right type with enough seats (never empty)."""
    return _get_static(problem).rooms[event_index]


def decode_position(problem: Problem, position: list) -> Schedule:
    """Decode continuous particle coordinates into discrete choices."""
    slots = []
    rooms = []

    for i in range(problem.n_events):
        slot_domain = get_slot_domain(problem, i)
        room_domain = get_room_domain(problem, i)

        slot_index = 2 * i
        room_index = slot_index + 1

        if slot_domain:
            idx = round(position[slot_index])
            idx = max(0, min(idx, len(slot_domain) - 1))
            slots.append(slot_domain[idx])
        else:
            slots.append(-1)

        if room_domain:
            idx = round(position[room_index])
            idx = max(0, min(idx, len(room_domain) - 1))
            rooms.append(room_domain[idx])
        else:
            rooms.append(-1)

    return Schedule(slots, rooms)


def random_particle(problem: Problem, rng: random.Random):
    """Create a particle and its coordinate limits."""
    position = []
    velocity = []
    limits = []

    for i in range(problem.n_events):
        slot_domain = get_slot_domain(problem, i)
        room_domain = get_room_domain(problem, i)

        for domain in (slot_domain, room_domain):
            if domain:
                upper = len(domain) - 1
                position.append(rng.uniform(0, upper))
                limits.append((0.0, float(upper)))
            else:
                position.append(0.0)
                limits.append((0.0, 0.0))

            velocity.append(0.0)

    return position, velocity, limits


def _repair_schedule(
    problem: Problem,
    schedule: Schedule,
    config: PSOConfig,
    rng: random.Random,
    steps: int = None,
) -> Schedule:
    """Bounded local repair (only when the schedule has hard violations)."""
    if not config.repair_enabled:
        return schedule
    if evaluate(problem, schedule).hard_violations == 0:
        return schedule
    return repair(problem, schedule, rng,
                  max_steps=config.repair_steps if steps is None else steps)


def _encode_schedule_position(
    problem: Problem,
    schedule: Schedule,
    position: list,
) -> list:
    """Map a repaired discrete schedule back to particle coordinates."""
    encoded = position.copy()

    for i in range(problem.n_events):
        slot_domain = get_slot_domain(problem, i)
        room_domain = get_room_domain(problem, i)

        if schedule.slots[i] in slot_domain:
            encoded[2 * i] = float(slot_domain.index(schedule.slots[i]))

        if schedule.rooms[i] in room_domain:
            encoded[2 * i + 1] = float(room_domain.index(schedule.rooms[i]))

    return encoded


def particle_swarm_optimization(
    problem: Problem,
    config: PSOConfig = None,
) -> PSOResult:
    """Run Hybrid PSO. Fewer hard violations take priority; lower fitness wins ties."""
    config = config or PSOConfig()

    if config.swarm_size < 2:
        raise ValueError("swarm_size must be at least 2")
    if config.iterations < 0:
        raise ValueError("iterations cannot be negative")
    if config.inertia < 0:
        raise ValueError("inertia cannot be negative")
    if config.cognitive < 0 or config.social < 0:
        raise ValueError("cognitive and social must be non-negative")
    if config.repair_passes < 0:
        raise ValueError("repair_passes cannot be negative")
    if config.repair_max_events < 0:
        raise ValueError("repair_max_events cannot be negative")
    if config.repair_candidates_per_event < 1:
        raise ValueError("repair_candidates_per_event must be at least 1")

    rng = random.Random(config.seed)

    particles = [
        random_particle(problem, rng)
        for _ in range(config.swarm_size)
    ]
    n_built = round(config.swarm_size * config.construct_fraction)

    # Initial schedules: part greedy-constructed, part random-decoded.
    schedules = []
    for p, (position, velocity, limits) in enumerate(particles):
        if p < n_built:
            schedules.append(construct(problem, rng, sample=rng.uniform(0.4, 1.0)).to_schedule())
        else:
            schedules.append(decode_position(problem, position))
    init_results = [evaluate(problem, s) for s in schedules]
    strong = sorted(range(config.swarm_size), key=lambda i: _result_key(init_results[i]))
    strong = set(strong[:config.initial_repair_count])

    personal_positions = []
    personal_keys = []
    for p, (position, velocity, limits) in enumerate(particles):
        schedule = schedules[p]
        steps = config.initial_repair_steps if p in strong else config.repair_steps
        schedule = _repair_schedule(problem, schedule, config, rng, steps)
        position = _encode_schedule_position(problem, schedule, position)
        particles[p] = (position, velocity, limits)

        result = evaluate(problem, schedule)
        personal_positions.append(position.copy())
        personal_keys.append(_result_key(result))

    global_index = min(
        range(config.swarm_size),
        key=lambda i: personal_keys[i],
    )

    global_position = personal_positions[global_index].copy()
    global_schedule = decode_position(problem, global_position)
    # The selected initial personal best is already repaired and encoded.
    global_result = evaluate(problem, global_schedule)

    history = []

    for iteration in range(config.iterations + 1):
        history.append({
            "iteration": iteration,
            "best_fitness": global_result.total,
            "best_hard_violations": global_result.hard_violations,
            "best_soft_penalty": global_result.soft_penalty,
        })

        if iteration == config.iterations:
            break

        for p in range(config.swarm_size):
            position, velocity, limits = particles[p]

            for d in range(len(position)):
                r1 = rng.random()
                r2 = rng.random()

                velocity[d] = (
                    config.inertia * velocity[d]
                    + config.cognitive * r1
                    * (personal_positions[p][d] - position[d])
                    + config.social * r2
                    * (global_position[d] - position[d])
                )

                position[d] += velocity[d]
                low, high = limits[d]
                position[d] = max(low, min(position[d], high))

            schedule = decode_position(problem, position)
            schedule = _repair_schedule(problem, schedule, config, rng)
            position = _encode_schedule_position(problem, schedule, position)
            particles[p] = (position, velocity, limits)

            result = evaluate(problem, schedule)
            result_key = _result_key(result)

            if result_key < personal_keys[p]:
                personal_positions[p] = position.copy()
                personal_keys[p] = result_key

                if result_key < _result_key(global_result):
                    global_position = position.copy()
                    global_schedule = schedule.copy()
                    global_result = result

    if global_result.hard_violations and config.repair_enabled and config.final_repair_steps > 0:
        polished = repair(problem, global_schedule, rng, max_steps=config.final_repair_steps)
        polished_result = evaluate(problem, polished)
        if _result_key(polished_result) < _result_key(global_result):
            global_schedule, global_result = polished, polished_result
            history[-1]["best_fitness"] = global_result.total
            history[-1]["best_hard_violations"] = global_result.hard_violations
            history[-1]["best_soft_penalty"] = global_result.soft_penalty

    return PSOResult(
        best_schedule=global_schedule,
        best_fitness=global_result.total,
        best_hard_violations=global_result.hard_violations,
        best_soft_penalty=global_result.soft_penalty,
        history=history,
    )
