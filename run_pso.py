
from time import perf_counter
from pathlib import Path

from core.problem import load_problem
from core.validator import validate_schedule
from algorithms.particle_swarm import (
    PSOConfig,
    particle_swarm_optimization,
)
from models.schedule import save_schedule_csv


def main():
    problem = load_problem("data/lecture/small")

    config = PSOConfig(
        swarm_size=30,
        iterations=100,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=42,
    )

    start = perf_counter()
    result = particle_swarm_optimization(problem, config)
    elapsed = perf_counter() - start

    print("Best fitness:", result.best_fitness)
    print("Hard violations:", result.best_hard_violations)
    print("Soft penalty:", result.best_soft_penalty)
    print("Runtime (seconds):", round(elapsed, 4))

    print("\nLast iterations:")
    for row in result.history[-5:]:
        print(row)

    report = validate_schedule(problem, result.best_schedule)
    print("\nIndependent validation:")
    print(report.text())

    output_dir = Path("results") / "schedules"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_schedule_csv(
        problem,
        result.best_schedule,
        str(output_dir / "pso_best_schedule.csv"),
    )

    print(f"\nSaved: {output_dir / 'pso_best_schedule.csv'}")


if __name__ == "__main__":
    main()
