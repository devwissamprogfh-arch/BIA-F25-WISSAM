
from pathlib import Path
from time import perf_counter

from algorithms.ant_colony import (
    ACOConfig,
    ant_colony_optimization,
)
from core.problem import load_problem
from core.validator import validate_schedule
from models.schedule import save_schedule_csv


def main():
    problem = load_problem("data/lecture/small")

    config = ACOConfig(
        ants=12,
        iterations=30,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.3,
        deposit_weight=1.0,
        seed=42,
    )

    start_time = perf_counter()
    result = ant_colony_optimization(problem, config)
    elapsed = perf_counter() - start_time

    print("=" * 45)
    print("ANT COLONY OPTIMIZATION (ACO)")
    print("=" * 45)

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

    output_path = output_dir / "aco_best_schedule.csv"

    save_schedule_csv(
        problem,
        result.best_schedule,
        str(output_path),
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
