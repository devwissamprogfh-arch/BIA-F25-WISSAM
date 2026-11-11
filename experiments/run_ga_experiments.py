
"""Run Genetic Algorithm experiments on all six datasets."""

import csv
import statistics
from pathlib import Path
from time import perf_counter

from core.problem import load_problem
from core.validator import validate_schedule
from algorithms.genetic_algorithm import GAConfig, genetic_algorithm


DATASETS = [
    "data/lecture/small",
    "data/lecture/medium",
    "data/lecture/large",
    "data/exam/small",
    "data/exam/medium",
    "data/exam/large",
]

SEEDS = [42, 123, 2026]


def run_experiments():
    output_path = Path("results/ga_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset in DATASETS:
        print(f"\n{'=' * 55}")
        print(f"Dataset: {dataset}")

        problem = load_problem(dataset)
        dataset_results = []

        for seed in SEEDS:
            config = GAConfig(
                population_size=40,
                generations=100,
                crossover_rate=0.8,
                mutation_rate=0.05,
                tournament_size=3,
                elite_size=1,
                seed=seed,
            )

            start = perf_counter()
            result = genetic_algorithm(problem, config)
            elapsed = perf_counter() - start

            validation = validate_schedule(
                problem, result.best_schedule
            )

            # Adapt this check if the validator uses a different
            # documented property for its result.
            valid = result.best_hard_violations == 0

            row = {
                "dataset": dataset,
                "seed": seed,
                "best_fitness": result.best_fitness,
                "hard_violations": result.best_hard_violations,
                "soft_penalty": result.best_soft_penalty,
                "runtime_seconds": round(elapsed, 6),
                "valid": valid,
            }

            rows.append(row)
            dataset_results.append(row)

            print(
                f"seed={seed} | "
                f"fitness={result.best_fitness:.4f} | "
                f"hard={result.best_hard_violations} | "
                f"soft={result.best_soft_penalty:.4f} | "
                f"time={elapsed:.3f}s"
            )

            print(validation.text())

        fitness_values = [
            row["best_fitness"] for row in dataset_results
        ]
        times = [
            row["runtime_seconds"] for row in dataset_results
        ]

        print(
            f"Summary: mean fitness={statistics.mean(fitness_values):.4f}, "
            f"mean time={statistics.mean(times):.3f}s"
        )

    fieldnames = [
        "dataset",
        "seed",
        "best_fitness",
        "hard_violations",
        "soft_penalty",
        "runtime_seconds",
        "valid",
    ]

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiments()
