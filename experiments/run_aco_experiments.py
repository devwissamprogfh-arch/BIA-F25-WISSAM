
"""Resource-conscious ACO experiments.

Start with one small dataset and a small computational budget.
Results are checkpointed after every completed run.
"""

import csv
from pathlib import Path
from time import perf_counter

from core.problem import load_problem
from core.validator import validate_schedule
from algorithms.ant_colony import ACOConfig, ant_colony_optimization


# Start with ONE dataset. Expand only after checking runtime and temperature.
DATASETS = [
    "data/lecture/small",
]

SEEDS = [42]

# Deliberately small pilot settings.
ANTS = 3
ITERATIONS = 5

OUTPUT_PATH = Path("results/aco_results.csv")

FIELDNAMES = [
    "dataset",
    "seed",
    "best_fitness",
    "hard_violations",
    "soft_penalty",
    "runtime_seconds",
    "valid",
]


def save_results(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=FIELDNAMES
        )
        writer.writeheader()
        writer.writerows(rows)


def run_experiments():
    rows = []

    print("ACO RESOURCE-CONSERVATIVE PILOT")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Seeds per dataset: {len(SEEDS)}")
    print(f"Ants: {ANTS} | Iterations: {ITERATIONS}")
    print("Only one small dataset is enabled initially.")

    for dataset in DATASETS:
        print(f"\nDataset: {dataset}", flush=True)
        problem = load_problem(dataset)

        for seed in SEEDS:
            print(
                f"Starting seed={seed}. "
                "Press Ctrl+C to stop if needed.",
                flush=True,
            )

            config = ACOConfig(
                ants=ANTS,
                iterations=ITERATIONS,
                alpha=1.0,
                beta=2.0,
                evaporation_rate=0.3,
                deposit_weight=1.0,
                seed=seed,
            )

            start = perf_counter()

            result = ant_colony_optimization(problem, config)

            elapsed = perf_counter() - start

            validation = validate_schedule(
                problem, result.best_schedule
            )
            validation_text = validation.text().strip()
            valid = validation_text.startswith("VALID:")

            row = {
                "dataset": dataset,
                "seed": seed,
                "best_fitness": result.best_fitness,
                "hard_violations": result.best_hard_violations,
                "soft_penalty": result.best_soft_penalty,
                "runtime_seconds": round(elapsed, 3),
                "valid": valid,
            }

            rows.append(row)

            # Save immediately after every completed experiment.
            save_results(rows)

            print(
                f"Fitness={result.best_fitness:.4f} | "
                f"Hard={result.best_hard_violations} | "
                f"Soft={result.best_soft_penalty:.4f} | "
                f"Time={elapsed:.2f}s | Valid={valid}",
                flush=True,
            )
            print(validation_text, flush=True)
            print(f"Checkpoint saved: {OUTPUT_PATH}", flush=True)

    print("\nPilot completed.", flush=True)
    print(f"Results saved to: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    try:
        run_experiments()
    except KeyboardInterrupt:
        print(
            "\nStopped by user. Completed results remain saved "
            "in results/aco_results.csv.",
            flush=True,
        )
