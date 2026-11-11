"""Tests for the search machinery (common.py) and the three metaheuristics."""
import random
import unittest

from algorithms.ant_colony import ACOConfig, ant_colony_optimization
from algorithms.common import SearchState, construct, repair
from algorithms.genetic_algorithm import GAConfig, genetic_algorithm
from algorithms.particle_swarm import PSOConfig, particle_swarm_optimization
from core.baseline import random_schedule
from core.fitness import evaluate
from core.problem import load_problem
from core.validator import validate_schedule

DATASETS = ["data/lecture/small", "data/exam/small", "data/lecture/medium"]


class SearchStateTests(unittest.TestCase):
    def test_incremental_cost_matches_evaluate(self):
        """Sum of per-event hard cost / 2 must equal evaluate()'s hard penalty."""
        for path in DATASETS:
            problem = load_problem(path)
            rng = random.Random(5)
            sched = construct(problem, rng).to_schedule()
            state = SearchState(problem, sched)
            hard_incremental = sum(state.hard_of(i) for i in range(problem.n_events)) / 2.0
            self.assertAlmostEqual(hard_incremental, evaluate(problem, sched).hard_penalty, places=6)

    def test_pair_hard_sums_to_hard_cost(self):
        problem = load_problem("data/lecture/medium")
        rng = random.Random(1)
        state = construct(problem, rng)
        for i in range(problem.n_events):
            g, r = state.start[i], state.room[i]
            self.assertAlmostEqual(sum(state.pair_hard(i, g, r).values()), state.hard_of(i), places=6)

    def test_repair_never_makes_things_worse_and_is_complete(self):
        problem = load_problem("data/lecture/medium")
        rng = random.Random(3)
        sched = random_schedule(problem, rng)
        before = evaluate(problem, sched).hard_violations
        fixed = repair(problem, sched, rng, max_steps=3000)
        after = evaluate(problem, fixed)
        self.assertTrue(fixed.is_complete())
        self.assertLessEqual(after.hard_violations, before)


class DeterminismTests(unittest.TestCase):
    def test_neighbor_lists_are_sorted(self):
        """Neighbour order must not depend on Python's hash seed (reproducible runs)."""
        problem = load_problem("data/lecture/medium")
        for nb in problem.neighbors:
            self.assertEqual(nb, sorted(nb))


class AlgorithmTests(unittest.TestCase):
    def _check(self, problem, result):
        report = validate_schedule(problem, result.best_schedule)
        self.assertTrue(report.is_valid, report.text())
        self.assertEqual(result.best_hard_violations, 0)
        ev = evaluate(problem, result.best_schedule)       # reported values == real values
        self.assertAlmostEqual(ev.total, result.best_fitness, places=6)
        best = [row["best_fitness"] for row in result.history]
        self.assertTrue(all(b2 <= b1 + 1e-9 for b1, b2 in zip(best, best[1:])))

    def test_ga_feasible(self):
        for path in DATASETS:
            problem = load_problem(path)
            self._check(problem, genetic_algorithm(problem, GAConfig(population_size=20, generations=15)))

    def test_pso_feasible(self):
        for path in DATASETS:
            problem = load_problem(path)
            self._check(problem, particle_swarm_optimization(problem, PSOConfig(swarm_size=10, iterations=8)))

    def test_aco_feasible(self):
        for path in DATASETS:
            problem = load_problem(path)
            self._check(problem, ant_colony_optimization(problem, ACOConfig(ants=5, iterations=6)))

    def test_same_seed_same_result(self):
        problem = load_problem("data/lecture/small")
        a = genetic_algorithm(problem, GAConfig(population_size=10, generations=5, seed=7))
        b = genetic_algorithm(problem, GAConfig(population_size=10, generations=5, seed=7))
        self.assertEqual(a.best_schedule.to_vector(), b.best_schedule.to_vector())


if __name__ == "__main__":
    unittest.main()
