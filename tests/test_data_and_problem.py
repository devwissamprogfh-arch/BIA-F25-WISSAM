"""Tests of the data layer: generator, loader, instance diagnostics, schedule codec,
and the agreement between the fitness function and the independent validator."""
import os
import random
import tempfile
import unittest

from tests.helpers import ROOT, make_problem

from core.baseline import greedy_schedule, random_schedule
from core.constraints import HARD_NAMES
from core.fitness import evaluate, fitness
from core.problem import DataError, Problem, load_problem, save_problem
from core.validator import diagnose_instance, validate_schedule
from data.generate_data import SIZES, generate_instance
from models.entities import Course, Room, Student, Teacher, TimeGrid
from models.schedule import Schedule, load_schedule_csv, save_schedule_csv

DATA = os.path.join(ROOT, "data")
MODES = ("lecture", "exam")


class GeneratedDataTests(unittest.TestCase):
    def test_event_counts_match_the_plan(self):
        for mode in MODES:
            for size, cfg in SIZES.items():
                p = load_problem(os.path.join(DATA, mode, size))
                self.assertEqual(p.n_events, cfg["n_events"], (mode, size))
                self.assertEqual(len(p.students), cfg["n_students"])

    def test_planted_solution_is_valid_everywhere(self):
        for mode in MODES:
            for size in SIZES:
                d = os.path.join(DATA, mode, size)
                p = load_problem(d)
                planted = load_schedule_csv(p, os.path.join(d, "planted_schedule.csv"))
                self.assertTrue(planted.is_complete())
                res, val = evaluate(p, planted), validate_schedule(p, planted)
                self.assertTrue(val.is_valid, val.text())
                self.assertEqual(res.hard_violations, 0, (mode, size))

    def test_no_instance_is_obviously_infeasible(self):
        for mode in MODES:
            for size in SIZES:
                rep = diagnose_instance(load_problem(os.path.join(DATA, mode, size)))
                self.assertTrue(rep.possibly_feasible, rep.text())

    def test_generation_is_deterministic(self):
        a, pa = generate_instance("lecture", "small")
        b, pb = generate_instance("lecture", "small")
        self.assertEqual(a.enrollments, b.enrollments)
        self.assertEqual(pa, pb)

    def test_save_and_load_round_trip(self):
        p, planted = generate_instance("exam", "small")
        with tempfile.TemporaryDirectory() as tmp:
            save_problem(p, tmp)
            q = load_problem(tmp)
            self.assertEqual(q.event_ids, p.event_ids)
            self.assertEqual(sorted(q.enrollments), sorted(p.enrollments))
            self.assertEqual(q.min_gap_slots, p.min_gap_slots)
            path = os.path.join(tmp, "s.csv")
            save_schedule_csv(p, planted, path)
            self.assertEqual(load_schedule_csv(q, path), planted)


class FitnessVsValidatorTests(unittest.TestCase):
    """The two independent implementations must report identical hard counts."""

    def _agree(self, problem, schedule):
        fit, val = evaluate(problem, schedule), validate_schedule(problem, schedule)
        for name in HARD_NAMES:
            self.assertEqual(fit.counts[name], val.counts[name], name)
        self.assertEqual(fit.feasible, val.is_valid)

    def test_agreement_on_random_and_perturbed_schedules(self):
        rng = random.Random(2026)
        for mode in MODES:
            for size in ("small", "medium"):
                d = os.path.join(DATA, mode, size)
                p = load_problem(d)
                planted = load_schedule_csv(p, os.path.join(d, "planted_schedule.csv"))
                for _ in range(15):
                    self._agree(p, random_schedule(p, rng))
                for _ in range(15):                       # planted schedule with a few moved events
                    s = planted.copy()
                    for i in rng.sample(range(p.n_events), 3):
                        s.assign(i, rng.randrange(p.n_slots), rng.randrange(p.n_rooms))
                    if rng.random() < 0.3:
                        s.slots[rng.randrange(p.n_events)] = -1      # also test unassigned
                    self._agree(p, s)

    def test_fitness_function_matches_evaluate(self):
        p = load_problem(os.path.join(DATA, "lecture", "small"))
        s = random_schedule(p, random.Random(1))
        self.assertAlmostEqual(fitness(p, s), evaluate(p, s).total)

    def test_schedule_length_mismatch_is_rejected(self):
        p = make_problem()
        with self.assertRaises(ValueError):
            evaluate(p, Schedule.empty(p.n_events + 1))
        with self.assertRaises(ValueError):
            validate_schedule(p, Schedule.empty(p.n_events + 1))


class BaselineTests(unittest.TestCase):
    def test_greedy_beats_random_and_solves_small(self):
        for mode in MODES:
            p = load_problem(os.path.join(DATA, mode, "small"))
            rnd = evaluate(p, random_schedule(p, random.Random(3)))
            grd = evaluate(p, greedy_schedule(p))
            self.assertLess(grd.total, rnd.total)
            self.assertTrue(grd.feasible, f"greedy should solve {mode}/small")


class ScheduleCodecTests(unittest.TestCase):
    def test_vector_round_trip(self):
        s = Schedule([5, -1, 9], [2, -1, 0])
        self.assertEqual(Schedule.from_vector(s.to_vector()), s)
        self.assertEqual(s.to_vector(), [5, 2, -1, -1, 9, 0])

    def test_copy_is_independent(self):
        s = Schedule([1, 2], [0, 1])
        c = s.copy()
        c.assign(0, 7, 3)
        self.assertEqual(s.slots[0], 1)

    def test_odd_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            Schedule.from_vector([1, 2, 3])


class DataErrorTests(unittest.TestCase):
    def _parts(self):
        grid = TimeGrid(("Sun",), 4)
        rooms = [Room("R1", "R1", 10, "standard")]
        teachers = [Teacher("T1", "T1")]
        courses = [Course("C1", "C1", "T1")]
        return grid, rooms, teachers, courses, [Student("S1", "S1")]

    def test_unknown_references_are_rejected(self):
        grid, rooms, teachers, courses, students = self._parts()
        with self.assertRaises(DataError):
            Problem("x", "lecture", grid, rooms, teachers, courses, students, [("S9", "C1")])
        with self.assertRaises(DataError):
            Problem("x", "lecture", grid, rooms, teachers, courses, students, [("S1", "C9")])
        with self.assertRaises(DataError):
            Problem("x", "lecture", grid, rooms, teachers,
                    [Course("C1", "C1", "T9")], students, [])

    def test_invalid_mode_and_teacher_slot(self):
        grid, rooms, teachers, courses, students = self._parts()
        with self.assertRaises(DataError):
            Problem("x", "weekend", grid, rooms, teachers, courses, students, [])
        with self.assertRaises(DataError):
            Problem("x", "lecture", grid, rooms, [Teacher("T1", "T1", frozenset({99}))],
                    courses, students, [])


class InstanceDiagnosticsTests(unittest.TestCase):
    def test_detects_room_too_small(self):
        grid = TimeGrid(("Sun", "Mon"), 4)
        students = [Student(f"S{k}", f"S{k}") for k in range(30)]
        p = Problem("x", "lecture", grid, [Room("R1", "R1", 10)], [Teacher("T1", "T1")],
                    [Course("C1", "C1", "T1")], students, [(s.student_id, "C1") for s in students])
        rep = diagnose_instance(p)
        self.assertFalse(rep.possibly_feasible)
        self.assertTrue(any("no 'standard' room" in m for m in rep.errors))

    def test_detects_not_enough_room_slots(self):
        grid = TimeGrid(("Sun",), 2)                      # only 2 slots in total
        courses = [Course(f"C{k}", f"C{k}", f"T{k}", 1, 2) for k in range(3)]   # 3 x 2 slots
        teachers = [Teacher(f"T{k}", f"T{k}") for k in range(3)]
        p = Problem("x", "lecture", grid, [Room("R1", "R1", 10)], teachers, courses, [], [])
        rep = diagnose_instance(p)
        self.assertFalse(rep.possibly_feasible)
        self.assertTrue(any("room-slots" in m for m in rep.errors))

    def test_detects_teacher_overload_and_duration_too_long(self):
        grid = TimeGrid(("Sun",), 4)
        p = Problem("x", "lecture", grid, [Room("R1", "R1", 10), Room("R2", "R2", 10)],
                    [Teacher("T1", "T1", frozenset({0, 1}))],
                    [Course("C1", "C1", "T1", 2, 2), Course("C2", "C2", "T1", 1, 5)], [], [])
        rep = diagnose_instance(p)
        self.assertTrue(any("does not fit in a day" in m for m in rep.errors))
        self.assertTrue(any("teacher T1" in m for m in rep.errors))

    def test_tight_instance_gives_warning_not_error(self):
        grid = TimeGrid(("Sun",), 10)
        p = Problem("x", "lecture", grid, [Room("R1", "R1", 10)], [Teacher("T1", "T1")],
                    [Course("C1", "C1", "T1", 9, 1)], [], [])
        rep = diagnose_instance(p)
        self.assertTrue(rep.possibly_feasible)
        self.assertTrue(rep.warnings)


if __name__ == "__main__":
    unittest.main()
