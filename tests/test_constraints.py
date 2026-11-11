"""Conflict-detection tests: every constraint is exercised on a hand-made case
whose correct answer is known in advance. Each case is checked with BOTH the
fitness function and the independent validator (they must agree)."""
import unittest

from tests.helpers import CLEAN, both, make_problem


def changed(**overrides):
    """CLEAN schedule with some events moved: changed(L_C2_1=(5, 'R1'))."""
    m = dict(CLEAN)
    m.update(overrides)
    return m


class HardConstraintTests(unittest.TestCase):
    def setUp(self):
        self.p = make_problem()

    def test_clean_schedule_has_no_hard_violation(self):
        fit, val = both(self.p, CLEAN)
        self.assertTrue(fit.feasible)
        self.assertTrue(val.is_valid)
        self.assertEqual(fit.hard_violations, 0)

    def test_teacher_conflict(self):
        # T2 teaches C2 and C3: put both at slot 6 (different rooms).
        fit, _ = both(self.p, changed(L_C2_1=(6, "R2")))
        self.assertEqual(fit.counts["teacher_conflicts"], 1)
        self.assertEqual(fit.counts["room_conflicts"], 0)

    def test_room_conflict(self):
        # C1 session 2 (Mon 08-10) and C2 (Mon 09) both in R1 -> also 5 shared students.
        fit, _ = both(self.p, changed(L_C2_1=(5, "R1")))
        self.assertEqual(fit.counts["room_conflicts"], 1)
        self.assertEqual(fit.counts["student_conflicts"], 5)

    def test_student_conflict_counts_each_affected_student(self):
        # C1 session 1 (slots 0-1, R1) and C2 (slot 1, R2): 5 common students.
        fit, _ = both(self.p, changed(L_C2_1=(1, "R2")))
        self.assertEqual(fit.counts["student_conflicts"], 5)
        self.assertEqual(fit.counts["room_conflicts"], 0)
        self.assertEqual(fit.counts["teacher_conflicts"], 0)

    def test_capacity_violation(self):
        # C1 has 25 students but R2 has 20 seats.
        fit, _ = both(self.p, changed(L_C1_1=(0, "R2"), L_C2_1=(2, "R1")))
        self.assertEqual(fit.counts["capacity_violations"], 1)

    def test_room_type_violation_both_directions(self):
        # lab event in a standard room, and a standard event in the lab.
        fit, _ = both(self.p, changed(L_C3_1=(6, "R1")))
        self.assertEqual(fit.counts["room_type_violations"], 1)
        fit, _ = both(self.p, changed(L_C2_1=(2, "LAB")))
        self.assertEqual(fit.counts["room_type_violations"], 1)

    def test_teacher_unavailability(self):
        # T1 is unavailable at slot 3; a 2-slot lecture starting at 2 covers 2 and 3.
        fit, _ = both(self.p, changed(L_C1_1=(2, "R1"), L_C2_1=(0, "R2")))
        self.assertEqual(fit.counts["availability_violations"], 1)

    def test_event_running_past_end_of_day(self):
        # slot 3 is the last slot of Sunday; a 2-slot lecture cannot start there.
        fit, _ = both(self.p, changed(L_C3_1=(3, "LAB")))
        self.assertEqual(fit.counts["bounds_violations"], 1)

    def test_unassigned_and_invalid_values(self):
        mapping = dict(CLEAN)
        del mapping["L_C2_1"]                                   # unassigned
        fit, _ = both(self.p, mapping)
        self.assertEqual(fit.counts["bounds_violations"], 1)
        mapping = changed(L_C2_1=(99, "R2"))                    # slot outside the grid
        fit, _ = both(self.p, mapping)
        self.assertEqual(fit.counts["bounds_violations"], 1)

    def test_back_to_back_events_do_not_conflict(self):
        # C1 session 1 ends at slot 2 (exclusive), C2 starts at slot 2 in the SAME room.
        fit, _ = both(self.p, changed(L_C2_1=(2, "R1")))
        self.assertEqual(fit.counts["room_conflicts"], 0)
        self.assertEqual(fit.counts["student_conflicts"], 0)

    def test_partial_overlap_of_multi_slot_events_is_detected(self):
        # C1 session 1 covers slots 0-1; C2 at slot 1 in the same room overlaps by one slot.
        fit, _ = both(self.p, changed(L_C2_1=(1, "R1")))
        self.assertEqual(fit.counts["room_conflicts"], 1)

    def test_unplaced_events_never_create_conflicts(self):
        fit, _ = both(self.p, {})                               # nothing placed at all
        self.assertEqual(fit.counts["bounds_violations"], 4)
        self.assertEqual(fit.hard_violations, 4)


class SoftConstraintTests(unittest.TestCase):
    def test_preference_violations(self):
        fit, _ = both(make_problem(), CLEAN)
        # T2 prefers {0,1,4,5}: C2 at slot 2 and C3 at slots 6-7 are both outside it.
        self.assertEqual(fit.counts["preference_violations"], 2)

    def test_preference_respected(self):
        # C2 at slot 0 (preferred) and C3 at slots 4-5 (preferred; students differ from C1 s2? no)
        fit, _ = both(make_problem(), changed(L_C2_1=(0, "R2"), L_C3_1=(4, "LAB")))
        self.assertEqual(fit.counts["preference_violations"], 0)

    def test_spread_two_sessions_same_day(self):
        # both sessions of C1 on Sunday (slots 0-1 and 2-3, R1) - T1 is unavailable at 3,
        # so this is a hard violation too; we only look at the soft counter here.
        fit, _ = both(make_problem(), changed(L_C1_2=(2, "R1"), L_C2_1=(4, "R2")))
        self.assertEqual(fit.counts["spread_violations"], 1)
        self.assertEqual(fit.counts["availability_violations"], 1)

    def test_spread_clean_schedule_is_zero(self):
        fit, _ = both(make_problem(), CLEAN)
        self.assertEqual(fit.counts["spread_violations"], 0)

    def test_room_waste_value(self):
        fit, _ = both(make_problem(), CLEAN)
        expected = 2 * (30 - 25) / 30 + (20 - 5) / 20 + (30 - 3) / 30   # C1 x2, C2, C3
        self.assertAlmostEqual(fit.counts["room_waste"], expected)

    def test_gap_disabled_for_lectures(self):
        fit, _ = both(make_problem(min_gap_slots=0), changed(L_C2_1=(2, "R2")))
        self.assertEqual(fit.counts["gap_violations"], 0)

    def test_gap_back_to_back_is_penalised_when_enabled(self):
        # CLEAN has two back-to-back pairs with a gap of 0 (< min gap 1):
        #   C1 session 1 (ends slot 2) -> C2 (starts slot 2): 5 common students
        #   C1 session 2 (ends slot 6) -> C3 (starts slot 6): 3 common students
        p = make_problem(min_gap_slots=1)
        fit, _ = both(p, CLEAN)
        self.assertEqual(fit.counts["gap_violations"], 5 + 3)
        # moving C2 to slot 3 leaves a one-slot rest, which satisfies the minimum gap
        fit, _ = both(p, changed(L_C2_1=(3, "R2")))
        self.assertEqual(fit.counts["gap_violations"], 3)

    def test_gap_only_within_the_same_day(self):
        # C2 on Monday slot 4 right after nothing on Sunday: no gap violation.
        p = make_problem(min_gap_slots=1)
        fit, _ = both(p, changed(L_C1_2=(6, "R1"), L_C2_1=(4, "R2"), L_C3_1=(0, "LAB")))
        self.assertEqual(fit.counts["gap_violations"], 0)


class WeightedTotalTests(unittest.TestCase):
    def test_total_is_weighted_sum_and_hard_dominates(self):
        p = make_problem()
        clean, _ = both(p, CLEAN)
        broken, _ = both(p, changed(L_C2_1=(6, "R2")))           # one teacher conflict
        self.assertAlmostEqual(broken.hard_penalty, 1000.0 * broken.hard_violations)
        self.assertAlmostEqual(clean.total, clean.hard_penalty + clean.soft_penalty)
        self.assertEqual(clean.hard_penalty, 0)
        self.assertGreater(broken.total, clean.total)

    def test_custom_weights(self):
        from core.constraints import FitnessConfig
        from core.fitness import evaluate
        from tests.helpers import sched
        p = make_problem()
        cfg = FitnessConfig().with_overrides(preference_violations=10.0)
        res = evaluate(p, sched(p, CLEAN), cfg)
        base = evaluate(p, sched(p, CLEAN))
        self.assertAlmostEqual(res.soft_penalty - base.soft_penalty, 8 * 2)   # 2 violations x (10-2)

    def test_unknown_weight_name_is_rejected(self):
        from core.constraints import FitnessConfig
        with self.assertRaises(ValueError):
            FitnessConfig({"not_a_constraint": 1.0})


if __name__ == "__main__":
    unittest.main()
