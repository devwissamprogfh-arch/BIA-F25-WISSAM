"""Print a quality report for every data set (used for the project report).

For each data set it shows: size and tightness, whether the planted solution is
valid, the cost of one fitness evaluation, and the reference results of the two
baselines (random placement and the greedy heuristic).

Usage (from the project root):  python -m data.describe_data
"""
from __future__ import annotations

import os
import random
import statistics
import time

from core.baseline import greedy_schedule, random_schedule
from core.fitness import evaluate
from core.problem import load_problem
from core.validator import diagnose_instance, validate_schedule
from models.schedule import load_schedule_csv

DATA = os.path.dirname(os.path.abspath(__file__))


def main(n_random: int = 30, n_greedy: int = 10) -> None:
    for mode in ("lecture", "exam"):
        for size in ("small", "medium", "large"):
            d = os.path.join(DATA, mode, size)
            p = load_problem(d)
            planted = load_schedule_csv(p, os.path.join(d, "planted_schedule.csv"))
            info, diag = p.summary(), diagnose_instance(p)
            res_p, val_p = evaluate(p, planted), validate_schedule(p, planted)

            t0 = time.perf_counter()
            for _ in range(200):
                evaluate(p, planted)
            eval_ms = (time.perf_counter() - t0) / 200 * 1000

            rnd = [evaluate(p, random_schedule(p, random.Random(s))) for s in range(n_random)]
            grd, grd_t = [], []
            for s in range(n_greedy):
                t0 = time.perf_counter()
                sched = greedy_schedule(p, random.Random(s))
                grd_t.append(time.perf_counter() - t0)
                grd.append(evaluate(p, sched))

            print(f"=== {mode}/{size}")
            print(f"  events={info['events']} courses={info['courses']} teachers={info['teachers']} "
                  f"rooms={info['rooms']} students={info['students']} enrollments={info['enrollments']}")
            print(f"  slots={info['slots']} slot-units={info['slot_units_to_place']} "
                  f"utilisation={info['room_utilisation_pct']}% avg-event-size={info['avg_event_size']} "
                  f"pairs-sharing-students={info['event_pairs_sharing_students']}")
            print(f"  diagnose: errors={len(diag.errors)} warnings={len(diag.warnings)}")
            print(f"  planted: valid={val_p.is_valid} hard={res_p.hard_violations} "
                  f"soft={res_p.soft_penalty:.1f}  eval_time={eval_ms:.2f} ms")
            print(f"  random (n={n_random}): hard mean={statistics.mean(r.hard_violations for r in rnd):.1f} "
                  f"soft mean={statistics.mean(r.soft_penalty for r in rnd):.1f}")
            feas = sum(r.feasible for r in grd)
            print(f"  greedy (n={n_greedy}): hard mean={statistics.mean(r.hard_violations for r in grd):.1f} "
                  f"min={min(r.hard_violations for r in grd)} feasible runs={feas}/{n_greedy} "
                  f"soft mean={statistics.mean(r.soft_penalty for r in grd):.1f} "
                  f"time mean={statistics.mean(grd_t) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
