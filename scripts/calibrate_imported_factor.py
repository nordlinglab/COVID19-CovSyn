"""Calibrate the transmission factor of the imported first-wave seeds in the community (B78 C).

In the 2020-04-12 scenario the 118 imported seeds living in the community (69 found there and
49 recalled into home quarantine on 2020-03-18, B73) infected 12 local cases in the data
(validation_reference/taiwan_local_case_sources.csv), 0.102 each; the run 19 model gives about
0.32 each. The factor q multiplies their infection probabilities until they enter home
quarantine (the breach probability applies from then on, B74).

Since B82 the seeds that arrived before the border control are held at 1.0, so the people the
community seeds infect are m(q) = A + B q with A = m(0) > 0, and the proportional update
q = 12 / m(1) this script used to make overshot (0.256 against about 0.11 on run 21's vector;
code review 2026-11-09, F9). The script now measures m(0) and m(q0) at a low starting factor q0,
the current IMPORTED_FACTOR unless given, solves the line through them for 12 -- two low-q
measurements, as B82 found necessary, since m(q) is not quite linear far from the root -- and
repeats false-position steps inside the bracket of measurements until one is within 0.5 people
of 12.
A factor that cannot reach the data stops the script with a non-zero exit.

Usage (repository root):
    python scripts/calibrate_imported_factor.py <firefly_best.txt> [runs] [workers] [isolation] [q0]

isolation is the B78 option A setting (model or imported_delay, the default since the
meeting of 2026-11-02); calibrate with the one that will be used, since earlier isolation
leaves less for the factor to remove.
"""

import concurrent.futures
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from covsyn.model.data_synthesis_main import IMPORTED_FACTOR, run_covid  # noqa: E402

LAYERS = ("household", "school", "workplace", "health_care", "municipality")
SEED_START = 710000
SEEDS = "variable/taiwan_first_wave_seeds.csv"
SOURCES = "validation_reference/taiwan_local_case_sources.csv"


def observed_offspring():
    """Local cases infected by the imported seeds living in the community, and their number."""
    seeds = pd.read_csv(SEEDS)
    community = set(seeds.case_id[(seeds.seed_type == "imported") & (seeds.status == "community")])
    infector = pd.to_numeric(pd.read_csv(SOURCES).infector_id, errors="coerce")
    return int(infector.isin(community).sum()), len(community)


def community_offspring(seed, P, dp, factor, isolation):
    """People infected by the imported community seeds of one run."""
    _, _, course, contact = run_covid(seed, P, dp, mode="taiwan_first_outbreak",
                                      imported_factor=factor, imported_isolation=isolation)
    return sum(sum(float(np.nansum(np.asarray(x[f"{layer}_effective_contacts"] or [], dtype=float)))
                   for layer in LAYERS)
               for c, x in zip(course, contact)
               if c.get("seed_type") == "imported" and c.get("seed_status") == "community")


def mean_offspring(P, dp, factor, isolation, runs, workers):
    """Mean over runs of the people the community seeds infect, with its 5-95% range."""
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        out = np.array(list(ex.map(community_offspring, range(SEED_START, SEED_START + runs),
                                   [P] * runs, [dp] * runs, [factor] * runs, [isolation] * runs)))
    return out.mean(), np.percentile(out, [5, 95])


def solve_line(q_low, m_low, q_high, m_high, target):
    """The q at which the line through (q_low, m_low) and (q_high, m_high) reaches target.

    Args:
        q_low, m_low: A lower factor and the mean people infected at it.
        q_high, m_high: A higher factor and the mean people infected at it.
        target: The people infected in the data.

    Returns:
        The factor, not clipped: one outside [0, 1] means the data cannot be reached.

    Raises:
        ValueError: The two measurements do not rise with q, so no line can be solved.
    """
    if m_high <= m_low:
        raise ValueError(f"m({q_high:.4f}) = {m_high:.2f} is not above m({q_low:.4f}) = {m_low:.2f}")
    return float(q_low + (target - m_low) * (q_high - q_low) / (m_high - m_low))


def calibrate(measure, target, q0, tolerance=0.5, max_steps=8):
    """The factor whose measured mean reaches target within tolerance.

    Until a measurement lies above target, each step extends the line from m(0) through the
    highest measurement below it, capped at q = 1, where m(1) is measured before the target is
    called unreachable. Once target is bracketed, each step is Illinois false position: the
    line between the two bracket ends, with the weight of an end that stays put twice halved,
    so a curved m(q) cannot stall one end (code review 2026-11-09). Every measurement is also
    its own check; m(0) or q0 is returned when it is already within tolerance.

    The bracket assumes the measurement noise is smaller than the distance from target: a mean
    that noise carries across target fixes a wrong bracket end, and the calibration then ends
    with an error, not with a wrong factor. More runs per batch reduce the noise.

    Args:
        measure: Function of q returning the mean people infected over the runs.
        target: The people infected in the data.
        q0: Starting factor, in (0, 1].
        tolerance: Accepted distance of the measured mean from target, in people.
        max_steps: Steps before giving up, at least 1.

    Returns:
        The calibrated factor.

    Raises:
        ValueError: q0 or max_steps is out of range, the seeds held at 1.0 alone exceed
            target, m(1) stays below it, the measurements below target do not rise with q,
            or no step converges.
    """
    if not 0.0 < q0 <= 1.0:
        raise ValueError(f"starting factor {q0} is outside (0, 1]")
    if max_steps < 1:
        raise ValueError(f"max_steps {max_steps} must be at least 1")
    m0 = measure(0.0)
    if abs(m0 - target) <= tolerance:
        return 0.0
    if m0 > target:
        raise ValueError(f"the seeds held at 1.0 alone infect {m0:.2f} > {target}; "
                         "no q in [0, 1] reaches the data")
    m = measure(q0)
    if abs(m - target) <= tolerance:
        return q0
    # bracket ends as (q, m - target): low below target, high above it once seen
    low, high = ((q0, m - target), None) if m < target else ((0.0, m0 - target), (q0, m - target))
    kept = None   # which end the last step left in place, for the Illinois halving
    for _ in range(max_steps):
        if high is None:
            q = min(solve_line(0.0, m0 - target, low[0], low[1], 0.0), 1.0)
        else:
            q = solve_line(low[0], low[1], high[0], high[1], 0.0)
        m = measure(q)
        if abs(m - target) <= tolerance:
            return q
        if m < target:
            if high is None and q == 1.0:
                raise ValueError(f"m(1) = {m:.2f} < {target}; no factor reaches the data")
            if kept == "high" and high is not None:
                high = (high[0], high[1] / 2)
            low, kept = (q, m - target), "high"
        else:
            if kept == "low":
                low = (low[0], low[1] / 2)
            high, kept = (q, m - target), "low"
    raise ValueError(f"no factor within {tolerance} of {target} after {max_steps} steps; "
                     f"last q = {q:.4f} gave {m:.2f}; more runs per batch reduce the noise")


def main():
    rows = np.atleast_2d(np.loadtxt(sys.argv[1]))
    P = rows[np.argmin(rows[:, -1]), 1:-1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    isolation = sys.argv[4] if len(sys.argv) > 4 else "imported_delay"
    q0 = float(sys.argv[5]) if len(sys.argv) > 5 else IMPORTED_FACTOR
    with open("variable/demographic_parameters.pkl", "rb") as f:
        dp = pickle.load(f)
    k, n = observed_offspring()

    def measure(q):
        m, band = mean_offspring(P, dp, q, isolation, runs, workers)
        print(f"q = {q:.4f}: {n} community seeds infect {m:.2f} "
              f"(5-95% of runs {band[0]:.0f}-{band[1]:.0f}), data {k}; {runs} runs")
        return m

    print(f"isolation {isolation}; starting factor {q0:.4f}")
    try:
        q = calibrate(measure, k, q0)
    except ValueError as error:
        sys.exit(f"no calibration: {error}")
    print(f"calibrated q = {q:.4f} (its runs above are the check)")


if __name__ == "__main__":
    main()
