#!/usr/bin/env python3
"""Monte Carlo simulation of the 2026 FIFA World Cup from the quarterfinal stage.

The model is described in the accompanying LaTeX note (main.tex):

* Goals in a match between teams i and j are independent Poisson variables,

      G_i ~ Poisson(lambda_ij),   G_j ~ Poisson(lambda_ji),

  with

      log lambda_ij = alpha + attack_i - defense_j
                      + gamma * (elo_i  - elo_j)
                      + delta * (form_i - form_j)
                      + rho   * (rest_i - rest_j).

* If a knockout match is level after regular time, team i wins the
  extra-time/penalty tie-break with logistic probability

      P(i wins tie-break | draw) = 1 / (1 + exp(-eta * (elo_i - elo_j))).

The script reads team covariates from ``data_template.csv`` (estimated from
real data by ``fit_parameters.py``; see that script for sources), prints the
analytic quarterfinal advancement probabilities, simulates the remaining
seven matches ``N_SIM`` times, prints the estimated conditional champion
probabilities, and writes them to ``champion_probabilities.csv``.

Requires: numpy and pandas (plus the Python standard library only).

Usage:
    python3 simulate_worldcup.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration. Coefficients fitted by fit_parameters.py on the
# github.com/martj42/international_results dataset (matches through
# 2026-07-07): time-decayed Poisson IRLS for alpha/gamma/delta/rho,
# shoot-out logistic MLE for eta. Re-run fit_parameters.py to refresh.
# ---------------------------------------------------------------------------

DATA_FILE = "data_template.csv"
OUTPUT_FILE = "champion_probabilities.csv"

COEFFS = {
    "alpha":  0.150326,  # baseline log scoring rate
    "gamma":  0.000972,  # effect per Elo-rating point
    "delta":  0.005260,  # effect per unit of form (last-5 goal difference)
    "rho":   -0.000886,  # effect per rest day (fitted: essentially nil)
    "eta":    0.001382,  # tie-break logistic slope, per Elo-rating point
}

MAX_GOALS = 10       # truncation of the score grid for exact probabilities
N_SIM = 100_000      # Monte Carlo replications
SEED = 42            # seed for numpy.random.default_rng (reproducibility)

# Quarterfinal bracket. Semifinal 1 pairs the winners of QF1 and QF2,
# semifinal 2 pairs the winners of QF3 and QF4, and the final pairs the
# two semifinal winners.
QUARTERFINALS = [
    ("France", "Morocco"),         # QF1
    ("Spain", "Belgium"),          # QF2
    ("Norway", "England"),         # QF3
    ("Argentina", "Switzerland"),  # QF4
]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

def scoring_rate(row_i: pd.Series, row_j: pd.Series, coeffs: dict) -> float:
    """Expected goals lambda_ij scored by team i against team j."""
    log_rate = (
        coeffs["alpha"]
        + row_i["attack"]
        - row_j["defense"]
        + coeffs["gamma"] * (row_i["elo"] - row_j["elo"])
        + coeffs["delta"] * (row_i["form"] - row_j["form"])
        + coeffs["rho"] * (row_i["rest_days"] - row_j["rest_days"])
    )
    return float(np.exp(log_rate))


def tie_break_probability(elo_i: float, elo_j: float, eta: float) -> float:
    """P(team i wins extra time + penalties | draw), logistic in Elo diff."""
    return float(1.0 / (1.0 + np.exp(-eta * (elo_i - elo_j))))


def truncated_poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
    """PMF of Poisson(lam) restricted to {0, ..., max_goals}, renormalized.

    The renormalization compensates for the (tiny) tail mass beyond the
    cutoff so that the score-grid probabilities sum exactly to one.
    """
    pmf = np.empty(max_goals + 1)
    pmf[0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf / pmf.sum()


def load_params(data_file: str = DATA_FILE,
                coeffs: dict | None = None,
                max_goals: int = MAX_GOALS,
                bracket: list[tuple[str, str]] | None = None) -> dict:
    """Read the team table and precompute all pairwise model quantities.

    Returns the ``params`` dictionary consumed by every function below:
      teams     -- DataFrame of covariates indexed by team name
      coeffs    -- model coefficients (alpha, gamma, delta, rho, eta)
      max_goals -- score-grid truncation point
      bracket   -- list of the four quarterfinal pairings
      rates     -- {(i, j): lambda_ij} for every ordered team pair
      tie_break -- {(i, j): P(i wins tie-break | draw)} for every ordered pair
    """
    coeffs = dict(COEFFS if coeffs is None else coeffs)
    bracket = list(QUARTERFINALS if bracket is None else bracket)

    teams = pd.read_csv(data_file, comment="#")
    teams["team"] = teams["team"].str.strip()
    teams = teams.set_index("team")
    if teams.index.duplicated().any():
        raise ValueError(f"Duplicate team names in {data_file}.")

    names = [t for pair in bracket for t in pair]
    missing = set(names) - set(teams.index)
    if missing:
        raise ValueError(
            f"Teams in the bracket but missing from {data_file}: {sorted(missing)}"
        )
    # The prediction is conditional on the realized quarterfinal line-up:
    # only the eight quarterfinalists are kept, in bracket order.
    teams = teams.loc[names]

    rates: dict[tuple[str, str], float] = {}
    tie_break: dict[tuple[str, str], float] = {}
    for i in teams.index:
        for j in teams.index:
            if i == j:
                continue
            rates[(i, j)] = scoring_rate(teams.loc[i], teams.loc[j], coeffs)
            tie_break[(i, j)] = tie_break_probability(
                teams.loc[i, "elo"], teams.loc[j, "elo"], coeffs["eta"]
            )

    return {
        "teams": teams,
        "coeffs": coeffs,
        "max_goals": max_goals,
        "bracket": bracket,
        "rates": rates,
        "tie_break": tie_break,
    }


# ---------------------------------------------------------------------------
# Match-level probabilities and simulation
# ---------------------------------------------------------------------------

def match_win_probability(team_i: str, team_j: str, params: dict) -> float:
    """Exact advancement probability p_ij for a knockout match.

    p_ij = P(i wins in regular time) + P(draw) * P(i wins tie-break | draw),
    computed on the truncated (and renormalized) score grid
    {0, ..., max_goals}^2.
    """
    lam_ij = params["rates"][(team_i, team_j)]
    lam_ji = params["rates"][(team_j, team_i)]
    pmf_i = truncated_poisson_pmf(lam_ij, params["max_goals"])
    pmf_j = truncated_poisson_pmf(lam_ji, params["max_goals"])
    joint = np.outer(pmf_i, pmf_j)        # joint[x, y] = P(G_i = x, G_j = y)
    p_win_regular = float(np.tril(joint, k=-1).sum())   # entries with x > y
    p_draw = float(np.trace(joint))                     # entries with x == y
    return p_win_regular + p_draw * params["tie_break"][(team_i, team_j)]


def simulate_match(team_i: str, team_j: str, params: dict,
                   rng: np.random.Generator) -> str:
    """Simulate one knockout match and return the advancing team.

    Goals are drawn from the (untruncated) Poisson distributions; a draw
    after regular time is resolved by the logistic Elo tie-break. The
    truncation used in ``match_win_probability`` is irrelevant here and
    the discrepancy is negligible for realistic scoring rates.
    """
    g_i = rng.poisson(params["rates"][(team_i, team_j)])
    g_j = rng.poisson(params["rates"][(team_j, team_i)])
    if g_i != g_j:
        return team_i if g_i > g_j else team_j
    # Level after regular time: extra time + penalties, collapsed into a
    # single logistic tie-break based on the Elo difference.
    if rng.random() < params["tie_break"][(team_i, team_j)]:
        return team_i
    return team_j


def simulate_tournament(params: dict, n_sim: int = 100_000,
                        seed: int = 42) -> pd.Series:
    """Simulate the remaining seven matches n_sim times.

    Returns the estimated conditional champion probabilities as a Series
    indexed by team, sorted in decreasing order.
    """
    rng = np.random.default_rng(seed)
    counts = {t: 0 for t in params["teams"].index}
    for _ in range(n_sim):
        # Quarterfinals
        w = [simulate_match(a, b, params, rng) for a, b in params["bracket"]]
        # Semifinals (bracket structure: (QF1 vs QF2), (QF3 vs QF4))
        s1 = simulate_match(w[0], w[1], params, rng)
        s2 = simulate_match(w[2], w[3], params, rng)
        # Final
        champion = simulate_match(s1, s2, params, rng)
        counts[champion] += 1
    probs = pd.Series(counts, name="champion_probability") / n_sim
    return probs.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params = load_params()

    print("Model coefficients (fitted by fit_parameters.py):",
          params["coeffs"])
    print(f"\nAnalytic advancement probabilities "
          f"(score grid truncated at {params['max_goals']} goals):")
    for a, b in params["bracket"]:
        p = match_win_probability(a, b, params)
        print(f"  {a:<12} vs {b:<12}  P({a} advances) = {p:.4f}")

    probs = simulate_tournament(params, n_sim=N_SIM, seed=SEED)

    worst_se = (0.25 / N_SIM) ** 0.5
    print(f"\nConditional champion probabilities from {N_SIM:,} simulated "
          f"tournaments (seed = {SEED}):")
    print(f"(worst-case Monte Carlo standard error ~ {worst_se:.4f})\n")
    print(f"  {'team':<12} {'P(champion | quarterfinalist)':>30}")
    for team, p in probs.items():
        print(f"  {team:<12} {p:>30.4f}")
    print(f"\n  {'TOTAL':<12} {probs.sum():>30.4f}")

    out = probs.rename_axis("team").reset_index()
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
