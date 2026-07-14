#!/usr/bin/env python3
"""Fit the Poisson score model of main.tex to real historical data.

Data source (download before running):
    https://github.com/martj42/international_results
    -> results.csv   (all international matches since 1872)
    -> shootouts.csv (penalty shoot-out winners)

    curl -sL -o results.csv   https://raw.githubusercontent.com/martj42/international_results/master/results.csv
    curl -sL -o shootouts.csv https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv

Pipeline
--------
1.  Replicate the eloratings.net Elo algorithm over the full match history
    (K by tournament importance, goal-difference multiplier, +100 home
    advantage for non-neutral venues, shoot-outs counted as draws) to obtain
    every team's pre-match Elo rating at every date, including today.
2.  Build per-team rolling covariates: form (mean goal difference over the
    last five internationals) and rest (days since the previous match,
    capped at REST_CAP).
3.  Fit the log-linear Poisson model of Eq. (2) in main.tex,
        log lambda = alpha + a_i - d_j + gamma*(E_i-E_j)
                     + delta*(F_i-F_j) + rho*(R_i-R_j) + h*home_i,
    by iteratively reweighted least squares (IRLS) on all matches since
    FIT_START, two observations per match, with exponential time-decay
    weights (half-life HALF_LIFE_DAYS) and a small ridge penalty on the
    team attack/defence dummies. The home-advantage term h is needed for
    unbiased estimation but is not used in World Cup knockout predictions
    (neutral venues).
4.  Fit the tie-break slope eta by maximum likelihood on historical penalty
    shoot-outs: P(i wins shoot-out) = 1 / (1 + exp(-eta*(E_i - E_j))).
5.  Write the eight quarterfinalists' covariate rows (Elo, attack, defence,
    form, rest days at their quarterfinal date) to TEAM_OUT and print the
    fitted coefficients to paste into simulate_worldcup.py.

Requires numpy and pandas only. Runtime: ~10-20 s.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIT_START = "2016-01-01"     # regression window start
HALF_LIFE_DAYS = 1095.0      # time-decay half-life (3 years)
REST_CAP = 14                # rest-day effect saturates beyond two weeks
RIDGE_TEAM = 1.0             # ridge penalty on attack/defence dummies
ELO_SCALE = 100.0            # elo_diff is entered in units of 100 points
TEAM_OUT = "data_template.csv"

# Quarterfinal schedule (2026 FIFA World Cup): team -> QF date
QF_DATE = {
    "France": "2026-07-09", "Morocco": "2026-07-09",
    "Spain": "2026-07-10", "Belgium": "2026-07-10",
    "Norway": "2026-07-11", "England": "2026-07-11",
    "Argentina": "2026-07-11", "Switzerland": "2026-07-11",
}
QF_TEAMS = ["France", "Morocco", "Spain", "Belgium",
            "Norway", "England", "Argentina", "Switzerland"]

MAJOR_TOURNAMENTS = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup",
    "OFC Nations Cup", "Confederations Cup",
    "CONMEBOL–UEFA Cup of Champions",
}


def k_factor(tournament: str) -> float:
    """eloratings.net K factor by tournament importance."""
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament in MAJOR_TOURNAMENTS:
        return 50.0
    if "qualification" in tournament or "Nations League" in tournament:
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def goal_multiplier(diff: int) -> float:
    """eloratings.net goal-difference multiplier."""
    diff = abs(diff)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11.0 + diff) / 8.0


# ---------------------------------------------------------------------------
# Pass 1: Elo ratings and rolling covariates over the full history
# ---------------------------------------------------------------------------

def compute_history(results: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match Elo, form, and rest columns for both sides."""
    elo: dict[str, float] = defaultdict(lambda: 1500.0)
    last5: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    last_date: dict[str, pd.Timestamp] = {}

    cols = {c: np.empty(len(results)) for c in
            ("elo_h", "elo_a", "form_h", "form_a", "rest_h", "rest_a")}

    def form(team):
        gd = last5[team]
        return float(np.mean(gd)) if gd else 0.0

    def rest(team, date):
        if team not in last_date:
            return float(REST_CAP)
        return float(min((date - last_date[team]).days, REST_CAP))

    for n, row in enumerate(results.itertuples(index=False)):
        h, a, date = row.home_team, row.away_team, row.date
        cols["elo_h"][n], cols["elo_a"][n] = elo[h], elo[a]
        cols["form_h"][n], cols["form_a"][n] = form(h), form(a)
        cols["rest_h"][n], cols["rest_a"][n] = rest(h, date), rest(a, date)

        gh, ga = row.home_score, row.away_score
        if np.isnan(gh) or np.isnan(ga):        # unplayed fixture
            continue
        # Elo update (shoot-outs count as draws: scores are level)
        dr = elo[h] - elo[a] + (0.0 if row.neutral else 100.0)
        we_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w_home = 1.0 if gh > ga else (0.0 if gh < ga else 0.5)
        change = k_factor(row.tournament) * goal_multiplier(int(gh - ga)) \
            * (w_home - we_home)
        elo[h] += change
        elo[a] -= change
        # Rolling covariates
        last5[h].append(float(gh - ga))
        last5[a].append(float(ga - gh))
        last_date[h] = date
        last_date[a] = date

    for c, v in cols.items():
        results[c] = v
    # Final state, exposed for the prediction snapshot
    results.attrs["elo"] = dict(elo)
    results.attrs["last5"] = {t: list(q) for t, q in last5.items()}
    results.attrs["last_date"] = dict(last_date)
    return results


# ---------------------------------------------------------------------------
# Pass 2: Poisson regression (IRLS)
# ---------------------------------------------------------------------------

def fit_poisson(results: pd.DataFrame):
    """Fit alpha, {a_i}, {d_i}, gamma, delta, rho, h by weighted IRLS."""
    played = results.dropna(subset=["home_score", "away_score"])
    window = played[played["date"] >= pd.Timestamp(FIT_START)]

    teams = sorted(set(window["home_team"]) | set(window["away_team"]))
    t_idx = {t: k for k, t in enumerate(teams)}
    T, M = len(teams), len(window)

    ref_date = played["date"].max()
    decay = 0.5 ** (((ref_date - window["date"]).dt.days
                     .to_numpy(float)) / HALF_LIFE_DAYS)

    # Two observations per match: home side scoring, away side scoring
    y = np.concatenate([window["home_score"].to_numpy(float),
                        window["away_score"].to_numpy(float)])
    w = np.concatenate([decay, decay])
    scorer = np.concatenate([window["home_team"].map(t_idx).to_numpy(),
                             window["away_team"].map(t_idx).to_numpy()])
    opponent = np.concatenate([window["away_team"].map(t_idx).to_numpy(),
                               window["home_team"].map(t_idx).to_numpy()])
    elo_diff = np.concatenate([
        (window["elo_h"] - window["elo_a"]).to_numpy(float),
        (window["elo_a"] - window["elo_h"]).to_numpy(float)]) / ELO_SCALE
    form_diff = np.concatenate([
        (window["form_h"] - window["form_a"]).to_numpy(float),
        (window["form_a"] - window["form_h"]).to_numpy(float)])
    rest_diff = np.concatenate([
        (window["rest_h"] - window["rest_a"]).to_numpy(float),
        (window["rest_a"] - window["rest_h"]).to_numpy(float)])
    not_neutral = (~window["neutral"].to_numpy(bool)).astype(float)
    home = np.concatenate([not_neutral, np.zeros(M)])

    # Design matrix: [1 | attack dummies | defence dummies | covariates]
    n, p = len(y), 1 + 2 * T + 4
    X = np.zeros((n, p))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + scorer] = 1.0            # + a_scorer
    X[np.arange(n), 1 + T + opponent] = -1.0     # - d_opponent
    X[:, 1 + 2 * T + 0] = elo_diff
    X[:, 1 + 2 * T + 1] = form_diff
    X[:, 1 + 2 * T + 2] = rest_diff
    X[:, 1 + 2 * T + 3] = home

    ridge = np.zeros(p)
    ridge[1:1 + 2 * T] = RIDGE_TEAM              # penalize team dummies only

    beta = np.zeros(p)
    beta[0] = np.log(max(np.average(y, weights=w), 0.1))
    for it in range(50):
        eta_lin = np.clip(X @ beta, -8, 5)
        mu = np.exp(eta_lin)
        wt = w * mu                               # IRLS working weights
        z = eta_lin + (y - mu) / mu               # working response
        XtW = X.T * wt
        H = XtW @ X + np.diag(ridge)
        beta_new = np.linalg.solve(H, XtW @ z)
        step = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if step < 1e-10:
            break

    alpha = beta[0]
    attack = beta[1:1 + T].copy()
    defence = beta[1 + T:1 + 2 * T].copy()
    gamma, delta, rho, h_adv = beta[1 + 2 * T:]

    # Sum-to-zero recentring (predictions unchanged)
    alpha = alpha + attack.mean() - defence.mean()
    attack -= attack.mean()
    defence -= defence.mean()

    coeffs = {
        "alpha": float(alpha),
        "gamma": float(gamma / ELO_SCALE),        # back to per-Elo-point
        "delta": float(delta),
        "rho": float(rho),
        "home_advantage": float(h_adv),
    }
    strengths = pd.DataFrame(
        {"attack": attack, "defence": defence}, index=teams)
    diag = {"n_matches": M, "n_obs": n, "n_teams": T,
            "effective_obs": float(w.sum()), "iterations": it + 1,
            "final_step": float(step)}
    return coeffs, strengths, diag


# ---------------------------------------------------------------------------
# Pass 3: tie-break slope eta from penalty shoot-outs
# ---------------------------------------------------------------------------

def fit_eta(results: pd.DataFrame, shootouts: pd.DataFrame) -> tuple[float, int]:
    """MLE of P(i wins shoot-out) = sigmoid(eta * (E_i - E_j))."""
    pre = {(r.date, r.home_team, r.away_team): (r.elo_h, r.elo_a)
           for r in results.itertuples(index=False)}
    x = []
    for r in shootouts.itertuples(index=False):
        key = (r.date, r.home_team, r.away_team)
        if key not in pre or r.winner not in (r.home_team, r.away_team):
            continue
        eh, ea = pre[key]
        x.append(eh - ea if r.winner == r.home_team else ea - eh)
    x = np.asarray(x, float)

    eta = 0.001
    for _ in range(100):                          # Newton-Raphson
        s = 1.0 / (1.0 + np.exp(-eta * x))
        grad = np.sum(x * (1.0 - s))
        hess = -np.sum(x * x * s * (1.0 - s))
        step = grad / hess
        eta -= step
        if abs(step) < 1e-12:
            break
    return float(eta), len(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--shootouts", default="shootouts.csv")
    ap.add_argument("--out", default=TEAM_OUT)
    args = ap.parse_args()

    results = pd.read_csv(args.results, parse_dates=["date"])
    results = results.sort_values("date", kind="stable").reset_index(drop=True)
    shootouts = pd.read_csv(args.shootouts, parse_dates=["date"])
    print(f"Loaded {len(results):,} matches "
          f"({results['date'].min().date()} .. {results['date'].max().date()})")

    results = compute_history(results)

    print("\nCurrent Elo ratings (this replication of eloratings.net):")
    elo_now = results.attrs["elo"]
    for team in sorted(QF_TEAMS, key=lambda t: -elo_now[t]):
        print(f"  {team:<12} {elo_now[team]:7.0f}")

    coeffs, strengths, diag = fit_poisson(results)
    eta, n_so = fit_eta(results, shootouts)
    coeffs["eta"] = eta

    print(f"\nPoisson fit: {diag['n_matches']:,} matches since {FIT_START} "
          f"({diag['n_obs']:,} observations, effective "
          f"{diag['effective_obs']:,.0f}), {diag['n_teams']} teams, "
          f"{diag['iterations']} IRLS iterations.")
    print(f"Tie-break fit: {n_so} shoot-outs.")
    print("\nFitted coefficients (paste into simulate_worldcup.py):")
    for k in ("alpha", "gamma", "delta", "rho", "eta", "home_advantage"):
        print(f"  {k:<16} {coeffs[k]: .6f}")

    # Snapshot covariates for the eight quarterfinalists at their QF date
    last5 = results.attrs["last5"]
    last_date = results.attrs["last_date"]
    rows = []
    for team in QF_TEAMS:
        qf = pd.Timestamp(QF_DATE[team])
        rows.append({
            "team": team,
            "elo": round(elo_now[team], 1),
            "attack": round(float(strengths.loc[team, "attack"]), 4),
            "defense": round(float(strengths.loc[team, "defence"]), 4),
            "form": round(float(np.mean(last5[team])), 2),
            "rest_days": int(min((qf - last_date[team]).days, REST_CAP)),
        })
    table = pd.DataFrame(rows)
    print("\nQuarterfinalist covariates:")
    print(table.to_string(index=False))

    header = (
        "# Model inputs estimated from real data (see fit_parameters.py).\n"
        f"# Sources: github.com/martj42/international_results "
        f"(matches through {results['date'].dropna().max().date()}),\n"
        "# Elo replicated with the eloratings.net algorithm; attack/defense,\n"
        "# form and rest from the same dataset. Columns as in main.tex, Sec. 7.\n"
    )
    with open(args.out, "w") as f:
        f.write(header)
        table.to_csv(f, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
