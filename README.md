# World Cup Poisson Model — Conditional Champion Prediction from the Quarterfinals

An Overleaf-ready LaTeX note plus a runnable Python implementation of a
conditional champion-prediction model for the 2026 FIFA World Cup, given that
the tournament has reached the quarterfinal stage.

**Dashboard:** [open the prediction dashboard](https://styyyyyyy.github.io/Worldcup_Prediction/)
or view the static source in [`index.html`](index.html).
If the dashboard URL is not live yet, enable GitHub Pages from this repository's
Settings using `main` / root as the source.

**Model in one line:** independent Poisson goals with covariate-driven log-rates
(attack, defence, Elo, form, rest), a logistic Elo tie-break for drawn knockout
matches, and Monte Carlo simulation of the remaining seven matches to estimate
`P(team wins the World Cup | team is in the quarterfinals)`.

## Prediction results

**All inputs are estimated from real data through 7 July 2026** (the eve of the
quarterfinals). The table reports
`P(team wins the World Cup | team reached the quarterfinals)`.

| Rank | Team | Champion probability |
| ---: | --- | ---: |
| 1 | Spain | 25.5% |
| 2 | Argentina | 22.9% |
| 3 | England | 14.5% |
| 4 | France | 13.3% |
| 5 | Morocco | 8.8% |
| 6 | Norway | 5.9% |
| 7 | Belgium | 4.8% |
| 8 | Switzerland | 4.2% |

The machine-readable output is saved in
[`champion_probabilities.csv`](champion_probabilities.csv). Re-running
`python3 simulate_worldcup.py` reproduces the same table with seed `42`; see
*Data provenance* below for the source data and model-fitting details.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | The paper: model, math, estimation, tables, interpretation (compile this) |
| `algorithm.tex` | Pseudocode for the Monte Carlo simulation (`\input` by `main.tex`) |
| `model_notes.tex` | Appendix: conditionality, why Monte Carlo, bracket-path effects, extensions (`\input` by `main.tex`) |
| `references.bib` | BibTeX references (all verified; incl. the two data-source entries) |
| `data_template.csv` | Estimated team covariates (Elo, attack, defence, form, rest days) |
| `fit_parameters.py` | Estimates everything from the raw match data (see below) |
| `simulate_worldcup.py` | Simulates the remaining bracket and prints/saves champion probabilities |
| `results.csv` | **Archived raw data**: 49,505 international matches, 1872 → 7 July 2026 |
| `shootouts.csv` | **Archived raw data**: 681 penalty shoot-outs, 1967–2026 |
| `index.html` | Static dashboard for GitHub Pages |
| `README.md` | This file |

Running the simulation additionally generates `champion_probabilities.csv`.

## Compiling on Overleaf

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project** →
   **Upload Project** and select `world_cup_poisson_model.zip`.
2. Overleaf opens the project with `main.tex` as the main document. (If the
   files appear nested inside a `world_cup_poisson_model/` folder, either drag
   them to the project root or set `main.tex` as the main document via
   **Menu → Main document**.)
3. Compile with the default **pdfLaTeX** compiler. The bibliography builds
   automatically; no shell-escape, custom fonts, or images are required.

Locally: `latexmk -pdf main.tex`

## Running the simulation locally

Requires Python ≥ 3.9 with `numpy` and `pandas`:

```bash
pip install numpy pandas
python3 simulate_worldcup.py
```

The script reads `data_template.csv`, prints the analytic advancement
probability for each quarterfinal, simulates the remaining bracket 100,000
times with `numpy.random.default_rng(seed=42)` (~1 s, fully reproducible),
prints the sorted champion probabilities, and writes
`champion_probabilities.csv` (columns: `team,champion_probability`).

Output with the shipped inputs:

```
Analytic advancement probabilities (score grid truncated at 10 goals):
  France       vs Morocco       P(France advances) = 0.5516
  Spain        vs Belgium       P(Spain advances) = 0.7060
  Norway       vs England       P(Norway advances) = 0.3919
  Argentina    vs Switzerland   P(Argentina advances) = 0.6944

Conditional champion probabilities from 100,000 simulated tournaments (seed = 42):
  Spain        0.2554
  Argentina    0.2291
  England      0.1451
  France       0.1333
  Morocco      0.0881
  Norway       0.0587
  Belgium      0.0480
  Switzerland  0.0424
```

## Data provenance and refitting

Everything numerical is produced by `fit_parameters.py` from one public
dataset: [martj42/international_results](https://github.com/martj42/international_results)
(49,505 international matches, 1872 → 7 July 2026, including the whole 2026
World Cup through the round of 16). **Archived copies of both raw files ship
with this project** (`results.csv`, `shootouts.csv`, retrieved 9 July 2026),
so every number is reproducible offline. MD5 checksums:
`3820fec01802a4303d91b00d6e298bed` (results),
`3461b3d538309f7f17b63259e4b0b524` (shootouts).

- **Elo ratings** — the full history is replayed with the
  [eloratings.net](https://www.eloratings.net) update rule (importance-based
  K factors, goal-difference multiplier, +100 home advantage, shoot-outs as
  draws). The replication matches the published 7 July 2026 ranking of the
  eight quarterfinalists and their gaps to within a few points (a uniform
  level offset of ≈ +70 cancels out, since only Elo *differences* enter the
  model).
- **Attack/defence and coefficients (α, γ, δ, ρ)** — time-decayed Poisson
  regression (IRLS) on all 10,048 internationals since 2016, three-year
  half-life, ridge on team dummies, home-advantage term included (fitted
  ≈ +26% goals — the textbook value).
- **Tie-break slope η** — logistic MLE on 681 recorded penalty shoot-outs
  (a 100-point Elo edge gives only a 53% shoot-out win probability).
- **Form** — mean goal difference over each team's last five internationals
  (= its five 2026 World Cup matches). **Rest days** — round-of-16 date to
  quarterfinal date.

To reproduce the shipped numbers exactly, run on the archived data:

```bash
python3 fit_parameters.py            # rewrites data_template.csv, prints the coefficients
python3 simulate_worldcup.py         # reproduces the champion table (seed 42)
```

To refresh after new results (e.g., after the quarterfinals are played),
re-download first:

```bash
curl -sL -o results.csv   https://raw.githubusercontent.com/martj42/international_results/master/results.csv
curl -sL -o shootouts.csv https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv
python3 fit_parameters.py            # rewrites data_template.csv, prints coefficients
# paste the printed coefficients into COEFFS in simulate_worldcup.py, then
python3 simulate_worldcup.py
```

(For a different tournament, also edit `QUARTERFINALS` in
`simulate_worldcup.py` and `QF_DATE` in `fit_parameters.py`.)

## Interpreting the output

- Each number is `P(team wins the World Cup | team reached the quarterfinals)`
  under the model. The eight probabilities sum to 1; teams eliminated before
  the quarterfinals carry probability 0 by construction.
- Monte Carlo noise is ≤ ~0.0016 (standard error at 100,000 replications);
  differences beyond the third decimal are meaningless.
- The numbers mix team strength **and** bracket position: England outranks
  France here despite a lower Elo because France's path runs through Morocco
  (the best-fitted defence of the eight) and then, most likely, Spain.
- Remaining caveats (paper, Sec. 10): independence of the two teams' goal
  counts, static strengths within the simulated bracket, extra-time goals
  contaminating recorded 90-minute scores, η identified from shoot-outs only,
  and no player-level (injury/suspension) information.
