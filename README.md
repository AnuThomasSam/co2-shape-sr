# co2-shape-sr — IT9115 code base

Shape-constrained symbolic regression (DCSR) for vehicle CO2 prediction, following the revised plan.

## Install
    python -m venv .venv && source .venv/bin/activate   # Python 3.12 recommended
    pip install -r requirements.txt
    python -m pytest -q                                  # 18 tests: derivatives, intervals, scaling, splits, ingest

## Run
    python pipeline.py --config configs/smoke.yaml run all   # quick end-to-end check (not research results)
    python pipeline.py run all                               # full configuration
    python pipeline.py status
    python pipeline.py run 07 --experiment main --methods dcsr_s --seeds 0-9 --n-jobs 8

## Pipeline steps (each reads only files written by earlier steps)
| Step | Writes to | Content |
|---|---|---|
| 01 ingest | outputs/01_ingest | harmonised raw files, whitelisted columns only |
| 02 clean | outputs/02_clean | cleaned data, cleaning log, data card |
| 03 prior | outputs/03_prior | matched-pair support for the monotone prior |
| 04 proxy | outputs/04_proxy | K-Means technology proxy and its validity (RQ1) |
| 05 splits | outputs/05_splits | P1 rolling-origin, P2 extrapolation, P3 proxy, P4 grouped CV; information ceilings |
| 06 domain | outputs/06_domain | per block: domain sample, evaluation grid, interval boxes, DCSR-V root boxes (all contexts) |
| 07 train | outputs/07_models/<ds>/<protocol>/<block>/<method>/seedNN | model.joblib, model.json (GP), predictions.csv, history.csv, run.json |
| 08 evaluate | outputs/08_evaluation | per-run metrics, domain violations, plain-interval certificate and branch-and-bound verdict (every GP run), complexity, response curves |
| 09 statistics | outputs/09_statistics | summaries, Mann-Whitney + A12 (Holm), Friedman-Nemenyi, constraint validity, verdict shares, interval false rejections and grid false acceptances, DCSR-V verification cost, lambda ablation, proxy vs temporal ranking |
| 10 audit | outputs/10_audit | manufacturer residual audit (Kruskal-Wallis + Dunn) |
| 11 figures | results/figures | PNG + PDF, 300 dpi, no titles |
| 12 tables | results/tables | booktabs LaTeX tables + CSV copies |
| 13 benchmark | outputs/13_benchmark | synthetic shape-constrained problems (Feynman equations, constraints derived from the true expressions); one JSON per run |
| 14 benchmark report | results/benchmark | benchmark tables (tab11-13) and figures (fig16-18), read from step 13 only |

A step is skipped when its config section, its input files and its code are unchanged (manifest in each folder).
Steps 07 and 08 cache per run: changing one method in the config re-runs only that method.
Note: editing a model/engine source file changes the code hash and re-trains the affected runs.

Tools: `tools/whatif.py` sweeps one input of a saved model and refuses points outside the declared domain.

## DCSR-V (proposed method, `constraint: verified`, code in src/co2sr/gp/verify.py)
Each candidate equation gets one of three verdicts on the declared domain D:
- **refuted**: an exact counterexample was found (a point where the slope is negative);
- **certified**: interval arithmetic proved the constraint on every box of an adaptive partition of D;
- **undecided**: neither, within the budget (`verify_max_boxes`, `verify_max_depth`).

Order of work per candidate: (1) cheap screen on the training points and on the archive of earlier
counterexamples; (2) branch-and-bound: bound the slope on each open box, probe its midpoint and corners,
bisect boxes that are neither proved nor refuted. Counterexamples join the archive (`counterexample_feedback`).
Selection uses Deb's rules with three levels: certified > undecided > refuted (`undecided_policy`).
Plain DCSR-IA is the special case without bisection and without probing.

Ablations: `dcsr_v_no_bisection`, `dcsr_v_no_feedback`, `dcsr_v_undecided_infeasible`,
`dcsr_v_undecided_feasible`, `dcsr_v_budget_small`. Baselines for the constraint handling: `dcsr_s`
(sampling penalty), `dcsr_f` (sampling, Deb), `dcsr_ia` (plain intervals), `dcsr_dyn` (penalty raised
geometrically during the run, in the spirit of Haider et al., EUROCAST 2024; not a reimplementation).

Step 08 computes the branch-and-bound verdict of every GP final model, whichever method produced it, with a
large budget. This measures how often plain interval arithmetic rejects a valid model (false rejection) and
how often a model that passes the sampled grid is in fact invalid (false acceptance).

Tests: `tests/test_verify.py` checks soundness on 600 random trees (no certified model is contradicted by
dense sampling), that every counterexample is a true violation, and a dependency-problem case that plain
intervals cannot prove and bisection proves.

## GP engine options (configs/default.yaml, gp_defaults; each can be switched off for ablation)
- `require_engine_size: true` — equations without engine size rank below all others. Without this, a
  constrained run can satisfy monotonicity trivially by dropping engine size (seen in a full-budget test run).
- `sign_consistent_scaling: true` — fuel groups rescale the shared shape g(x) in the same direction as the
  pooled fit. Without it, one test run gave diesel a negative slope, reversing every effect for diesel.
- `parsimony: 0.0002` and `selection_tolerance: 0.01` — bloat control during evolution, and choice of the
  smallest equation within 1% of the best validation NMSE. Equations dropped from 37-39 nodes (size cap) to 7-31.
- Ablations for these three options are defined: `gp_no_parsimony`, `dcsr_f_engine_optional`,
  `dcsr_f_free_sign_scaling`.

## Compute estimate (measured on a 2-core machine, full default budget, one seed)
| Method | Fit time per run (P1 block, ~2,000 rows) | (P4 fold, ~6,000 rows) |
|---|---|---|
| Standard GP | 13 s | 17 s |
| DCSR-T | 12 s | 26 s |
| DCSR-S | 25 s | 30 s |
| DCSR-F | 17 s | 24 s |
| DCSR-IA | 58 s | 71 s |
| DCSR-V | 75 s | not measured |
| Dynamic penalty | 41 s | not measured |
| XGBoost (40 Optuna trials) | 7 s | 12 s |
DCSR-V, DCSR-IA (77 s) and the dynamic penalty were timed with three runs in parallel on one P1 block.
Roughly 28 CPU-hours for the experiments of the second revision; DCSR-V and the dynamic penalty add about
14 CPU-hours (main + proxy) and the new ablations about 10 CPU-hours, on the test data's block count.
Benchmark (steps 13-14), one run on I.12.2 (4 inputs, 4 constraints, the largest problem): Standard GP 5 s,
DCSR-F 27 s, DCSR-IA 74 s, DCSR-V 158 s; the full benchmark needs at most about 36 CPU-hours.
Start with `--seeds 0-4` to check results before launching all 30 seeds; later seeds are added without
re-running earlier ones.

## Not yet done
- Protocol P5 (train on NRCan, test on US EPA fueleconomy.gov data) is optional in the plan and not implemented.
- PySR could not be tested here (its Julia runtime could not be downloaded). Test it first on the real machine.
- The code was tested on the Kaggle file and on a third-party copy of NRCan data, not on the official NRCan
  files. Check `outputs/01_ingest/nrcan_ingest_report.csv` after step 01: it lists which column was mapped
  from which header, and which columns were dropped.

## Notes
- PySR is optional; if it cannot be imported the runs are recorded as skipped.
- A constrained model can satisfy monotonicity trivially by ignoring engine size; step 08 records
  `uses_engine_size` and step 09 reports the share of such runs. Check it before interpreting DCSR results.
- Write YAML floats with a signed exponent (1.0e+9), otherwise PyYAML reads them as strings.
