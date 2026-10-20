# AUV Navigation Optimization v12

**Private leaderboard score: 65.904**

Physics-informed endpoint navigation for the AUV Navigation Challenge. The model estimates an underwater vehicle's distance from its planned straight route using one noisy initial position fix and onboard sensor sequences.

The solution combines mission-level instrument calibration, joint velocity/current smoothing, uncertainty-weighted population calibration, and a small residual-learning ensemble. It uses CPU inference and training.

## Results

| Evaluation | Score | RMSE (m) |
|---|---:|---:|
| **Private leaderboard** | **65.904** | Approximately 0.51736, implied by the displayed score |
| Local mission-grouped validation, v12 | 71.038781 | 0.407681809 |
| Matched local validation, v11 | 71.014473 | 0.408163657 |

The metric is `100 / (1 + RMSE_m)`. Local validation uses 25 folds and three seeds, holding out entire missions. Predictions are averaged across seeds before calculating the score. The local dataset informed model selection; its score measures a different partition from the private leaderboard. Private conditions include distribution shifts, so the local result should not be treated as the private result.

Aggregate results and ensemble weights are recorded in [results_summary.json](results_summary.json). See [METHOD.md](METHOD.md) for the model and [EXPERIMENTS.md](EXPERIMENTS.md) for development history and rejected alternatives.

## Reproduce

Python 3.13 was used for development. Install the recorded dependencies:

```bash
python -m pip install -r requirements.txt
```

Obtain the original competition files through the authorized competition download and place these five files beside `navigation_v12.py`:

```text
train.csv
test.csv
train_timeseries.parquet
test_timeseries.parquet
sample_submission.csv
```

Run the full reconstruction and training pipeline:

```bash
python navigation_v12.py --rebuild --refit-core
```

After the first run, unchanged feature caches can be reused:

```bash
python navigation_v12.py
```

Validate generated outputs:

```bash
python audit_v12.py
```

The pipeline writes `submission_v12.csv`, an identical `submission.csv`, out-of-fold predictions, a validation report and component caches. These generated artifacts are ignored by Git. The default validation configuration is 25 folds with seeds 2026, 731 and 1907.

The repository includes source code and aggregate results. Competition data, prediction rows, fitted parameters and derived sensor caches are not distributed. Historical v10/v11 out-of-fold CSVs are optional cross-checks; rebuilding v12 does not require them. If supplied locally, the original comparison checks still run.

## Repository map

| File | Purpose |
|---|---|
| `navigation_v12.py` | Main training, grouped validation and submission pipeline |
| `production_sensor.py` | Selected sensor reconstruction configuration |
| `environment_solver.py` | Independent window states with shared mission calibration and environmental current |
| `population_model.py` | Fold-trained population-prior updates |
| `calibration_search.py` | Fixed uncertainty-weighted calibration |
| `compare_environment.py` | Posterior displacement variance calculation |
| `baseline_features.py` | Regeneration and fingerprint checks for inherited feature stages |
| `navigation_v3.py`, `navigation_v4.py`, `motion_v5.py`, `joint_v5.py` | Earlier reconstruction stages required by the ensemble |
| `adaptive_sensor.py`, `colored_sensor.py`, `mixture_sensor.py`, `mixture_solver.py` | Complementary legacy ensemble components |
| `model_experiment.py` | Feature assembly and calibration helpers |
| `features.py`, `metric.py` | Competition feature/rotation and evaluation utilities |
| `audit_v12.py` | Submission, metric, fingerprint and ensemble checks |

## Data and attribution

This is a synthetic educational navigation benchmark organized by AUV VIT Pune. It is not a certified navigation system or evidence of real-world deployment performance. Competition data must be obtained and used under the organizer's terms. Attribution and the supplied-code notice are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
