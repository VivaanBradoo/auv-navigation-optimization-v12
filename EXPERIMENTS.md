# Development and experiments

Scores below use `100 / (1 + RMSE_m)`. Local and leaderboard evaluations are distinct. Different fold counts change the amount of training data; only matched comparisons establish the reported local model gains.

## Version history

| Version | Local score | Reported public score | Main development |
|---|---:|---:|---|
| v1 | 36.206334 | 34.065 | Physical integration features with learned residual correction |
| v2 | 66.676723 | 63.493 | Gyro compass correction, mission calibration, body-frame gap filling and transition acceleration |
| v3 | 69.209612 | 68.640 | Joint velocity/current smoothing and propulsion-based current information |
| v4 | 70.018408 | 68.996 | Soft propulsion constraints and endpoint correction |
| v5 | 70.576822 | 70.058 | Correlated motion residuals and jointly estimated propulsion |
| v6 | 70.596816 | 70.113 | Squared distance calibration with folded-normal correction |
| v7 | 70.772597 | 70.259 | Population prior means/variances and a complementary ensemble |
| v8 | 70.821101 | 70.366 | Separate compass/DVL yaw and axis-specific motion correlation |
| v9 | 70.844121 | 70.498 | Anisotropic DVL observations and spike mixture likelihood |
| v10 | 70.853108 | Unreported | Finite compass-frame relinearization |
| v11 | 71.014473 | 70.501 | Conditional innovations, environmental current and posterior uncertainty |
| v12 | 71.038781 | Unreported | Fixed inverse-variance calibration |

**V12 private leaderboard: 65.904.** Earlier local scores used five mission folds; final v7–v12 scores use 25 folds and three seeds. For example, v7 scored 70.729099 with five folds, and v6 scored 70.644886 when recomputed with the matching 25-fold protocol.

## Successful changes

The largest early gains came from correcting observation semantics and shared sensor errors: mode-aware DVL velocity, delayed ping rotation, compass distortion, sound-speed scaling and mission current calibration. A generic direct regressor scored 32.941350 locally, compared with 36.206334 for the first residual model and 66.676723 after the v2 physical calibration.

Joint current/velocity smoothing and weak propulsion constraints reduced error under water-track coverage and dropouts. Temporally correlated residuals improved the way repeated nearby motion constraints contributed information. Learning mission-parameter population distributions and blending complementary models supplied further gains.

Later gains were smaller. The conditional spike detector improved on a detector that partly fitted away an observation's own error. Independent initial current states around an environmental mean and physical posterior folding raised the five-fold development blend to 70.980310 before the final 25-fold v11 evaluation.

## V12 matched development comparisons

The following use the same five-fold, three-seed mission partitions and 62.5% new / 37.5% legacy blend. Baseline v11: **70.980310**.

| Experiment | Score | Outcome |
|---|---:|---|
| Structured correlations / nonlinear population shrinkage | 70.977578 | Worse |
| Bounded population priors | 70.986540 | Small gain; complexity excluded |
| Revised current observation/process noise | 70.974813 | Worse |
| Separate DVL axis quality/noise corrections | 70.982048 | Negligible gain |
| Tighter environmental-current prior | 70.981610 | Negligible gain |
| Physical-variance calibration objective | 70.985675 | Small gain |
| Half inverse-variance calibration weighting | 71.001663 | Improved |
| **Fixed inverse-variance calibration** | **71.013951** | **Selected** |
| Robust calibration loss | 70.982489 | No meaningful gain |
| Mission-correlated GLS calibration | 71.005763 | Below selected |
| Weighted/deeper residual trees | 71.014758 | Tiny gain over selected; simpler trees retained |
| Position-invariant residual features | 71.013584 | No gain over selected |
| Weighting legacy calibration components | 71.012106 | Below selected |

The selected change then scored **71.038781 versus 71.014473** on the matched final 25-fold comparison. Three additional five-fold partitions scored **70.966527 versus 70.943583**. These are stability checks on the same dataset, not untouched test sets.

## Rejected or inconclusive earlier directions

| Direction | What was tested | Result |
|---|---|---|
| Stronger propulsion constraints | Tightened motion priors | Worse than weak constraints |
| Gyro attitude smoothing | Several early heading/tilt fusion variants | No reliable full-model gain in the evaluated formulations |
| Extra DVL tilt/3D calibration | Roll/pitch and vertical-velocity information | No meaningful improvement over separate compass/yaw calibration |
| More complex residual learners | Deeper/more trees, ridge, kernels, MLPs and CatBoost variants | Mostly worse or small split-specific gains |
| Additional control features | Thrusters, motor load, battery, turn-rate interactions and lags | Weak extra information; no reliable integrated target gain |
| Correlated population priors | Link propulsion and instrument parameter distributions | Below the simpler population model |
| Very smooth current trajectories | Smaller random-walk noise and strong curvature penalties | Large losses: tested full blends 70.696859/70.671417 versus 70.807083 |
| Turn-response delays | Half-second and one-second lateral response delays | Full blends 70.805652/70.803560 versus 70.807083 |
| Axis-correct spike likelihood alone | Separate determinant/Mahalanobis terms | Full blend 70.806524 versus 70.807083 |
| Free additional shared biases | Extend population calibration from 11 to16 parameters | Added flexibility did not beat the selected model |
| Retain all valid pings | Remove median prefilter and rely on iterative classification | Worse; extra iterations alone negligible |
| Soft/joint sign inference | Uncertainty-aware signed targets near zero | No gain over physical folding with the selected signed target |
| Full folded multivariate likelihood | Fit eight-window uncertainty jointly | Underperformed tested point-error calibration; not selected |
| Tighter initial-position prior | Exploit apparent starting-position distribution | Starting cross-track spread was too broad to help |
| Mission compass harmonics | Estimate mission-specific distortion adjustments | Worse held-out sensor consistency than the global correction |
| Initial-fix invariant residual features | Remove/replace fix-dependent inputs | No gain over v12 |

An interrupted leave-one-mission-out experiment was not used as a completed score. Sensor-only diagnostics and partial runs were not treated as competition results.

## Interpretation

Improving the physical reconstruction produced stronger gains than expanding the residual learner. Small improvements across development partitions did not guarantee leaderboard improvement. The final private score of65.904 should be retained alongside the local score when evaluating the solution's generalization.
