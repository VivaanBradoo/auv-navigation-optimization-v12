# Method

## Geometry and observations

The planned route has heading `theta`. Its right-facing cross-track direction is `[-sin(theta), cos(theta)]`. The target is the absolute value of the initial signed cross-track position plus the integral of true ground velocity in that direction.

Each mission contains eight independent windows with shared environment and instrument settings. Windows retain separate trajectories; only shared calibration variables connect them. The logs stop half a second before the target time.

The main observation corrections are:

- Correct DVL velocity for sound speed using temperature and depth.
- Deduplicate repeated pings and evaluate attitude at each ping's delayed timestamp.
- Distinguish bottom-track ground velocity from water-track relative velocity.
- Estimate heading-dependent compass distortion using gyro consistency.
- Account for shared DVL bias, mounting yaw, scale, compass offset and accelerometer bias.
- Downweight spikes using a conditional ordinary/contamination likelihood, with DVL quality supplying additional observation weighting.

## Joint reconstruction

The state system contains north/east ground velocity and current trajectories for each window, local accelerometer terms, and shared mission instrument/propulsion parameters. Independent initial currents vary around a shared environmental-current mean. Mean-reverting current dynamics use correlation 0.98 and process SD 0.008 m/s per half-second.

Through-water motion is softly constrained around a fitted propulsion/turning model. Forward/lateral motion SDs are 0.012823/0.029955 m/s, with correlations 0.77555/0.85832. Their observation errors differ by axis. A finite +0.04 rad working-frame rotation improves the linearization of compass correction.

The spike detector accounts for an observation's influence on its own fitted trajectory. Leave-one-ping-out innovations include conditional state uncertainty before classifying a ping as ordinary or contaminated. The mixture uses probabilities 0.98/0.02 and broad-noise SD 0.15 m/s.

Independent window blocks and a small shared Schur complement make the solve practical on CPU. Posterior covariance and endpoint sensitivities are retained for subsequent calibration.

## Fold-specific population calibration

Shared instrument and propulsion parameters differ between missions. Their population means and variance scales are learned using only the training missions of each validation fold. Saved sensor posterior means, covariance and endpoint sensitivities support a small posterior update without rebuilding the full trajectory for each population fit.

The v12 change is a second calibration fit with fixed uncertainty weights:

1. Fit the original population calibration on the training missions.
2. Calculate posterior displacement variance plus known initial-fix variance.
3. Set residual weights proportional to inverse posterior standard deviation, clip to `[0.4, 2.5]`, and normalize their mean square to one.
4. Refit calibration while keeping those weights fixed.

Keeping weights fixed prevents the fit from improving its weighted objective by inflating its own predicted uncertainty. Mean/log-SD/basis regularization remains unchanged.

## Residual model and absolute-distance prediction

A small LightGBM model estimates the remaining signed navigation error. The primary setting uses 150 trees, five leaves, learning rate 0.015, minimum child size 70, L2 penalty 30 and column fraction 0.85.

The signed prediction is converted to expected absolute distance under a normal uncertainty approximation. For the new component, the standard deviation comes from posterior displacement variance plus fix variance and is clipped to `[0.12, 1.2]` metres.

The final weights are:

| Component | Weight |
|---|---:|
| Uncertainty-weighted environmental-current reconstruction | 0.625 |
| Adaptive-prior legacy reconstruction | 0.1875 |
| Legacy baseline | 0.09375 |
| Colored-motion legacy reconstruction | 0.046875 |
| Linear-tree legacy component | 0.046875 |

The older components contribute complementary errors. More weight on the individually strongest component did not consistently improve the full blend.

## Validation and limitations

All label-dependent fitting holds out complete missions. IDs serve only as alignment and grouping keys. No windows are stitched together and no hidden test labels enter training.

The final local score is 71.038781 with RMSE 0.407681809 m. The private leaderboard score is 65.904. Additional five-fold partitions supported a small local improvement over v11, but they reused the same training dataset. The private result demonstrates that local development performance did not transfer fully to the shifted evaluation partition.
