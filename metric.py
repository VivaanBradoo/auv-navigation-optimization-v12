"""Standalone Kaggle custom metric. Copy this file into the metric notebook."""

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


TARGET = "cross_track_error_m"
MAX_ABS_PREDICTION = 1_000_000.0


def navigation_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """100/(1+RMSE); metres are numerically normalized by one metre."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or pred.shape != truth.shape or truth.size == 0:
        raise ParticipantVisibleError("Predictions must be a nonempty one-dimensional vector.")
    if not np.isfinite(pred).all() or (np.abs(pred) > MAX_ABS_PREDICTION).any():
        raise ParticipantVisibleError("Predictions must be finite numbers with absolute value at most 1000000.")
    if not np.isfinite(truth).all() or (truth < 0).any() or (truth > MAX_ABS_PREDICTION).any():
        raise ValueError("Invalid organizer ground truth.")
    error = pred - truth
    scale = float(np.max(np.abs(error)))
    rmse = 0.0 if scale == 0 else scale * float(np.sqrt(np.mean((error / scale) ** 2)))
    return float(100.0 / (1.0 + rmse))


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """AUV Navigation Score: 100 / (1 + root mean squared error in metres).

    Higher is better; perfect predictions score 100. The metric computes RMSE
    across all rows in the leaderboard partition, then transforms it once.
    Required prediction columns: row ID and cross_track_error_m, exactly once.
    IDs must uniquely match the solution; row order does not affect scoring.
    Predictions must be finite real numbers with absolute value <= 1,000,000.
    Negative predictions are allowed and penalized normally. No clipping,
    rounding, weighting, normalization by label spread, or mutation occurs.
    Kaggle's Usage column selects Public/Private partitions outside this metric.
    """
    required = [row_id_column_name, TARGET]
    if len(submission.columns) != 2 or set(submission.columns) != set(required):
        raise ParticipantVisibleError("Submission must contain exactly the ID and cross_track_error_m columns.")
    if row_id_column_name not in solution or TARGET not in solution:
        raise ValueError("Invalid organizer solution schema.")
    if len(solution) == 0 or len(submission) != len(solution):
        raise ParticipantVisibleError("Submission row count must match the required rows.")
    for frame, is_submission in [(solution, False), (submission, True)]:
        ids = frame[row_id_column_name]
        if ids.isna().any() or ids.duplicated().any():
            if is_submission:
                raise ParticipantVisibleError("Submission IDs must be present and unique.")
            raise ValueError("Invalid organizer IDs.")
    # No string coercion: changing ID spelling/type must not produce collisions.
    if set(submission[row_id_column_name]) != set(solution[row_id_column_name]):
        raise ParticipantVisibleError("Submission IDs must match the required IDs exactly.")
    values = submission.set_index(row_id_column_name).loc[solution[row_id_column_name], TARGET]
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype) or pd.api.types.is_complex_dtype(values.dtype):
        raise ParticipantVisibleError("Predictions must be numeric real values.")
    try:
        pred = values.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError, OverflowError):
        raise ParticipantVisibleError("Predictions must be finite numeric real values.") from None
    return navigation_score(solution[TARGET].to_numpy(dtype=np.float64), pred)
