"""Participant-safe summary features and a body-to-world rotation helper."""

import numpy as np
import pandas as pd

ANGLE_COLUMNS = ["roll_rad", "pitch_rad", "yaw_rad", "desired_heading_rad"]
MISSING_COLUMNS = ["dvl_x_mps", "dvl_y_mps", "dvl_z_mps", "dvl_age_s", "altitude_m"]
META_COLUMNS = ["Id", "mission_id", "track_heading_rad", "duration_s", "fix_north_m", "fix_east_m", "fix_sigma_m"]


def rotation_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """ZYX body (forward/right/down) -> world (north/east/down)."""
    r, p, y = np.broadcast_arrays(roll, pitch, yaw)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack([
        cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
        sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
        -sp, cp*sr, cp*cr,
    ], axis=-1).reshape(r.shape + (3, 3))


def make_summary(raw: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Same deterministic function for train and test; never accepts labels."""
    raw = raw.sort_values(["Id", "step"], kind="stable")
    excluded = {"Id", "step", "t_s", *ANGLE_COLUMNS}
    columns = [c for c in raw.columns if c not in excluded]
    grouped = raw.groupby("Id", sort=False)
    stats = grouped[columns].agg(["mean", "std", "min", "max", "last"])
    # pandas "last" skips missing values, so call it last_valid.
    stats.columns = [f"{column}__{'last_valid' if stat == 'last' else stat}" for column, stat in stats.columns]
    missing = raw[MISSING_COLUMNS].isna().groupby(raw["Id"], sort=False).mean()
    missing.columns = [f"{c}__missing_fraction" for c in MISSING_COLUMNS]
    circular = raw[["Id"]].copy()
    for column in ANGLE_COLUMNS:
        circular[f"{column}__sin"] = np.sin(raw[column])
        circular[f"{column}__cos"] = np.cos(raw[column])
    angles = circular.groupby("Id", sort=False).agg(["mean", "std"])
    angles.columns = [f"{c}__{s}" for c, s in angles.columns]
    features = stats.join(missing).join(angles).reset_index()
    result = metadata[META_COLUMNS].merge(features, on="Id", validate="one_to_one", sort=False)
    result["track_heading_sin"] = np.sin(result["track_heading_rad"])
    result["track_heading_cos"] = np.cos(result["track_heading_rad"])
    for column in result.select_dtypes(include="number").columns:
        result[column] = result[column].astype(np.float32)
    return result


def model_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.select_dtypes(include="number").columns if c != "cross_track_error_m"]
