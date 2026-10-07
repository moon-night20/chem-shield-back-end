"""Validation and feature engineering for physical-process telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = ("rpm", "temperature", "pressure", "power")
REQUIRED_TELEMETRY_COLUMNS = (*BASE_FEATURE_COLUMNS, "target_rpm")
RATE_FEATURE_COLUMNS = (
    "rpm_rate_of_change",
    "temperature_rate_of_change",
    "pressure_rate_of_change",
)
FEATURE_COLUMNS = (
    *BASE_FEATURE_COLUMNS,
    "rpm_deviation",
    *RATE_FEATURE_COLUMNS,
)
RATE_SOURCE_COLUMNS = {
    "rpm_rate_of_change": "rpm",
    "temperature_rate_of_change": "temperature",
    "pressure_rate_of_change": "pressure",
}


class TelemetryValidationError(ValueError):
    """Raised when telemetry cannot be safely transformed into ML features."""


def _validate_dataframe(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("telemetry must be provided as a pandas DataFrame")
    if data.empty:
        raise TelemetryValidationError("telemetry must contain at least one observation")

    missing = [column for column in REQUIRED_TELEMETRY_COLUMNS if column not in data.columns]
    if missing:
        raise TelemetryValidationError(
            "telemetry is missing required process columns: " + ", ".join(missing)
        )


def _validate_timestamp(data: pd.DataFrame) -> None:
    if "timestamp" not in data.columns:
        return
    timestamp = data["timestamp"]
    if timestamp.isna().any():
        raise TelemetryValidationError("timestamp values must not be missing")
    if not timestamp.is_monotonic_increasing:
        raise TelemetryValidationError(
            "telemetry must be ordered by non-decreasing timestamp before feature generation"
        )


def _clean_process_values(data: pd.DataFrame) -> pd.DataFrame:
    numeric = data.loc[:, list(REQUIRED_TELEMETRY_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_columns = [
        column
        for column in REQUIRED_TELEMETRY_COLUMNS
        if numeric[column].isna().any()
        or not np.isfinite(numeric[column].to_numpy(dtype=float)).all()
    ]
    if invalid_columns:
        raise TelemetryValidationError(
            "process values must be finite numeric values with no missing entries: "
            + ", ".join(invalid_columns)
        )
    return numeric.astype(float)


def _coerce_previous_state(
    previous_state: Mapping[str, Any] | None,
) -> dict[str, float] | None:
    if previous_state is None:
        return None
    if not isinstance(previous_state, Mapping):
        raise TypeError("previous_state must map process variable names to values")

    state: dict[str, float] = {}
    for column in RATE_SOURCE_COLUMNS.values():
        if column not in previous_state:
            raise TelemetryValidationError(
                f"previous_state is missing required process value: {column}"
            )
        value = pd.to_numeric(previous_state[column], errors="coerce")
        if pd.isna(value) or not np.isfinite(float(value)):
            raise TelemetryValidationError(
                f"previous_state value for {column} must be finite and numeric"
            )
        state[column] = float(value)
    return state


def build_feature_frame(
    data: pd.DataFrame,
    *,
    previous_state: Mapping[str, Any] | None = None,
    stream_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return validated telemetry and the consistently ordered ML features.

    Rate features are first differences per observation. The first sample in a
    stream receives zero unless a previous sensor state is provided.
    """

    _validate_dataframe(data)
    _validate_timestamp(data)
    if stream_column is not None and stream_column not in data.columns:
        raise TelemetryValidationError(f"stream column is not present: {stream_column}")
    if stream_column is not None and data[stream_column].isna().any():
        raise TelemetryValidationError("stream column values must not be missing")
    if stream_column is not None and previous_state is not None:
        raise TelemetryValidationError("provide either stream_column or previous_state, not both")

    prepared = data.copy().reset_index(drop=True)
    cleaned = _clean_process_values(prepared)
    prepared.loc[:, list(REQUIRED_TELEMETRY_COLUMNS)] = cleaned
    previous = _coerce_previous_state(previous_state)

    features = prepared.loc[:, list(BASE_FEATURE_COLUMNS)].copy()
    features["rpm_deviation"] = cleaned["rpm"] - cleaned["target_rpm"]

    for feature_name, source_name in RATE_SOURCE_COLUMNS.items():
        if stream_column is None:
            rates = cleaned[source_name].diff()
            first_rate = (
                cleaned[source_name].iloc[0] - previous[source_name]
                if previous is not None
                else 0.0
            )
            rates.iloc[0] = first_rate
        else:
            rates = cleaned[source_name].groupby(
                prepared[stream_column], sort=False
            ).diff().fillna(0.0)
        features[feature_name] = rates.astype(float)

    features = features.loc[:, list(FEATURE_COLUMNS)].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise TelemetryValidationError("generated features must be finite")
    return prepared, features


def last_process_state(prepared_telemetry: pd.DataFrame) -> dict[str, float]:
    """Return the final sensor values needed to continue rate calculation."""

    _validate_dataframe(prepared_telemetry)
    return {
        column: float(prepared_telemetry[column].iloc[-1])
        for column in RATE_SOURCE_COLUMNS.values()
    }
