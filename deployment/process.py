#!/usr/bin/env python3
"""
Turn the validated readings CSV written by predict_co2.py into the
feature row the model expects, run the prediction, and write the
result to a file.

Feature engineering (load_readings through build_feature_row) is a
straight port of notebooks/serve_01, which itself mirrors the
preprocessing (train_02) and feature engineering (train_04) steps of
the training pipeline - see PROJECT.md's "Train/serve feature parity"
invariant. Any feature change in train_04 must be re-applied here by
hand until the shared package (PROJECT.md open issue 6) exists.

Model loading and prediction (load_model, predict) follow what
PROJECT.md documents about serve_02 and model-regressor.pkl: the file
unpickles to {"model": fitted_model, "feature_names": list}; the
feature row is subset to feature_names (order and set must match what
the model was trained on) before predicting.

Posting the prediction back into Home Assistant is out of scope here;
this module only writes the predicted value to a local file.

run_prediction() ties the steps above together and is what
predict_co2.py calls after it writes the readings CSV; this module's
own main() is a thin standalone CLI around the same function, useful
for re-running a prediction against an existing CSV without
re-fetching from HA.

Environment variables:
    HA_OUTPUT_PATH        - path to the readings CSV to read (same
                             variable predict_co2.py uses to decide
                             where it wrote the CSV; default:
                             last_co2_values.csv)
    MODEL_PATH             - path to the model pickle. Default is
                              relative ("model-regressor.pkl"), since
                              the Dockerfile COPYs it into the same
                              WORKDIR the script runs from; "deployment"
                              is only meaningful as the source-tree
                              location, not inside the container.
    PREDICTION_OUTPUT_PATH - path to write the prediction JSON to
                              (default: predicted_co2.json)

Exit codes (this module's own CLI):
    0 - success
    1 - readings CSV or model file not found
    2 - feature row contains NaN (upstream gap), or prediction failed
        (e.g. a feature_names mismatch against the loaded model)
"""

import json
import logging
import os
import pickle
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    DateOffset,
    GoodFriday,
    Holiday,
    MO,
    next_monday,
    next_monday_or_tuesday,
)

log = logging.getLogger(__name__)

LOCAL_TZ = "America/Toronto"
LAG_LIST = [10, 20, 30, 60]
ROLLING_WINDOWS = [10, 20, 30, 60]
HORIZON_MINUTES = 10  # forecast horizon the model was trained for

DEFAULT_CSV_PATH = "last_co2_values.csv"
CSV_INPUT_PATH = os.environ.get("HA_OUTPUT_PATH", DEFAULT_CSV_PATH)

DEFAULT_MODEL_PATH = "model-regressor.pkl"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

DEFAULT_PREDICTION_PATH = "predicted_co2.json"
PREDICTION_OUTPUT_PATH = os.environ.get(
    "PREDICTION_OUTPUT_PATH", DEFAULT_PREDICTION_PATH
)


class OntarioCalendar(AbstractHolidayCalendar):
    """Ontario statutory holidays, ported verbatim from serve_01."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=next_monday),
        Holiday("Family Day", month=2, day=1, offset=DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday("Victoria Day", month=5, day=25, offset=DateOffset(weekday=MO(-1))),
        Holiday("Canada Day", month=7, day=1, observance=next_monday),
        Holiday("Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=10, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday("Christmas Day", month=12, day=25, observance=next_monday),
        Holiday("Boxing Day", month=12, day=26, observance=next_monday_or_tuesday),
    ]


def load_readings(csv_path):
    """Read the readings CSV into a DataFrame. Mirrors serve_01 cells 5-8.

    Parses value as numeric (coercing anything unparseable to NaN) and
    timestamp as a datetime, then drops rows missing either.
    """
    df = pd.read_csv(csv_path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df[df["timestamp"].notnull() & df["value"].notnull()]
    return df


def prepare_series(df):
    """Floor/sort/dedup/pad readings into a 1-minute-frequency series.

    Mirrors serve_01 cells 9-12. Must stay consistent with train_02 per
    PROJECT.md's ordering invariant: sort_index(kind='stable') before
    dedup (same-minute rows are exact ties; an unstable sort could
    reorder them), dedup keep='last' (latest reading within a minute
    wins), and asfreq('min', method='pad') only after both are done -
    padding a descending or duplicate-containing index silently leaks
    future values into gap rows (verified on pandas 2.3.3 and 3.0.3).
    """
    df = df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(LOCAL_TZ)
    df["timestamp"] = df["timestamp"].dt.floor(freq="min")
    df = df.set_index("timestamp")
    df = df.sort_index(kind="stable")

    dupl = df.index.duplicated(keep="last")
    if dupl.any():
        log.info("Dropping %d duplicate-minute reading(s).", dupl.sum())
    df = df[~dupl]

    df = df.asfreq("min", method="pad")
    return df


def engineer_features(df):
    """Add lag and rolling-window features. Mirrors serve_01 cells 13-16.

    Operates on the full padded series, since lags and rolling windows
    need the preceding rows; the caller reduces to a single row later
    (build_feature_row).
    """
    df = df.copy()
    for lag in LAG_LIST:
        df[f"lag_{lag}"] = df["value"].shift(lag)
    df["lag_0"] = df["value"]

    for window in ROLLING_WINDOWS:
        df[f"roll{window}_mean"] = (
            df["value"].rolling(window=window, min_periods=1).mean()
        )
        df[f"roll{window}_std"] = (
            df["value"].rolling(window=window, min_periods=1).std()
        )

    df["minute"] = df["timestamp_local"].dt.minute
    df["hour"] = df["timestamp_local"].dt.hour
    df["dayofweek"] = df["timestamp_local"].dt.dayofweek
    return df


def add_calendar_features(df):
    """Add is_weekend/is_holiday/day_off/hr_sin/hr_cos. Mirrors serve_01 cells 17-20.

    Holiday lookup is computed over local_date (a date, not a
    timestamp) - PROJECT.md notes that using raw timestamps here
    excludes midnight-normalized holiday dates and made is_holiday
    always False for single-row frames.
    """
    df = df.copy()
    df["local_date"] = df["timestamp_local"].dt.date

    calendar = OntarioCalendar()
    holidays = calendar.holidays(
        start=df["local_date"].min(), end=df["local_date"].max()
    )

    df["is_weekend"] = df["dayofweek"].isin([5, 6])
    df["is_holiday"] = df["timestamp_local"].dt.date.isin(holidays.date)
    df["day_off"] = df["is_weekend"] | df["is_holiday"]

    hr_cont = df["hour"] + df["minute"] / 60
    df["hr_sin"] = np.sin(2 * np.pi * hr_cont / 24)
    df["hr_cos"] = np.cos(2 * np.pi * hr_cont / 24)
    return df


def build_feature_row(df):
    """Reduce a fully-featured series to the single row to predict from.

    Mirrors serve_01 cells 16 and 21: take the latest row, compute the
    calendar features only for it (cheaper than for the whole series),
    then drop the columns that are not model inputs.
    """
    row = df.tail(1).copy()
    row = add_calendar_features(row)
    X = row.drop(columns=["timestamp_local", "local_date", "value"])
    return X


def build_features(csv_path):
    """Full feature pipeline: readings CSV -> single feature row."""
    df = load_readings(csv_path)
    df = prepare_series(df)
    df = engineer_features(df)
    return build_feature_row(df)


def load_model(model_path):
    """Load the {"model", "feature_names"} bundle saved by train_07."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_names"]


def predict(X, model, feature_names):
    """Subset X to feature_names and predict.

    feature_names must match, by name, what the model was trained on;
    order is enforced here by indexing with the list, so callers do
    not need X's columns to already be in the right order.
    """
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(
            f"Feature row is missing expected columns: {sorted(missing)}"
        )
    X = X[feature_names]
    prediction = model.predict(X)
    return float(prediction[0])


def write_prediction(value, based_on, horizon_minutes, output_path):
    """Write the predicted value to a JSON file.

    based_on is the timestamp of the input data the prediction was
    made from; predicted_for is based_on + horizon_minutes, matching
    what the model was trained to predict (README: "10 minutes
    ahead"). Posting this into Home Assistant is out of scope here -
    this just makes the prediction available on disk.
    """
    predicted_for = based_on + timedelta(minutes=horizon_minutes)
    payload = {
        "predicted_co2_ppm": value,
        "based_on": based_on.isoformat(),
        "predicted_for": predicted_for.isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


class PredictionError(Exception):
    """Raised when the prediction stage cannot produce a value.

    Covers a NaN feature row or a feature_names mismatch against the
    loaded model - distinct from FileNotFoundError, which covers a
    missing CSV or model file. Kept as one exception type rather than
    two so callers only need to catch one thing for "something about
    the prediction itself went wrong."
    """


def run_prediction(csv_path, model_path, output_path, horizon_minutes=HORIZON_MINUTES):
    """Build features, load the model, predict, and write the result.

    This is the single entry point other modules (predict_co2.py) call
    to get from a readings CSV to a written prediction; main() below is
    a thin CLI wrapper around it.

    Returns (value, based_on). Raises FileNotFoundError if csv_path or
    model_path does not exist, or PredictionError for a NaN feature row
    or a feature_names mismatch.
    """
    X = build_features(csv_path)  # may raise FileNotFoundError

    if X.isna().any().any():
        # NaN guard (PROJECT.md open issue 3): a history gap upstream
        # would otherwise yield NaN lag/rolling features, and HGBR
        # accepts NaN features and predicts on them silently.
        nan_cols = X.columns[X.isna().any()].tolist()
        raise PredictionError(f"Feature row contains NaN in columns: {nan_cols}")

    model, feature_names = load_model(model_path)  # may raise FileNotFoundError

    try:
        value = predict(X, model, feature_names)
    except ValueError as exc:
        raise PredictionError(str(exc)) from exc

    based_on = X.index[0]
    write_prediction(value, based_on, horizon_minutes, output_path)
    return value, based_on


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    try:
        value, based_on = run_prediction(CSV_INPUT_PATH, MODEL_PATH, PREDICTION_OUTPUT_PATH)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except PredictionError as exc:
        log.error("%s", exc)
        sys.exit(2)

    log.info(
        "Predicted %.2f ppm for %s (based on %s); wrote to %s",
        value, based_on + timedelta(minutes=HORIZON_MINUTES), based_on,
        PREDICTION_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()