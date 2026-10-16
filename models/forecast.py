"""
PowerSense Pro -- 24 hour energy forecast.

sklearn LinearRegression on cyclical time features (hour-of-day sin/cos,
day-of-week sin/cos, weekend flag) plus a recent-load level feature. Trains in
milliseconds, needs no GPU and no 48-hour history: with only a couple of hours
of data it degrades gracefully into a mean-plus-profile forecast rather than
refusing to answer.
"""

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("powersense.forecast")

try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    import pickle

    joblib = None  # type: ignore
    JOBLIB_AVAILABLE = False

from sklearn.linear_model import LinearRegression

# Torch is optional; only used when a pre-trained .pt file is present (see
# models/anomaly.py for the same pattern -- kept consistent across both).
try:  # pragma: no cover - torch is not installed on the free tier
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "trained"))

TARIFF_RATE = float(os.environ.get("TARIFF_RATE", "9.0"))
INDIA_GRID_FACTOR = float(os.environ.get("INDIA_GRID_FACTOR", "0.82"))

MIN_TRAIN_ROWS = 20

# Below this observed span the data covers too little of the daily cycle for
# the cyclical time features to extrapolate -- a 30-minute training window fits
# a narrow slice of the sin/cos curve and explodes when projected 24 h out. In
# that regime the forecast falls back to the observed level.
MIN_SPAN_HOURS_FOR_TIME_MODEL = 6.0

# Hard ceiling on any hourly prediction, as a multiple of the highest load ever
# observed in training. A monitored machine cannot suddenly draw 1000x.
MAX_KW_HEADROOM = 1.5

FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "recent_level_kw",
]


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:
        logger.warning("Could not create model dir %s: %s", path, exc)


def _dump(obj: Any, path: str) -> None:
    if JOBLIB_AVAILABLE:
        joblib.dump(obj, path)
    else:  # pragma: no cover
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


def _load(path: str) -> Any:
    if JOBLIB_AVAILABLE:
        return joblib.load(path)
    with open(path, "rb") as fh:  # pragma: no cover
        return pickle.load(fh)


if TORCH_AVAILABLE:  # pragma: no cover

    class _BiLSTMForecaster(nn.Module):
        """48 hourly timesteps x 3 features -> next 24 hourly kWh values.

        Must match train_powersense_pytorch.py's BiLSTMForecaster exactly --
        this is loading that script's state_dict, not a model trained
        in-process.
        """

        def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2,
                     output_size: int = 24, dropout: float = 0.3) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_size),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.fc(last)


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class EnergyForecaster:
    """Hourly kWh forecaster for one device."""

    def __init__(self, device_id: str, model_dir: str = MODEL_DIR) -> None:
        self.device_id = device_id
        self.model_dir = model_dir
        _ensure_dir(self.model_dir)

        self.model: Optional[LinearRegression] = None
        self.hourly_profile: Dict[int, float] = {}   # hour -> mean kW
        self.global_mean_kw: float = 0.0
        self.train_rows: int = 0
        self.residual_std: float = 0.0
        self.r2: float = 0.0
        self.max_kw: float = 0.0          # highest load seen in training
        self.span_hours: float = 0.0      # observed time span of training data
        self.last_trained: Optional[str] = None

        # BiLSTM state (only populated when a .pt file has been uploaded).
        self.bilstm = None
        self.bilstm_window: int = 48
        self.bilstm_scaler: Optional[Any] = None
        self.bilstm_last_trained: Optional[str] = None

        self.model_path = os.path.join(self.model_dir, f"forecast_{device_id}.pkl")
        self.bilstm_path = os.path.join(self.model_dir, f"forecast_{device_id}.pt")
        self.bilstm_scaler_path = os.path.join(self.model_dir, "forecast_scaler.pkl")
        self.load(self.model_path)

    # -- state ------------------------------------------------------------
    def is_trained(self) -> bool:
        return self.bilstm is not None or self.model is not None or bool(self.hourly_profile)

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "trained": self.is_trained(),
            "train_rows": self.train_rows,
            "r2": round(self.r2, 4),
            "global_mean_kw": round(self.global_mean_kw, 4),
            "max_kw": round(self.max_kw, 4),
            "span_hours": round(self.span_hours, 2),
            "profile_hours": len(self.hourly_profile),
            "time_model_usable": self._time_model_usable(),
            "last_trained": self.last_trained,
            "backend": self._backend(),
            "bilstm_loaded": self.bilstm is not None,
            "torch_available": TORCH_AVAILABLE,
        }

    def _time_model_usable(self) -> bool:
        """True when the training window covers enough of the daily cycle."""
        return (
            self.model is not None
            and self.span_hours >= MIN_SPAN_HOURS_FOR_TIME_MODEL
        )

    def _backend(self) -> str:
        # Checked fresh each call (not a stored field) so a BiLSTM stays
        # reported as active regardless of what later sklearn retrains do --
        # train() never touches self.bilstm, so this can't get demoted.
        if self.bilstm is not None:
            return "bilstm"
        if self._time_model_usable():
            return "linear_regression"
        if self.hourly_profile:
            return "recent_level" if self.model is not None else "hourly_profile"
        return "none"

    # -- BiLSTM: hourly window construction ---------------------------------
    @staticmethod
    def _synthetic_hourly_tail(n: int, seed: int = 123) -> pd.DataFrame:
        """n hours of a plausible weekday-shift load profile, ending just
        before "now". Used only to pad a short live reading window up to the
        BiLSTM's required 48-hour input when the device hasn't been running
        long enough to have that much real history yet -- mirrors the shift
        pattern train_powersense_pytorch.py's own synthetic fallback uses, so
        a cold-started forecast isn't wildly inconsistent with how the model
        was trained to fall back during training.
        """
        rng = np.random.default_rng(seed)
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        timestamps = [end - timedelta(hours=n - 1 - i) for i in range(n)]

        records = []
        for ts in timestamps:
            is_weekend = ts.weekday() >= 5
            if not is_weekend and 8 <= ts.hour < 18:
                base_kw = 1.15
            elif not is_weekend and (6 <= ts.hour < 8 or 18 <= ts.hour < 20):
                base_kw = 0.5
            else:
                base_kw = 0.08
            kw = max(0.0, base_kw + rng.normal(0, 0.05))
            pf = float(np.clip(0.9 + rng.normal(0, 0.02), 0.5, 0.99))
            voltage = 230.0 + rng.normal(0, 2.0)
            records.append((ts, kw * 1000.0, pf, voltage))

        return pd.DataFrame(
            records, columns=["__ts", "active_power", "power_factor", "voltage"]
        ).set_index("__ts")

    def _build_hourly_window(self, recent_readings: List[Dict[str, Any]], window: int) -> "tuple[pd.DataFrame, int]":
        """Resample recent raw readings to hourly [power, pf, voltage] and
        return exactly `window` rows ending at the most recent hour,
        left-padded with the synthetic profile if real history is short.
        Returns (frame, real_points) so the caller can scale confidence down
        when most of the window is padding.
        """
        rows = [r for r in (recent_readings or []) if isinstance(r, dict)]
        if rows:
            frame = pd.DataFrame(rows)
            ts_col = "created_at" if "created_at" in frame.columns else (
                "timestamp" if "timestamp" in frame.columns else None
            )
            for col in ("active_power", "power_factor", "voltage"):
                if col not in frame.columns:
                    frame[col] = 0.0
            if ts_col is not None:
                frame["__ts"] = frame[ts_col].apply(_parse_ts)
                hourly = (
                    frame.set_index("__ts")[["active_power", "power_factor", "voltage"]]
                    .apply(pd.to_numeric, errors="coerce")
                    .resample("1h")
                    .mean()
                    .dropna()
                )
            else:
                hourly = pd.DataFrame(columns=["active_power", "power_factor", "voltage"])
        else:
            hourly = pd.DataFrame(columns=["active_power", "power_factor", "voltage"])

        real_points = len(hourly)
        if real_points < window:
            padding = self._synthetic_hourly_tail(window - real_points)
            hourly = pd.concat([padding, hourly]).sort_index()

        return hourly.iloc[-window:], real_points

    def _cap_kw(self) -> float:
        """Physically plausible ceiling for an hourly prediction."""
        basis = max(self.max_kw, self.global_mean_kw)
        if basis <= 0:
            return 0.0
        return basis * MAX_KW_HEADROOM

    # -- feature engineering ------------------------------------------------
    @staticmethod
    def _features(when: datetime, recent_level_kw: float) -> List[float]:
        hour = when.hour + when.minute / 60.0
        dow = when.weekday()
        return [
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * dow / 7.0),
            math.cos(2 * math.pi * dow / 7.0),
            1.0 if dow >= 5 else 0.0,
            recent_level_kw,
        ]

    def _frame(self, readings_df: pd.DataFrame) -> pd.DataFrame:
        """Normalise input rows into a timestamped kW series."""
        frame = readings_df.copy()

        ts_col = None
        for candidate in ("created_at", "timestamp"):
            if candidate in frame.columns:
                ts_col = candidate
                break
        if ts_col is None:
            frame["__ts"] = pd.Timestamp.now(tz="UTC")
        else:
            frame["__ts"] = frame[ts_col].apply(_parse_ts)

        if "active_power" not in frame.columns:
            frame["active_power"] = 0.0

        frame["__kw"] = (
            pd.to_numeric(frame["active_power"], errors="coerce").fillna(0.0) / 1000.0
        )
        frame = frame.sort_values("__ts").reset_index(drop=True)
        return frame[["__ts", "__kw"]]

    # -- training ----------------------------------------------------------
    def train(self, readings_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit the regression and the hourly fallback profile."""
        if isinstance(readings_df, list):
            readings_df = pd.DataFrame(readings_df)
        if readings_df is None or len(readings_df) == 0:
            return {
                "status": "no_data",
                "trained": self.is_trained(),
                "rows": 0,
                "message": "No rows supplied for training.",
            }

        try:
            frame = self._frame(readings_df)
            rows = len(frame)

            # Hourly mean profile -- always built; it is the fallback whenever
            # the regression has not seen a given hour of the day.
            frame["__hour"] = frame["__ts"].apply(lambda d: d.hour)
            profile = frame.groupby("__hour")["__kw"].mean().to_dict()
            self.hourly_profile = {int(h): float(v) for h, v in profile.items()}
            self.global_mean_kw = float(frame["__kw"].mean())
            self.max_kw = float(frame["__kw"].max())
            self.span_hours = float(
                (frame["__ts"].iloc[-1] - frame["__ts"].iloc[0]).total_seconds() / 3600.0
            )
            self.train_rows = rows
            self.last_trained = datetime.now(timezone.utc).isoformat()

            if rows < MIN_TRAIN_ROWS:
                self.model = None
                self.r2 = 0.0
                self.residual_std = float(frame["__kw"].std(ddof=0) or 0.0)
                self.save(self.model_path)
                return {
                    "status": "profile_only",
                    "trained": True,
                    "backend": "hourly_profile",
                    "rows": rows,
                    "rows_needed_for_regression": MIN_TRAIN_ROWS - rows,
                    "global_mean_kw": round(self.global_mean_kw, 4),
                    "profile_hours": len(self.hourly_profile),
                    "last_trained": self.last_trained,
                    "message": (
                        f"Only {rows} rows -- using the hourly mean profile until "
                        f"{MIN_TRAIN_ROWS} rows are available."
                    ),
                }

            # Rolling recent level (20 samples ~ 100 s at 5 s cadence) gives
            # the regression a notion of "where the load is right now".
            recent = frame["__kw"].rolling(window=20, min_periods=1).mean()

            X = np.array(
                [
                    self._features(frame["__ts"].iloc[i], float(recent.iloc[i]))
                    for i in range(rows)
                ],
                dtype=float,
            )
            y = frame["__kw"].to_numpy(dtype=float)

            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(X)

            self.model = model
            self.r2 = float(model.score(X, y))
            self.residual_std = float(np.std(y - pred))
            self.save(self.model_path)
            # Pick up a .pt uploaded since the last load (e.g. via
            # POST /models/upload) without needing a restart. Never clears
            # self.bilstm if one is already active -- see _backend()'s comment.
            self._load_bilstm()

            result = {
                "status": "trained",
                "trained": True,
                "backend": self._backend(),
                "rows": rows,
                "features": FEATURE_NAMES,
                "r2": round(self.r2, 4),
                "residual_std_kw": round(self.residual_std, 5),
                "global_mean_kw": round(self.global_mean_kw, 4),
                "max_kw": round(self.max_kw, 4),
                "profile_hours": len(self.hourly_profile),
                "span_hours": round(self.span_hours, 2),
                "time_model_usable": self._time_model_usable(),
                "last_trained": self.last_trained,
            }
            if not self._time_model_usable():
                result["message"] = (
                    f"Training data spans only {self.span_hours:.2f} h. The "
                    f"time-of-day model needs {MIN_SPAN_HOURS_FOR_TIME_MODEL:.0f} h "
                    "to extrapolate safely, so forecasts will hold the observed "
                    "load level instead of projecting a daily cycle."
                )
            return result
        except Exception as exc:
            logger.error("Forecast training failed for %s: %s", self.device_id, exc)
            return {
                "status": "error",
                "trained": self.is_trained(),
                "message": str(exc),
            }

    # -- inference ---------------------------------------------------------
    def predict_24h(self, recent_readings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Forecast the next 24 hourly kWh values.

        `recent_readings` anchors the forecast to the current load level; an
        empty list falls back to the trained global mean.
        """
        if not self.is_trained():
            return {
                "status": "not_trained",
                "hourly_forecast": [],
                "total_kwh_forecast": 0.0,
                "peak_hour": None,
                "estimated_cost": 0.0,
                "confidence": 0.0,
                "message": "Forecast model has not been trained yet.",
            }

        try:
            if self.bilstm is not None and TORCH_AVAILABLE:
                return self._predict_bilstm(recent_readings)
            return self._predict_sklearn(recent_readings)
        except Exception as exc:
            logger.error("Forecast predict failed for %s: %s", self.device_id, exc)
            return {
                "status": "error",
                "hourly_forecast": [],
                "total_kwh_forecast": 0.0,
                "peak_hour": None,
                "estimated_cost": 0.0,
                "confidence": 0.0,
                "message": str(exc),
            }

    def _predict_bilstm(self, recent_readings: List[Dict[str, Any]]) -> Dict[str, Any]:  # pragma: no cover - requires torch
        """BiLSTM forecast: resample recent readings to an hourly window
        (synthetic-padded if the device hasn't got 48 real hours yet -- see
        _build_hourly_window), scale with forecast_scaler.pkl, and run the
        network. Its output is already in kWh (the training script did not
        scale the target), so no inverse-transform is needed on the way out.
        """
        hourly, real_points = self._build_hourly_window(recent_readings, self.bilstm_window)

        scaled = self.bilstm_scaler.transform(
            hourly[["active_power", "power_factor", "voltage"]].to_numpy(dtype=np.float32)
        )
        with torch.no_grad():
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, window, 3)
            pred = self.bilstm(tensor).numpy()[0]  # (24,) kWh
        pred = np.clip(pred, 0.0, None)

        now = datetime.now(timezone.utc)
        start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        hourly_forecast = []
        for step in range(len(pred)):
            slot = start + timedelta(hours=step)
            kw = float(pred[step])
            hourly_forecast.append(
                {
                    "hour": slot.hour,
                    "timestamp": slot.isoformat(),
                    "kwh": round(kw, 4),
                    "kw": round(kw, 4),
                    "cost": round(kw * TARIFF_RATE, 2),
                }
            )

        total_kwh = float(np.sum(pred))
        peak = max(hourly_forecast, key=lambda h: h["kwh"])

        # No live fit-quality signal is available for a static loaded model;
        # confidence instead reflects how much of the input window was real
        # history vs synthetic padding (see _build_hourly_window).
        confidence = round(max(0.3, min(0.9, 0.9 * real_points / self.bilstm_window)), 3)

        return {
            "status": "ok",
            "backend": "bilstm",
            "real_hourly_points": real_points,
            "generated_at": now.isoformat(),
            "forecast_start": start.isoformat(),
            "hourly_forecast": hourly_forecast,
            "hourly_kwh": [round(float(v), 4) for v in pred],
            "total_kwh_forecast": round(total_kwh, 4),
            "peak_hour": peak["hour"],
            "peak_kwh": round(peak["kwh"], 4),
            "min_hour": min(hourly_forecast, key=lambda h: h["kwh"])["hour"],
            "estimated_cost": round(total_kwh * TARIFF_RATE, 2),
            "estimated_carbon_kg": round(total_kwh * INDIA_GRID_FACTOR, 3),
            "tariff_rate": TARIFF_RATE,
            "anchor_level_kw": round(float(hourly["active_power"].iloc[-1]) / 1000.0, 4),
            "confidence": confidence,
            "train_rows": self.train_rows,
            "last_trained": self.bilstm_last_trained,
        }

    def _predict_sklearn(self, recent_readings: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            rows = [r for r in (recent_readings or []) if isinstance(r, dict)]
            levels = [_f(r.get("active_power")) / 1000.0 for r in rows[-20:]]
            recent_level_kw = (
                sum(levels) / len(levels) if levels else self.global_mean_kw
            )

            now = datetime.now(timezone.utc)
            start = (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )

            use_time_model = self._time_model_usable()
            cap = self._cap_kw()

            hourly: List[Dict[str, Any]] = []
            for step in range(24):
                slot = start + timedelta(hours=step)

                if use_time_model:
                    features = np.array(
                        [self._features(slot, recent_level_kw)], dtype=float
                    )
                    kw = float(self.model.predict(features)[0])

                    # Blend in the learned hourly profile: a linear fit on
                    # sin/cos cannot express a sharp shift start, the profile can.
                    profile_kw = self.hourly_profile.get(slot.hour)
                    if profile_kw is not None:
                        kw = 0.65 * kw + 0.35 * float(profile_kw)
                else:
                    # Not enough of the daily cycle observed to extrapolate.
                    # Hold the profile value for hours we actually saw, and the
                    # current load level for the rest.
                    profile_kw = self.hourly_profile.get(slot.hour)
                    kw = (
                        0.5 * float(profile_kw) + 0.5 * recent_level_kw
                        if profile_kw is not None
                        else recent_level_kw
                    )

                # A monitored machine cannot draw more than it ever has by a
                # wide margin. Without this, extrapolating cyclical features
                # outside the observed window can produce absurd kWh totals.
                kw = max(0.0, kw)
                if cap > 0.0:
                    kw = min(kw, cap)
                hourly.append(
                    {
                        "hour": slot.hour,
                        "timestamp": slot.isoformat(),
                        "kwh": round(kw, 4),
                        "kw": round(kw, 4),
                        "cost": round(kw * TARIFF_RATE, 2),
                    }
                )

            values = [h["kwh"] for h in hourly]
            total_kwh = float(sum(values))
            peak = max(hourly, key=lambda h: h["kwh"])

            # Confidence from fit quality, sample count, observed span and
            # forecast spread. Span matters most: a tight fit over 30 minutes
            # says nothing about tomorrow.
            sample_factor = min(1.0, self.train_rows / 288.0)
            span_factor = min(1.0, self.span_hours / 24.0)
            fit_factor = max(0.0, min(1.0, self.r2)) if self.model is not None else 0.35
            spread_penalty = 0.0
            if total_kwh > 0:
                spread_penalty = min(
                    0.3, self.residual_std / max(1e-6, total_kwh / 24.0) * 0.1
                )
            confidence = max(
                0.2,
                min(
                    0.95,
                    0.30 * fit_factor
                    + 0.20 * sample_factor
                    + 0.40 * span_factor
                    + 0.10
                    - spread_penalty,
                ),
            )
            if not use_time_model:
                # Holding the current level is a guess, not a projection.
                confidence = min(confidence, 0.40)

            return {
                "status": "ok",
                "backend": self._backend(),
                "time_model_usable": use_time_model,
                "capped_at_kw": round(cap, 4),
                "train_span_hours": round(self.span_hours, 2),
                "generated_at": now.isoformat(),
                "forecast_start": start.isoformat(),
                "hourly_forecast": hourly,
                "hourly_kwh": [round(v, 4) for v in values],
                "total_kwh_forecast": round(total_kwh, 4),
                "peak_hour": peak["hour"],
                "peak_kwh": round(peak["kwh"], 4),
                "min_hour": min(hourly, key=lambda h: h["kwh"])["hour"],
                "estimated_cost": round(total_kwh * TARIFF_RATE, 2),
                "estimated_carbon_kg": round(total_kwh * INDIA_GRID_FACTOR, 3),
                "tariff_rate": TARIFF_RATE,
                "anchor_level_kw": round(recent_level_kw, 4),
                "confidence": round(confidence, 3),
                "train_rows": self.train_rows,
                "r2": round(self.r2, 4),
                "last_trained": self.last_trained,
            }
        except Exception as exc:
            logger.error("Forecast predict failed for %s: %s", self.device_id, exc)
            return {
                "status": "error",
                "hourly_forecast": [],
                "total_kwh_forecast": 0.0,
                "peak_hour": None,
                "estimated_cost": 0.0,
                "confidence": 0.0,
                "message": str(exc),
            }

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> bool:
        try:
            _ensure_dir(os.path.dirname(path) or ".")
            _dump(
                {
                    "model": self.model,
                    "hourly_profile": self.hourly_profile,
                    "global_mean_kw": self.global_mean_kw,
                    "max_kw": self.max_kw,
                    "span_hours": self.span_hours,
                    "train_rows": self.train_rows,
                    "residual_std": self.residual_std,
                    "r2": self.r2,
                    "last_trained": self.last_trained,
                    "device_id": self.device_id,
                    "features": FEATURE_NAMES,
                },
                path,
            )
            logger.info("Saved forecast model for %s -> %s", self.device_id, path)
            return True
        except Exception as exc:
            logger.error("Could not save forecast model for %s: %s", self.device_id, exc)
            return False

    def load(self, path: str) -> bool:
        """Load whichever backend is available for this device.

        Priority: a BiLSTM (forecast_{device_id}.pt + forecast_scaler.pkl)
        is tried first -- trained offline on real history, strictly more
        capable than the in-process LinearRegression/hourly-profile fallback.
        Falls back to the sklearn bundle this class trains itself. Returns
        True if either backend loaded.
        """
        bilstm_ok = self._load_bilstm()
        sklearn_ok = self._load_sklearn_bundle(path)

        logger.info(
            "Forecast model for %s: backend=%s (bilstm=%s, sklearn/profile=%s)",
            self.device_id, self._backend(), bilstm_ok, sklearn_ok,
        )
        return bilstm_ok or sklearn_ok

    def _load_bilstm(self) -> bool:
        """Load a locally-trained BiLSTM, if one has been uploaded.

        Looks for exactly the files train_powersense_pytorch.py produces:
        forecast_{device_id}.pt and forecast_scaler.pkl (both required; a
        .pt without its scaler can't be fed correctly-scaled input, so it's
        refused rather than guessing).
        """
        if not TORCH_AVAILABLE or not os.path.exists(self.bilstm_path):
            return False
        if not os.path.exists(self.bilstm_scaler_path):
            logger.warning(
                "Found %s but forecast_scaler.pkl is missing -- cannot use the "
                "BiLSTM without its scaler.", os.path.basename(self.bilstm_path),
            )
            return False

        try:  # pragma: no cover - requires torch + a trained .pt
            bundle = torch.load(self.bilstm_path, map_location="cpu")
            net = _BiLSTMForecaster(
                input_size=int(bundle.get("input_size", 3)),
                hidden_size=int(bundle.get("hidden_size", 64)),
                num_layers=int(bundle.get("num_layers", 2)),
                output_size=int(bundle.get("output_size", 24)),
            )
            net.load_state_dict(bundle["state_dict"])
            net.eval()

            self.bilstm = net
            self.bilstm_window = int(bundle.get("window", 48))
            self.bilstm_scaler = _load(self.bilstm_scaler_path)
            self.bilstm_last_trained = bundle.get("last_trained")

            logger.info(
                "Loaded BiLSTM forecaster for %s (window=%d)", self.device_id, self.bilstm_window
            )
            return True
        except Exception as exc:
            logger.warning("BiLSTM load failed for %s: %s", self.device_id, exc)
            self.bilstm = None
            return False

    def _load_sklearn_bundle(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            bundle = _load(path)
            self.model = bundle.get("model")
            self.hourly_profile = {
                int(k): float(v) for k, v in (bundle.get("hourly_profile") or {}).items()
            }
            self.global_mean_kw = _f(bundle.get("global_mean_kw"))
            self.max_kw = _f(bundle.get("max_kw"), self.global_mean_kw)
            self.span_hours = _f(bundle.get("span_hours"))
            self.train_rows = int(_f(bundle.get("train_rows")))
            self.residual_std = _f(bundle.get("residual_std"))
            self.r2 = _f(bundle.get("r2"))
            self.last_trained = bundle.get("last_trained")
            ok = self.model is not None or bool(self.hourly_profile)
            if ok:
                logger.info(
                    "Loaded sklearn forecast bundle for %s (%s rows, r2 %.3f)",
                    self.device_id, self.train_rows, self.r2,
                )
            return ok
        except Exception as exc:
            logger.warning(
                "Could not load sklearn forecast bundle for %s: %s", self.device_id, exc
            )
            self.model = None
            self.hourly_profile = {}
            return False
