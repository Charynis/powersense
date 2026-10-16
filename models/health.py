"""
PowerSense Pro -- motor health scoring and remaining useful life (RUL).

Health score reuses the engine's weighted composite (so the number on the
dashboard and the number in the trend are always the same formula). RUL is a
linear regression on the health score history: fit the degradation slope,
extrapolate to the end-of-life threshold, and convert to days.
"""

import logging
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("powersense.health")

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - sklearn is a hard requirement
    LinearRegression = None  # type: ignore
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn unavailable, RUL falls back to slope maths: %s", exc)

# Health score at which the motor is considered to need replacement/overhaul.
END_OF_LIFE_SCORE = 40.0

# Minimum history before a trend may be extrapolated.
MIN_POINTS_FOR_RUL = 10

# How much history to keep per device (6h job cadence -> ~250 days).
HISTORY_MAXLEN = 1000

# RUL is clipped into a believable range; a flat trend is "no degradation".
MAX_RUL_DAYS = 3650.0

# Order spare parts this many days before the predicted maintenance date.
PARTS_LEAD_TIME_DAYS = 14


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def classify_health_label(score: float) -> str:
    if score > 85.0:
        return "optimal"
    if score > 70.0:
        return "good"
    if score > 50.0:
        return "moderate"
    if score > 30.0:
        return "poor"
    return "critical"


class MotorHealthModel:
    """Health score, trend tracking and RUL prediction for many devices."""

    def __init__(self) -> None:
        # device_id -> deque of (epoch_seconds, health_score)
        self._history: Dict[str, Deque[Tuple[float, float]]] = {}
        self._last_rul: Dict[str, Dict[str, Any]] = {}

    # -- scoring ----------------------------------------------------------
    def compute_health_score(self, derived: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the 0-100 health score and its component breakdown.

        Uses the pre-computed score from the engine when present, and
        otherwise recomputes it from the raw derived metrics so this model can
        also be called on a bare reading.
        """
        derived = derived if isinstance(derived, dict) else {}

        power_factor = _clamp(_f(derived.get("power_factor")), 0.0, 1.0)
        thermal = _f(derived.get("thermal_stress_index"))
        voltage_dev = _f(derived.get("voltage_deviation_pct"))
        stability = _clamp(_f(derived.get("power_stability_index"), 1.0), 0.0, 1.0)
        load_factor = _f(derived.get("load_factor"))

        pf_score = _clamp((power_factor / 0.95) * 100.0, 0.0, 100.0)
        thermal_score = _clamp(100.0 - max(0.0, thermal - 70.0), 0.0, 100.0)
        voltage_score = _clamp(100.0 - abs(voltage_dev) * 10.0, 0.0, 100.0)
        stability_score = _clamp(stability * 100.0, 0.0, 100.0)
        if load_factor > 100.0:
            load_score = _clamp(100.0 - (load_factor - 100.0) * 2.5, 0.0, 100.0)
        elif load_factor < 60.0:
            load_score = _clamp(100.0 - (60.0 - load_factor) * 0.8, 0.0, 100.0)
        else:
            load_score = 100.0

        computed = (
            pf_score * 0.30
            + thermal_score * 0.25
            + voltage_score * 0.20
            + stability_score * 0.15
            + load_score * 0.10
        )

        engine_score = derived.get("motor_health_score")
        score = _clamp(
            _f(engine_score, computed) if engine_score is not None else computed,
            0.0,
            100.0,
        )

        weakest = min(
            [
                ("power_factor", pf_score),
                ("thermal", thermal_score),
                ("voltage", voltage_score),
                ("stability", stability_score),
                ("load", load_score),
            ],
            key=lambda pair: pair[1],
        )

        return {
            "health_score": round(score, 2),
            "health_label": classify_health_label(score),
            "components": {
                "pf_score": round(pf_score, 2),
                "thermal_score": round(thermal_score, 2),
                "voltage_score": round(voltage_score, 2),
                "stability_score": round(stability_score, 2),
                "load_score": round(load_score, 2),
            },
            "weights": {
                "pf": 0.30,
                "thermal": 0.25,
                "voltage": 0.20,
                "stability": 0.15,
                "load": 0.10,
            },
            "weakest_component": weakest[0],
            "weakest_score": round(weakest[1], 2),
        }

    # -- trend ------------------------------------------------------------
    def update_trend(
        self,
        device_id: str,
        health_score: float,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Append a health score to the device's in-memory trend history.

        Returns the number of points now held for that device.
        """
        score = _clamp(_f(health_score), 0.0, 100.0)
        when = timestamp or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)

        series = self._history.setdefault(device_id, deque(maxlen=HISTORY_MAXLEN))
        series.append((when.timestamp(), score))
        return len(series)

    def seed_history(self, device_id: str, rows: List[Dict[str, Any]]) -> int:
        """Load trend history from stored derived_values rows.

        Lets RUL work immediately after a Render restart (the in-memory trend
        would otherwise be empty on every cold start).
        """
        series: Deque[Tuple[float, float]] = deque(maxlen=HISTORY_MAXLEN)
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            score = row.get("motor_health_score")
            if score is None:
                continue
            ts = self._parse_ts(row.get("timestamp") or row.get("created_at"))
            series.append((ts, _clamp(_f(score), 0.0, 100.0)))

        if series:
            self._history[device_id] = series
        return len(series)

    def history_points(self, device_id: str) -> int:
        return len(self._history.get(device_id, ()))

    def get_history(self, device_id: str) -> List[Dict[str, Any]]:
        """Trend history as a JSON-friendly list."""
        return [
            {
                "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                "health_score": round(score, 2),
            }
            for ts, score in self._history.get(device_id, ())
        ]

    @staticmethod
    def _parse_ts(value: Any) -> float:
        if isinstance(value, datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        if isinstance(value, str):
            text = value.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                pass
        return datetime.now(timezone.utc).timestamp()

    # -- RUL --------------------------------------------------------------
    def compute_rul(self, device_id: str) -> Dict[str, Any]:
        """Fit the health trend and extrapolate remaining useful life."""
        series = list(self._history.get(device_id, ()))
        points = len(series)

        if points < MIN_POINTS_FOR_RUL:
            return {
                "status": "insufficient_data",
                "trend_direction": "unknown",
                "rul_days": None,
                "maintenance_date": None,
                "order_parts_by": None,
                "confidence": 0.0,
                "data_points": points,
                "points_needed": MIN_POINTS_FOR_RUL - points,
                "message": (
                    f"Need {MIN_POINTS_FOR_RUL} health samples to project a "
                    f"trend; have {points}."
                ),
            }

        t0 = series[0][0]
        # x in days since the first sample, y = health score
        x = np.array([(ts - t0) / 86400.0 for ts, _ in series], dtype=float)
        y = np.array([score for _, score in series], dtype=float)

        span_days = float(x[-1] - x[0])
        current_score = float(y[-1])

        slope, intercept, r2 = self._fit(x, y)

        # slope is health points per day; negative means degrading.
        if slope < -0.01:
            trend_direction = "degrading"
        elif slope > 0.01:
            trend_direction = "improving"
        else:
            trend_direction = "stable"

        now = datetime.now(timezone.utc)
        if trend_direction == "degrading":
            rul_days = (current_score - END_OF_LIFE_SCORE) / abs(slope)
            rul_days = float(_clamp(rul_days, 0.0, MAX_RUL_DAYS))
            maintenance_dt = now + timedelta(days=rul_days)
            order_dt = maintenance_dt - timedelta(days=PARTS_LEAD_TIME_DAYS)
            if order_dt < now:
                order_dt = now
            maintenance_date = maintenance_dt.date().isoformat()
            order_parts_by = order_dt.date().isoformat()
        else:
            rul_days = None
            maintenance_date = None
            order_parts_by = None

        # Confidence blends fit quality, sample count and observation span: a
        # tight fit over two weeks of data is trustworthy, ten points over ten
        # minutes is not.
        sample_factor = _clamp(points / 60.0, 0.0, 1.0)
        span_factor = _clamp(span_days / 7.0, 0.0, 1.0)
        confidence = _clamp(
            0.5 * _clamp(r2, 0.0, 1.0) + 0.25 * sample_factor + 0.25 * span_factor,
            0.0,
            0.99,
        )
        if trend_direction != "degrading":
            confidence = _clamp(confidence * 0.8, 0.0, 0.95)

        result = {
            "status": "ok",
            "trend_direction": trend_direction,
            "slope_per_day": round(slope, 4),
            "rul_days": None if rul_days is None else round(rul_days, 1),
            "maintenance_date": maintenance_date,
            "order_parts_by": order_parts_by,
            "confidence": round(confidence, 3),
            "data_points": points,
            "observation_span_days": round(span_days, 3),
            "current_health_score": round(current_score, 2),
            "current_health_label": classify_health_label(current_score),
            "projected_score_30d": round(
                _clamp(current_score + slope * 30.0, 0.0, 100.0), 2
            ),
            "end_of_life_score": END_OF_LIFE_SCORE,
            "r2": round(_clamp(r2, 0.0, 1.0), 4),
            "method": "sklearn.LinearRegression" if SKLEARN_AVAILABLE else "least_squares",
        }
        self._last_rul[device_id] = result
        return result

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Return (slope, intercept, r2) for y ~ x."""
        if np.allclose(x, x[0]):
            # All samples at the same instant -- no trend is extractable.
            return 0.0, float(np.mean(y)), 0.0

        if SKLEARN_AVAILABLE:
            try:
                model = LinearRegression()
                model.fit(x.reshape(-1, 1), y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
                r2 = float(model.score(x.reshape(-1, 1), y))
                return slope, intercept, r2
            except Exception as exc:
                logger.warning("LinearRegression fit failed, using numpy: %s", exc)

        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
        return float(slope), float(intercept), r2

    # -- combined status --------------------------------------------------
    def get_status(self, device_id: str) -> Dict[str, Any]:
        """Latest health score, RUL projection and a short trend window."""
        series = list(self._history.get(device_id, ()))
        current = series[-1][1] if series else None
        rul = self.compute_rul(device_id)

        recent = [
            {
                "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                "health_score": round(score, 2),
            }
            for ts, score in series[-30:]
        ]

        return {
            "device_id": device_id,
            "current_health_score": None if current is None else round(current, 2),
            "current_health_label": (
                None if current is None else classify_health_label(current)
            ),
            "data_points": len(series),
            "rul": rul,
            "trend_direction": rul.get("trend_direction", "unknown"),
            "recent_trend": recent,
        }
