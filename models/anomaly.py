"""
PowerSense Pro -- anomaly detection.

Primary model: sklearn IsolationForest over 15 engineered features. It trains
in under a second on a few hundred readings, needs no GPU and no warm-up, so
it works on Render's free CPU tier on the first deploy.

Optional upgrade path: if a PyTorch autoencoder has been trained offline and
dropped next to the scaler as `anomaly_<device>.pt`, it is loaded and used for
reconstruction-error scoring instead, with the IsolationForest kept as the
fallback. Torch is never a hard dependency -- the import is guarded.
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("powersense.anomaly")

try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    import pickle

    joblib = None  # type: ignore
    JOBLIB_AVAILABLE = False

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "trained"))

MIN_TRAIN_ROWS = 30
CONTAMINATION = 0.05

# Torch is optional; only used when a pre-trained .pt file is present.
try:  # pragma: no cover - torch is not installed on the free tier
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False


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

    class _AutoEncoder(nn.Module):
        """10 -> 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64 -> 10, BatchNorm+ReLU+Dropout.

        Must match train_powersense_pytorch.py's Autoencoder exactly -- this
        is loading that script's state_dict, not a model trained in-process.
        """

        def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))


# The offline training script engineers 10 features under slightly different
# names than engine.py's derived_values columns (it was written against the
# literal feature list the user specified, not engine.py's own naming). These
# two are the only ones that differ -- everything else in the training
# script's FEATURES list (voltage, current, active_power, power_factor,
# apparent_power, reactive_power, load_factor, thd_estimate) already matches
# a derived_values column of the same name.
TORCH_FEATURE_ALIASES = {
    "thermal_stress": "thermal_stress_index",
    "voltage_deviation": "voltage_deviation_pct",
}


class AnomalyDetector:
    """IsolationForest anomaly detector with an optional autoencoder upgrade."""

    FEATURES = [
        "voltage",
        "current",
        "active_power",
        "power_factor",
        "apparent_power",
        "reactive_power",
        "thermal_stress_index",
        "load_factor",
        "voltage_deviation_pct",
        "power_stability_index",
        "motor_efficiency_est",
        "thd_estimate",
        "cable_loss",
        "energy_intensity",
        "motor_health_score",
    ]

    def __init__(self, device_id: str, model_dir: str = MODEL_DIR) -> None:
        self.device_id = device_id
        self.model_dir = model_dir
        _ensure_dir(self.model_dir)

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: float = 0.0
        # Typical (median) training score and the distance from it to the
        # threshold. Raw IsolationForest scores are tightly packed, so a plain
        # score/threshold ratio puts normal operation at ~0.95 and makes every
        # healthy reading look borderline. Normalising against this spread
        # instead maps median -> 0.0 and threshold -> 1.0.
        self.train_median: float = 0.0
        self.train_scale: float = 1.0
        self.train_rows: int = 0
        self.last_trained: Optional[str] = None
        self.backend: str = "none"

        # autoencoder state (only populated when a .pt file has been uploaded).
        # It has its own scaler and feature list, distinct from self.scaler /
        # self.FEATURES above -- it was trained offline by
        # train_powersense_pytorch.py on a different (10-feature) set with a
        # couple of renamed columns, not this class's own 15-feature set.
        self.autoencoder = None
        self.ae_features: List[str] = []
        self.ae_scaler: Optional[StandardScaler] = None
        self.ae_threshold: float = 0.0
        self.ae_scale: float = 1.0
        self.ae_last_trained: Optional[str] = None

        self.model_path = os.path.join(self.model_dir, f"anomaly_{device_id}.pkl")
        self.scaler_path = os.path.join(self.model_dir, f"scaler_{device_id}.pkl")
        self.torch_path = os.path.join(self.model_dir, f"anomaly_{device_id}.pt")
        self.torch_scaler_path = os.path.join(self.model_dir, "anomaly_scaler.pkl")
        self.torch_threshold_path = os.path.join(self.model_dir, f"anomaly_threshold_{device_id}.pkl")

        self.load(self.model_path)

    # -- state ------------------------------------------------------------
    def is_trained(self) -> bool:
        return (self.autoencoder is not None) or (self.model is not None and self.scaler is not None)

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "trained": self.is_trained(),
            "backend": self.backend,
            "train_rows": self.train_rows,
            "threshold": round(self.threshold, 4),
            "train_median": round(self.train_median, 4),
            "train_scale": round(self.train_scale, 6),
            "last_trained": self.last_trained,
            "features": len(self.FEATURES),
            "autoencoder_loaded": self.autoencoder is not None,
            "autoencoder_features": len(self.ae_features),
            "torch_available": TORCH_AVAILABLE,
        }

    # -- feature extraction ------------------------------------------------
    def _vector(self, row: Dict[str, Any]) -> np.ndarray:
        return np.array(
            [_f(row.get(name)) for name in self.FEATURES], dtype=float
        ).reshape(1, -1)

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        frame = df.copy()
        for name in self.FEATURES:
            if name not in frame.columns:
                frame[name] = 0.0
        matrix = (
            frame[self.FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return matrix

    # -- training ----------------------------------------------------------
    def train(self, readings_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit the scaler + IsolationForest and derive the score threshold.

        `readings_df` should hold derived_values rows (it needs the engineered
        columns). Raw readings also work -- missing columns are zero-filled --
        but detection quality is much better with derived values.
        """
        if readings_df is None or len(readings_df) == 0:
            return {
                "status": "no_data",
                "trained": self.is_trained(),
                "rows": 0,
                "message": "No rows supplied for training.",
            }

        if isinstance(readings_df, list):
            readings_df = pd.DataFrame(readings_df)

        rows = len(readings_df)
        if rows < MIN_TRAIN_ROWS:
            return {
                "status": "insufficient_data",
                "trained": self.is_trained(),
                "rows": rows,
                "rows_needed": MIN_TRAIN_ROWS - rows,
                "message": (
                    f"Need at least {MIN_TRAIN_ROWS} rows to train; got {rows}."
                ),
            }

        try:
            matrix = self._matrix(readings_df)

            scaler = StandardScaler()
            scaled = scaler.fit_transform(matrix)

            model = IsolationForest(
                n_estimators=150,
                contamination=CONTAMINATION,
                max_samples="auto",
                random_state=42,
                n_jobs=1,
            )
            model.fit(scaled)

            # IsolationForest score_samples: higher = more normal. Convert to a
            # positive "anomaly energy" and set the threshold at the
            # contamination quantile of the training set.
            raw = -model.score_samples(scaled)
            threshold = float(np.quantile(raw, 1.0 - CONTAMINATION))
            median = float(np.median(raw))
            # Distance from "typical" to "flagged" -- the unit the normalised
            # score is expressed in. Guard against a degenerate (constant)
            # training set collapsing it to zero.
            scale = max(threshold - median, 1e-6)

            self.model = model
            self.scaler = scaler
            self.threshold = threshold
            self.train_median = median
            self.train_scale = scale
            self.train_rows = rows
            self.last_trained = datetime.now(timezone.utc).isoformat()
            # Refreshing the sklearn fallback must not demote an already-active
            # PyTorch autoencoder -- a nightly retrain should keep the better
            # uploaded model in charge, not silently fall back every night.
            if self.backend != "autoencoder":
                self.backend = "isolation_forest"

            self.save(self.model_path)
            # Pick up a .pt uploaded since the last load (e.g. via
            # POST /models/upload) without needing a restart.
            self._load_torch_autoencoder()

            flagged = int(np.sum(raw > threshold))
            return {
                "status": "trained",
                "trained": True,
                "backend": self.backend,
                "rows": rows,
                "features": len(self.FEATURES),
                "threshold": round(threshold, 4),
                "train_score_median": round(median, 4),
                "train_score_scale": round(scale, 6),
                "train_score_mean": round(float(np.mean(raw)), 4),
                "train_score_std": round(float(np.std(raw)), 4),
                "train_score_max": round(float(np.max(raw)), 4),
                "flagged_in_training": flagged,
                "flagged_pct": round(flagged / rows * 100.0, 2),
                "autoencoder_loaded": self.autoencoder is not None,
                "last_trained": self.last_trained,
            }
        except Exception as exc:
            logger.error("Anomaly training failed for %s: %s", self.device_id, exc)
            return {
                "status": "error",
                "trained": self.is_trained(),
                "rows": rows,
                "message": str(exc),
            }

    def _load_torch_autoencoder(self) -> bool:
        """Load a locally-trained PyTorch autoencoder, if one has been uploaded.

        Looks for exactly the three files train_powersense_pytorch.py
        produces: anomaly_{device_id}.pt, anomaly_scaler.pkl (shared across
        devices, not device-suffixed -- matches the training script), and
        anomaly_threshold_{device_id}.pkl. All three must be present; a .pt
        without its scaler can't be scored correctly, so it's refused rather
        than silently scoring on unscaled inputs.
        """
        if not TORCH_AVAILABLE or not os.path.exists(self.torch_path):
            return False
        if not os.path.exists(self.torch_scaler_path):
            logger.warning(
                "Found %s but anomaly_scaler.pkl is missing -- cannot use the "
                "PyTorch autoencoder without its scaler.",
                os.path.basename(self.torch_path),
            )
            return False

        try:  # pragma: no cover - requires torch + a trained .pt
            bundle = torch.load(self.torch_path, map_location="cpu")
            state = bundle.get("state_dict", bundle) if isinstance(bundle, dict) else bundle
            features = list(bundle.get("features") or self.FEATURES) if isinstance(bundle, dict) else list(self.FEATURES)
            input_dim = int(bundle.get("input_dim", len(features))) if isinstance(bundle, dict) else len(features)

            net = _AutoEncoder(input_dim=input_dim)
            net.load_state_dict(state)
            net.eval()

            scaler = _load(self.torch_scaler_path)

            if os.path.exists(self.torch_threshold_path):
                threshold = _f(_load(self.torch_threshold_path))
            else:
                threshold = _f(bundle.get("threshold")) if isinstance(bundle, dict) else 0.0

            if threshold <= 0.0:
                logger.warning(
                    "PyTorch autoencoder for %s has no usable threshold -- skipping.",
                    self.device_id,
                )
                return False

            self.autoencoder = net
            self.ae_features = features
            self.ae_scaler = scaler
            self.ae_threshold = threshold
            # No training batch on hand at load time to compute a proper
            # median-based spread (see train_scale above) -- half the
            # threshold is the same documented fallback used there.
            self.ae_scale = max(threshold * 0.5, 1e-9)
            self.ae_last_trained = bundle.get("last_trained") if isinstance(bundle, dict) else None

            self.backend = "autoencoder"
            logger.info(
                "Loaded PyTorch autoencoder for %s (%d features, threshold %.6f)",
                self.device_id,
                input_dim,
                threshold,
            )
            return True
        except Exception as exc:
            logger.warning("Autoencoder load failed for %s: %s", self.device_id, exc)
            self.autoencoder = None
            return False

    # -- inference ---------------------------------------------------------
    def predict(self, derived: Dict[str, Any]) -> Dict[str, Any]:
        """Score one derived-values row.

        Dispatches on backend BEFORE building a feature vector: the PyTorch
        autoencoder and the IsolationForest were fit on different feature
        sets of different dimensionality (10 vs 15), so there is no shared
        "vector" to build first and branch on afterwards -- each path builds
        and scales its own.
        """
        if not self.is_trained():
            return {
                "status": "not_trained",
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "threshold": 0.0,
                "confidence": 0.0,
                "label": "not_trained",
                "backend": "none",
                "message": "Anomaly model has not been trained yet.",
            }

        row = derived if isinstance(derived, dict) else {}
        try:
            if self.backend == "autoencoder" and self.autoencoder is not None and TORCH_AVAILABLE:
                return self._predict_torch(row)
            return self._predict_isolation_forest(row)
        except Exception as exc:
            logger.error("Anomaly predict failed for %s: %s", self.device_id, exc)
            return {
                "status": "error",
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "threshold": round(self.threshold, 6),
                "confidence": 0.0,
                "label": "error",
                "backend": self.backend,
                "message": str(exc),
            }

    def _predict_isolation_forest(self, row: Dict[str, Any]) -> Dict[str, Any]:
        vector = self._vector(row)
        scaled = self.scaler.transform(vector)
        raw_score = float(-self.model.score_samples(scaled)[0])
        return self._score_result(
            raw_score=raw_score,
            threshold=self.threshold,
            scale=self.train_scale or 1.0,
            backend="isolation_forest",
            contributors=self._top_contributors(scaled[0], self.FEATURES),
        )

    def _predict_torch(self, row: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - requires torch
        vector = self._torch_vector(row)
        scaled = self.ae_scaler.transform(vector)
        with torch.no_grad():
            tensor = torch.tensor(scaled, dtype=torch.float32)
            recon = self.autoencoder(tensor).numpy()
        raw_score = float(np.mean((scaled - recon) ** 2))
        return self._score_result(
            raw_score=raw_score,
            threshold=self.ae_threshold or 1.0,
            scale=self.ae_scale or 1.0,
            backend="autoencoder",
            contributors=self._top_contributors(scaled[0], self.ae_features),
        )

    def _torch_vector(self, row: Dict[str, Any]) -> np.ndarray:
        """Build the autoencoder's input vector, resolving the couple of
        feature names the training script spells differently than
        engine.py's derived_values columns (see TORCH_FEATURE_ALIASES)."""
        values = []
        for name in self.ae_features:
            v = row.get(name)
            if v is None:
                v = row.get(TORCH_FEATURE_ALIASES.get(name, name))
            values.append(_f(v))
        return np.array(values, dtype=float).reshape(1, -1)

    @staticmethod
    def _label_for_score(normalised: float) -> str:
        if normalised <= 0.7:
            return "normal"
        if normalised <= 1.0:
            return "borderline"
        if normalised <= 1.5:
            return "anomaly"
        if normalised <= 2.0:
            return "strong_anomaly"
        return "severe_anomaly"

    def _score_result(
        self, raw_score: float, threshold: float, scale: float, backend: str, contributors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Normalised score: 0.0 is a textbook-typical operating point, 1.0
        # sits exactly on the detection threshold, and each further 1.0 is
        # another threshold-width out. The fault classifier compares against
        # 2.0 for UNKNOWN_ANOMALY.
        normalised = max(0.0, 1.0 + (raw_score - threshold) / scale)
        is_anomaly = normalised > 1.0
        confidence = round(max(0.3, min(0.99, abs(normalised - 1.0))), 3)

        return {
            "status": "ok",
            "anomaly_score": round(normalised, 4),
            "raw_score": round(raw_score, 6),
            "is_anomaly": bool(is_anomaly),
            "threshold": round(threshold, 6),
            "raw_scale": round(scale, 6),
            "normalised_threshold": 1.0,
            "confidence": confidence,
            "label": self._label_for_score(normalised),
            "backend": backend,
            "top_contributors": contributors,
            "train_rows": self.train_rows,
            "last_trained": self.last_trained if backend == "isolation_forest" else self.ae_last_trained,
        }

    def _top_contributors(self, scaled_row: np.ndarray, feature_names: List[str], top_n: int = 3):
        """Features sitting furthest from their training mean (in sigmas)."""
        try:
            pairs = [
                (feature_names[i], float(abs(scaled_row[i])))
                for i in range(len(feature_names))
            ]
            pairs.sort(key=lambda p: p[1], reverse=True)
            return [
                {"feature": name, "z_score": round(z, 3)} for name, z in pairs[:top_n]
            ]
        except Exception:
            return []

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> bool:
        """Persist the IsolationForest and the scaler to disk."""
        try:
            _ensure_dir(os.path.dirname(path) or ".")
            _dump(
                {
                    "model": self.model,
                    "threshold": self.threshold,
                    "train_median": self.train_median,
                    "train_scale": self.train_scale,
                    "train_rows": self.train_rows,
                    "last_trained": self.last_trained,
                    "features": self.FEATURES,
                    "device_id": self.device_id,
                },
                path,
            )
            _dump(self.scaler, self.scaler_path)
            logger.info("Saved anomaly model for %s -> %s", self.device_id, path)
            return True
        except Exception as exc:
            logger.error("Could not save anomaly model for %s: %s", self.device_id, exc)
            return False

    def load(self, path: str) -> bool:
        """Load whichever backend is available for this device.

        Priority: a PyTorch autoencoder (anomaly_{device_id}.pt +
        anomaly_scaler.pkl [+ anomaly_threshold_{device_id}.pkl]) is tried
        first -- it was trained offline on real history and is strictly more
        capable than the in-process IsolationForest. Falls back to the
        sklearn IsolationForest bundle this class trains itself. Returns
        True if either backend loaded.
        """
        torch_ok = self._load_torch_autoencoder()
        sklearn_ok = self._load_isolation_forest(path)

        if torch_ok:
            self.backend = "autoencoder"
        elif sklearn_ok:
            self.backend = "isolation_forest"
        else:
            self.backend = "none"

        logger.info(
            "Anomaly model for %s: backend=%s (pytorch=%s, isolation_forest=%s)",
            self.device_id, self.backend, torch_ok, sklearn_ok,
        )
        return torch_ok or sklearn_ok

    def _load_isolation_forest(self, path: str) -> bool:
        """Restore the sklearn IsolationForest bundle. Returns False when absent."""
        try:
            if not (os.path.exists(path) and os.path.exists(self.scaler_path)):
                return False

            bundle = _load(path)
            self.model = bundle.get("model")
            self.threshold = _f(bundle.get("threshold"))
            self.train_median = _f(bundle.get("train_median"))
            # Models saved before the scale was tracked fall back to half the
            # threshold, which keeps the normalised score in a sane range.
            self.train_scale = _f(
                bundle.get("train_scale"), max(self.threshold * 0.5, 1e-6)
            ) or max(self.threshold * 0.5, 1e-6)
            self.train_rows = int(_f(bundle.get("train_rows")))
            self.last_trained = bundle.get("last_trained")
            self.scaler = _load(self.scaler_path)

            ok = self.model is not None and self.scaler is not None
            if ok:
                logger.info(
                    "Loaded IsolationForest for %s (%s rows, threshold %.4f)",
                    self.device_id, self.train_rows, self.threshold,
                )
            return ok
        except Exception as exc:
            logger.warning("Could not load IsolationForest for %s: %s", self.device_id, exc)
            self.model = None
            self.scaler = None
            return False
