"""
PowerSense Pro -- offline PyTorch training script.

Trains the two optional PyTorch upgrades that models/anomaly.py and
models/forecast.py will pick up automatically if present:

  - an Autoencoder (10 features) for anomaly detection, replacing the
    in-process IsolationForest
  - a BiLSTM (48h window -> 24h forecast) for energy forecasting, replacing
    the in-process LinearRegression / hourly-profile fallback

This never runs on Render -- torch is intentionally not in requirements.txt
(see the note there). Run it locally instead, in its own venv:

    pip install torch joblib scikit-learn pandas numpy python-dotenv
    python train_powersense_pytorch.py                  # all devices
    python train_powersense_pytorch.py --device my_dev   # one device

It needs the same SUPABASE_URL / SUPABASE_SERVICE_KEY as the main app (a
.env file in this directory is picked up automatically via db.py).

Output (dropped into models/trained/, picked up on the app's next restart):
    anomaly_{device_id}.pt
    anomaly_scaler.pkl                 (shared across devices)
    anomaly_threshold_{device_id}.pkl
    forecast_{device_id}.pt
    forecast_scaler.pkl                (shared across devices)
"""

import argparse
import logging
import math
import os
from datetime import datetime, timezone
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import db  # repo-root module -- Supabase reads

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("powersense.train")

MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained")
)

# Autoencoder feature set. Names here match the `ae_features` list
# models/anomaly.py stores in the .pt bundle; the last two are read back
# from differently-named derived_values columns via its TORCH_FEATURE_ALIASES.
ANOMALY_FEATURES = [
    "voltage", "current", "active_power", "power_factor",
    "apparent_power", "reactive_power", "load_factor", "thd_estimate",
    "thermal_stress", "voltage_deviation",
]
ANOMALY_SOURCE_COLUMNS = {
    "thermal_stress": "thermal_stress_index",
    "voltage_deviation": "voltage_deviation_pct",
}

FORECAST_FEATURES = ["active_power", "power_factor", "voltage"]
FORECAST_WINDOW = 48       # hours of history fed to the BiLSTM
FORECAST_HORIZON = 24      # hours predicted
MIN_FORECAST_HOURS = FORECAST_WINDOW + FORECAST_HORIZON

AE_DROPOUT = 0.2
AE_EPOCHS = 60
AE_BATCH_SIZE = 64
AE_LR = 1e-3
AE_THRESHOLD_PERCENTILE = 95

BILSTM_HIDDEN = 64
BILSTM_LAYERS = 2
BILSTM_DROPOUT = 0.3
BILSTM_EPOCHS = 80
BILSTM_BATCH_SIZE = 32
BILSTM_LR = 1e-3


# ---------------------------------------------------------------------------
# model architectures -- must match models/anomaly.py's _AutoEncoder and
# models/forecast.py's _BiLSTMForecaster exactly (state_dicts are swapped
# directly between this script and those classes).
# ---------------------------------------------------------------------------
class Autoencoder(nn.Module):
    """10 -> 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64 -> 10, BatchNorm+ReLU+Dropout."""

    def __init__(self, input_dim: int, dropout: float = AE_DROPOUT) -> None:
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


class BiLSTMForecaster(nn.Module):
    """48 hourly timesteps x 3 features -> next 24 hourly kWh values."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = BILSTM_HIDDEN,
        num_layers: int = BILSTM_LAYERS,
        output_size: int = FORECAST_HORIZON,
        dropout: float = BILSTM_DROPOUT,
    ) -> None:
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if (math.isnan(out) or math.isinf(out)) else out


def _clip(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.clip(lower=low, upper=high)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def fetch_real_data(device_id: str, limit: int = 50000) -> pd.DataFrame:
    """Oldest-first derived_values history for one device."""
    rows = db.get_derived_history(device_id, limit=limit)
    df = pd.DataFrame(rows)
    logger.info("Fetched %d real rows for %s from Supabase", len(df), device_id)
    return df


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------
def _apply_gaussian_noise(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add sensor-realistic jitter to voltage/current/power_factor, in place
    on a copy, clamped back into physically plausible bounds."""
    frame = frame.copy()
    n = len(frame)
    if "voltage" in frame.columns:
        frame["voltage"] = _clip(
            pd.to_numeric(frame["voltage"], errors="coerce").fillna(0.0) + rng.normal(0, 0.5, n),
            200.0, 240.0,
        )
    if "current" in frame.columns:
        frame["current"] = _clip(
            pd.to_numeric(frame["current"], errors="coerce").fillna(0.0) + rng.normal(0, 0.002, n),
            0.0, 0.5,
        )
    if "power_factor" in frame.columns:
        frame["power_factor"] = _clip(
            pd.to_numeric(frame["power_factor"], errors="coerce").fillna(0.0) + rng.normal(0, 0.02, n),
            0.0, 1.0,
        )
    return frame


def _gaussian_noise_copies(df: pd.DataFrame, rng: np.random.Generator, copies: int = 10) -> pd.DataFrame:
    """Method 1 (most important) -- `copies` noisy near-duplicates of every
    real row. This is the bulk of the augmented set: it teaches the model
    the few-percent sensor jitter every reading already has, instead of
    letting it memorise exact training values.
    """
    if df.empty:
        return df.iloc[0:0]
    tiled = pd.concat([df] * copies, ignore_index=True)
    return _apply_gaussian_noise(tiled, rng)


def _time_interpolated_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Method 2 -- one synthetic row linearly interpolated between each pair
    of consecutive real readings (sorted by timestamp), simulating a higher
    sampling frequency than the sensor actually runs at.
    """
    if len(df) < 2 or "timestamp" not in df.columns:
        return df.iloc[0:0]

    ordered = df.copy()
    ordered["__ts"] = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=["__ts"]).sort_values("__ts").reset_index(drop=True)
    if len(ordered) < 2:
        return df.iloc[0:0]

    skip_cols = {"__ts", "timestamp", "device_id", "machine_name", "pf_status",
                 "voltage_status", "machine_state", "motor_health_label", "alerts"}
    numeric_cols = [
        c for c in ordered.columns
        if c not in skip_cols and pd.to_numeric(ordered[c], errors="coerce").notna().any()
    ]

    rows = []
    for i in range(len(ordered) - 1):
        a, b = ordered.iloc[i], ordered.iloc[i + 1]
        frac = float(rng.uniform(0.3, 0.7))
        mid = a.copy()
        for col in numeric_cols:
            av, bv = _f(a.get(col)), _f(b.get(col))
            mid[col] = av + (bv - av) * frac
        mid["timestamp"] = (a["__ts"] + (b["__ts"] - a["__ts"]) * frac).isoformat()
        rows.append(mid)

    out = pd.DataFrame(rows).drop(columns=["__ts"], errors="ignore")
    if "voltage" in out.columns:
        out["voltage"] = _clip(pd.to_numeric(out["voltage"], errors="coerce"), 200.0, 240.0)
    if "current" in out.columns:
        out["current"] = _clip(pd.to_numeric(out["current"], errors="coerce"), 0.0, 0.5)
    if "power_factor" in out.columns:
        out["power_factor"] = _clip(pd.to_numeric(out["power_factor"], errors="coerce"), 0.0, 1.0)
    return out


def _speed_mode_cycles(df: pd.DataFrame, rng: np.random.Generator, cycles: int = 2) -> pd.DataFrame:
    """Method 3 -- duplicate the whole sequence `cycles` times, each scaled
    by a factor in [0.85, 1.15] on the load-bearing columns, simulating the
    same machine running at a different load / speed setting.
    """
    if df.empty:
        return df.iloc[0:0]

    load_cols = [c for c in ("current", "active_power", "apparent_power", "reactive_power", "cable_loss")
                 if c in df.columns]
    out_frames = []
    for _ in range(cycles):
        scale = float(rng.uniform(0.85, 1.15))
        copy = df.copy()
        for col in load_cols:
            copy[col] = pd.to_numeric(copy[col], errors="coerce").fillna(0.0) * scale
        if "current" in copy.columns:
            copy["current"] = _clip(copy["current"], 0.0, 0.5)
        out_frames.append(copy)
    return pd.concat(out_frames, ignore_index=True) if out_frames else df.iloc[0:0]


def augment_data(df: pd.DataFrame, target_rows: int = 20000) -> pd.DataFrame:
    """Expand real sensor rows up to `target_rows` so the networks trained
    below have enough samples without needing weeks of real logging first.

    Combines three methods -- Gaussian noise injection (the bulk of the
    volume), time interpolation between real readings, and scaled
    "speed mode" duplicate sequences -- then tops up and shuffles to land
    on an exact row count.
    """
    real_rows = len(df)
    if real_rows == 0:
        logger.warning("augment_data: no real rows supplied, nothing to augment")
        return df

    if real_rows >= target_rows:
        result = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Augmented {real_rows} real rows -> {len(result)} total rows")
        return result

    rng = np.random.default_rng(42)
    base = df.reset_index(drop=True)

    pieces = [
        base,
        _gaussian_noise_copies(base, rng, copies=10),
        _time_interpolated_rows(base, rng),
        _speed_mode_cycles(base, rng, cycles=2),
    ]
    augmented = pd.concat([p for p in pieces if len(p)], ignore_index=True)

    if len(augmented) < target_rows:
        shortfall = target_rows - len(augmented)
        topup_base = base.sample(n=shortfall, replace=True, random_state=42).reset_index(drop=True)
        topup = _apply_gaussian_noise(topup_base, rng)
        augmented = pd.concat([augmented, topup], ignore_index=True)
    elif len(augmented) > target_rows:
        augmented = augmented.sample(n=target_rows, random_state=42)

    augmented = augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Augmented {real_rows} real rows -> {len(augmented)} total rows")
    return augmented


# ---------------------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------------------
def build_anomaly_matrix(df: pd.DataFrame) -> np.ndarray:
    """ANOMALY_FEATURES columns, in order, as a float32 matrix."""
    frame = df.copy()
    source_cols = [ANOMALY_SOURCE_COLUMNS.get(name, name) for name in ANOMALY_FEATURES]
    for col in source_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    return (
        frame[source_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def build_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Resample FORECAST_FEATURES to hourly means, oldest first."""
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=FORECAST_FEATURES)
    frame = df.copy()
    frame["__ts"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["__ts"])
    for col in FORECAST_FEATURES:
        if col not in frame.columns:
            frame[col] = 0.0
    return (
        frame.set_index("__ts")[FORECAST_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .resample("1h")
        .mean()
        .dropna()
    )


def build_forecast_windows(hourly: pd.DataFrame, scaler: StandardScaler):
    """Sliding 48h-in / 24h-out windows over the hourly series."""
    values = scaler.transform(hourly[FORECAST_FEATURES].to_numpy(dtype=np.float32))
    kwh = hourly["active_power"].to_numpy(dtype=np.float32) / 1000.0

    total = FORECAST_WINDOW + FORECAST_HORIZON
    X, y = [], []
    for i in range(len(values) - total + 1):
        X.append(values[i:i + FORECAST_WINDOW])
        y.append(kwh[i + FORECAST_WINDOW:i + total])

    if not X:
        return (
            np.empty((0, FORECAST_WINDOW, len(FORECAST_FEATURES)), dtype=np.float32),
            np.empty((0, FORECAST_HORIZON), dtype=np.float32),
        )
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_autoencoder(device_id: str, matrix: np.ndarray, scaler: StandardScaler) -> None:
    if len(matrix) < 50:
        logger.warning("Skipping autoencoder for %s: only %d rows", device_id, len(matrix))
        return

    scaled = scaler.transform(matrix)
    tensor = torch.tensor(scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=AE_BATCH_SIZE, shuffle=True)

    net = Autoencoder(input_dim=matrix.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    net.train()
    for epoch in range(AE_EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = net(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        if epoch % 10 == 0 or epoch == AE_EPOCHS - 1:
            logger.info(
                "[%s] autoencoder epoch %d/%d loss=%.6f",
                device_id, epoch + 1, AE_EPOCHS, epoch_loss / len(tensor),
            )

    net.eval()
    with torch.no_grad():
        recon = net(tensor).numpy()
    errors = np.mean((scaled - recon) ** 2, axis=1)
    threshold = float(np.percentile(errors, AE_THRESHOLD_PERCENTILE))
    last_trained = datetime.now(timezone.utc).isoformat()

    _ensure_dir(MODEL_DIR)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "features": ANOMALY_FEATURES,
            "input_dim": matrix.shape[1],
            "threshold": threshold,
            "last_trained": last_trained,
        },
        os.path.join(MODEL_DIR, f"anomaly_{device_id}.pt"),
    )
    joblib.dump(threshold, os.path.join(MODEL_DIR, f"anomaly_threshold_{device_id}.pkl"))
    logger.info("Saved autoencoder for %s (threshold=%.6f)", device_id, threshold)


def train_forecaster(device_id: str, X: np.ndarray, y: np.ndarray) -> None:
    if len(X) < 10:
        logger.warning("Skipping forecaster for %s: only %d windows", device_id, len(X))
        return

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BILSTM_BATCH_SIZE, shuffle=True)

    net = BiLSTMForecaster(input_size=len(FORECAST_FEATURES))
    optimizer = torch.optim.Adam(net.parameters(), lr=BILSTM_LR)
    loss_fn = nn.MSELoss()

    net.train()
    for epoch in range(BILSTM_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if epoch % 10 == 0 or epoch == BILSTM_EPOCHS - 1:
            logger.info(
                "[%s] forecaster epoch %d/%d loss=%.6f",
                device_id, epoch + 1, BILSTM_EPOCHS, epoch_loss / len(X_t),
            )

    last_trained = datetime.now(timezone.utc).isoformat()
    _ensure_dir(MODEL_DIR)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_size": len(FORECAST_FEATURES),
            "hidden_size": BILSTM_HIDDEN,
            "num_layers": BILSTM_LAYERS,
            "output_size": FORECAST_HORIZON,
            "window": FORECAST_WINDOW,
            "last_trained": last_trained,
        },
        os.path.join(MODEL_DIR, f"forecast_{device_id}.pt"),
    )
    logger.info("Saved BiLSTM forecaster for %s (%d windows)", device_id, len(X))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline PyTorch training for PowerSense Pro")
    parser.add_argument(
        "--device", action="append", dest="devices",
        help="Device ID to train (repeatable). Default: every device seen in Supabase.",
    )
    parser.add_argument("--target-rows", type=int, default=20000, help="Rows to augment each device up to.")
    args = parser.parse_args()

    devices = args.devices or db.get_devices()
    if not devices:
        logger.error("No devices found in Supabase -- nothing to train.")
        return

    per_device_data: Dict[str, pd.DataFrame] = {}
    for device_id in devices:
        real_df = fetch_real_data(device_id)
        if real_df.empty:
            logger.warning("No data for %s -- skipping.", device_id)
            continue
        per_device_data[device_id] = augment_data(real_df, target_rows=args.target_rows)

    if not per_device_data:
        logger.error("No usable data for any device -- aborting.")
        return

    # -- shared scalers, fit once across every device's augmented data -----
    _ensure_dir(MODEL_DIR)

    anomaly_scaler = StandardScaler()
    anomaly_scaler.fit(np.concatenate([build_anomaly_matrix(df) for df in per_device_data.values()]))
    joblib.dump(anomaly_scaler, os.path.join(MODEL_DIR, "anomaly_scaler.pkl"))

    hourly_by_device = {dev: build_hourly_series(df) for dev, df in per_device_data.items()}
    forecastable_hourly = [h for h in hourly_by_device.values() if len(h) >= MIN_FORECAST_HOURS]

    forecast_scaler = StandardScaler()
    if forecastable_hourly:
        forecast_scaler.fit(np.concatenate([h[FORECAST_FEATURES].to_numpy(dtype=np.float32) for h in forecastable_hourly]))
        joblib.dump(forecast_scaler, os.path.join(MODEL_DIR, "forecast_scaler.pkl"))
    else:
        logger.warning("No device has %d+ distinct hours of data -- skipping forecaster training entirely.", MIN_FORECAST_HOURS)

    # -- per-device training -------------------------------------------------
    for device_id, aug_df in per_device_data.items():
        train_autoencoder(device_id, build_anomaly_matrix(aug_df), anomaly_scaler)

        hourly = hourly_by_device[device_id]
        if len(hourly) < MIN_FORECAST_HOURS:
            logger.warning(
                "Skipping forecaster for %s: only %d distinct hours (need %d)",
                device_id, len(hourly), MIN_FORECAST_HOURS,
            )
            continue
        X, y = build_forecast_windows(hourly, forecast_scaler)
        train_forecaster(device_id, X, y)

    logger.info("Training complete. Artifacts written to %s", MODEL_DIR)


if __name__ == "__main__":
    main()
