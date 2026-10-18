"""
PowerSense Pro -- offline PyTorch training script.

Trains the two optional PyTorch upgrades that models/anomaly.py and
models/forecast.py pick up automatically if present:

  - an Autoencoder (10 features) for anomaly detection, replacing the
    in-process IsolationForest
  - a BiLSTM (48h window -> 24h forecast) for energy forecasting, replacing
    the in-process LinearRegression / hourly-profile fallback

Pipeline: fetch_from_supabase() -> augment_data() -> derive_features() (runs
every row through engine.py's own calculate_derived_values, so the feature
set here is identical to what the live pipeline computes) -> train + save.

This never runs on Render -- torch is intentionally not in requirements.txt
(see the note there). Run it locally instead, in its own venv (see
requirements_training.txt and the commands at the bottom of this docstring).

Reads SUPABASE_URL / SUPABASE_KEY from a .env file in this directory (see
.env.example) -- the *anon* key for the read-only energy_readings_public
table, independent of the main app's SUPABASE_SERVICE_KEY.

Output (dropped into models/trained/ -- the same directory POST
/models/upload writes to and the live app reads from):
    anomaly_{device_id}.pt
    anomaly_scaler.pkl                 (shared across devices)
    anomaly_threshold_{device_id}.pkl
    forecast_{device_id}.pt
    forecast_scaler.pkl                (shared across devices)

Setup + run:
    python -m venv venv311
    venv311\\Scripts\\activate
    pip install -r requirements_training.txt
    copy .env.example .env   # then fill in SUPABASE_URL / SUPABASE_KEY
    python train_powersense_pytorch.py

Push the result to a deployed instance (e.g. Render):
    python upload_models.py --url https://your-app.onrender.com
"""

import argparse
import logging
import math
import os
from datetime import datetime, timezone
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from supabase import create_client
from torch.utils.data import DataLoader, TensorDataset

import engine  # repo-root module -- calculate_derived_values(), DEFAULT_CONFIG

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("powersense.train")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "energy_readings_public")

# Same default the live app uses (main.py's MODEL_UPLOAD_DIR, models/anomaly.py
# and models/forecast.py's MODEL_DIR) -- training writes straight into the
# directory the running app already reads from.
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
# 1. fetching
# ---------------------------------------------------------------------------
def fetch_from_supabase(table: str = SUPABASE_TABLE, page_size: int = 1000) -> pd.DataFrame:
    """All rows from `table`, oldest first, paginated past Supabase's
    default 1000-row response cap (harmless at 658 rows today, keeps this
    working once the sensor has logged a lot more).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY are not set. Copy .env.example to .env "
            "in this directory and fill in your project URL + anon key."
        )

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    rows: List[dict] = []
    start = 0
    while True:
        res = (
            client.table(table)
            .select("*")
            .order("created_at", desc=False)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    df = pd.DataFrame(rows)
    logger.info("Fetched %d real rows from Supabase table '%s'", len(df), table)
    return df


# ---------------------------------------------------------------------------
# 2. augmentation
# ---------------------------------------------------------------------------
def _apply_gaussian_noise(
    frame: pd.DataFrame, rng: np.random.Generator, power_bounds: "tuple[float, float]"
) -> pd.DataFrame:
    """Add sensor-realistic jitter to voltage/current/power_factor/active_power,
    in place on a copy, clamped back into physically plausible bounds."""
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
    if "active_power" in frame.columns:
        low, high = power_bounds
        frame["active_power"] = _clip(
            pd.to_numeric(frame["active_power"], errors="coerce").fillna(0.0) + rng.normal(0, 0.1, n),
            low, high,
        )
    return frame


def _gaussian_noise_copies(
    df: pd.DataFrame, rng: np.random.Generator, power_bounds: "tuple[float, float]", copies: int = 10
) -> pd.DataFrame:
    """`copies` noisy near-duplicates of every real row -- the bulk of the
    augmented set. Teaches the model the few-percent sensor jitter every
    reading already has, instead of letting it memorise exact training
    values.
    """
    if df.empty:
        return df.iloc[0:0]
    tiled = pd.concat([df] * copies, ignore_index=True)
    return _apply_gaussian_noise(tiled, rng, power_bounds)


def _time_interpolated_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One synthetic row linearly interpolated between each pair of
    consecutive real readings (sorted by created_at), simulating a higher
    sampling frequency than the sensor actually runs at.
    """
    if len(df) < 2 or "created_at" not in df.columns:
        return df.iloc[0:0]

    ordered = df.copy()
    ordered["__ts"] = pd.to_datetime(ordered["created_at"], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=["__ts"]).sort_values("__ts").reset_index(drop=True)
    if len(ordered) < 2:
        return df.iloc[0:0]

    skip_cols = {"__ts", "created_at", "device_id", "id"}
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
        mid["created_at"] = (a["__ts"] + (b["__ts"] - a["__ts"]) * frac).isoformat()
        rows.append(mid)

    out = pd.DataFrame(rows).drop(columns=["__ts"], errors="ignore")
    if "voltage" in out.columns:
        out["voltage"] = _clip(pd.to_numeric(out["voltage"], errors="coerce"), 200.0, 240.0)
    if "current" in out.columns:
        out["current"] = _clip(pd.to_numeric(out["current"], errors="coerce"), 0.0, 0.5)
    if "power_factor" in out.columns:
        out["power_factor"] = _clip(pd.to_numeric(out["power_factor"], errors="coerce"), 0.0, 1.0)
    return out


def augment_data(df: pd.DataFrame, target: int = 20000) -> pd.DataFrame:
    """Expand real sensor rows up to `target` rows via Gaussian noise
    injection (the bulk of the volume) and linear interpolation between
    real readings, so the networks trained below have enough samples
    without needing weeks of real logging first. Shuffles before returning.
    """
    real_rows = len(df)
    if real_rows == 0:
        logger.warning("augment_data: no real rows supplied, nothing to augment")
        return df

    if real_rows >= target:
        result = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Augmented {real_rows} real rows -> {len(result)} total rows")
        return result

    rng = np.random.default_rng(42)
    base = df.reset_index(drop=True)

    power_bounds = (0.0, 0.0)
    if "active_power" in base.columns:
        base_power = pd.to_numeric(base["active_power"], errors="coerce").dropna()
        if len(base_power):
            power_bounds = (max(0.0, float(base_power.min()) - 1.0), float(base_power.max()) + 1.0)

    pieces = [
        base,
        _gaussian_noise_copies(base, rng, power_bounds, copies=10),
        _time_interpolated_rows(base, rng),
    ]
    augmented = pd.concat([p for p in pieces if len(p)], ignore_index=True)

    if len(augmented) < target:
        shortfall = target - len(augmented)
        topup_base = base.sample(n=shortfall, replace=True, random_state=42).reset_index(drop=True)
        topup = _apply_gaussian_noise(topup_base, rng, power_bounds)
        augmented = pd.concat([augmented, topup], ignore_index=True)
    elif len(augmented) > target:
        augmented = augmented.sample(n=target, random_state=42)

    augmented = augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Augmented {real_rows} real rows -> {len(augmented)} total rows")
    return augmented


# ---------------------------------------------------------------------------
# 3. feature engineering -- reuses engine.py so this matches the live
#    pipeline's math exactly instead of re-deriving it independently.
# ---------------------------------------------------------------------------
def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every row through engine.calculate_derived_values() -- this
    table's columns already match engine.py's names (active_power,
    active_energy, ...); only created_at needs to be handed through as the
    timestamp engine.py expects.
    """
    records = []
    for row in df.to_dict("records"):
        reading = {
            "device_id": row.get("device_id"),
            "timestamp": row.get("created_at"),
            "voltage": row.get("voltage"),
            "current": row.get("current"),
            "active_power": row.get("active_power"),
            "active_energy": row.get("active_energy"),
            "frequency": row.get("frequency"),
            "power_factor": row.get("power_factor"),
        }
        records.append(engine.calculate_derived_values(reading, engine.DEFAULT_CONFIG))
    return pd.DataFrame(records)


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
# 4. training
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
    parser.add_argument("--table", default=SUPABASE_TABLE, help="Supabase table to read raw readings from.")
    parser.add_argument("--target", type=int, default=20000, help="Row count to augment up to.")
    args = parser.parse_args()

    raw_df = fetch_from_supabase(table=args.table)
    if raw_df.empty:
        logger.error("No rows returned from Supabase table '%s' -- aborting.", args.table)
        return

    augmented = augment_data(raw_df, target=args.target)
    derived = derive_features(augmented)

    devices = (
        sorted(d for d in derived["device_id"].dropna().unique().tolist())
        if "device_id" in derived.columns else []
    ) or ["default"]

    _ensure_dir(MODEL_DIR)

    # -- shared scalers, fit once across every device's data ---------------
    anomaly_scaler = StandardScaler()
    anomaly_scaler.fit(build_anomaly_matrix(derived))
    joblib.dump(anomaly_scaler, os.path.join(MODEL_DIR, "anomaly_scaler.pkl"))

    hourly_by_device: Dict[str, pd.DataFrame] = {}
    for device_id in devices:
        subset = derived[derived["device_id"] == device_id] if "device_id" in derived.columns else derived
        hourly_by_device[device_id] = build_hourly_series(subset)

    forecastable = [h for h in hourly_by_device.values() if len(h) >= MIN_FORECAST_HOURS]
    forecast_scaler = StandardScaler()
    if forecastable:
        forecast_scaler.fit(np.concatenate([h[FORECAST_FEATURES].to_numpy(dtype=np.float32) for h in forecastable]))
        joblib.dump(forecast_scaler, os.path.join(MODEL_DIR, "forecast_scaler.pkl"))
    else:
        logger.warning(
            "No device has %d+ distinct hours of data yet -- skipping forecaster training "
            "entirely (the anomaly autoencoder doesn't need a time span, so it still trains).",
            MIN_FORECAST_HOURS,
        )

    # -- per-device training -------------------------------------------------
    for device_id in devices:
        subset = derived[derived["device_id"] == device_id] if "device_id" in derived.columns else derived
        train_autoencoder(device_id, build_anomaly_matrix(subset), anomaly_scaler)

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
