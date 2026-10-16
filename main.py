"""
PowerSense Pro -- FastAPI application.

Serves the React dashboard's API surface and hosts the background scheduler
that runs the derived-values engine and the ML jobs. CORS is wide open because
the dashboard is a static SPA on a different origin.

Run locally:   uvicorn main:app --reload
Run on Render: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import alerts as alerts_mod
import db
import scheduler as sched
from engine import calculate_derived_values
from models.fault import classify_fault

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("powersense.api")

DEVICE_ID = os.environ.get("DEVICE_ID", "hackathon_machine")
ENABLE_SCHEDULER = os.environ.get("ENABLE_SCHEDULER", "true").lower() not in (
    "false",
    "0",
    "no",
)
VERSION = "1.0"

VALID_PREDICTION_TYPES = ("anomaly", "fault", "forecast", "health")

# --- model upload (see POST /models/upload) ---
MODEL_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained")
ALLOWED_MODEL_EXTENSIONS = (".pt", ".pkl")
MAX_MODEL_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB -- generous for these small models
# Optional shared-secret guard. Unset (the default) means the endpoint is
# open -- fine for a local/hackathon demo, but set this in Render's
# environment once the URL is something other people could stumble onto:
# anyone who can reach POST /models/upload can overwrite the live models.
UPLOAD_API_KEY = os.environ.get("UPLOAD_API_KEY", "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------
class ReadingPayload(BaseModel):
    voltage: float = 0.0
    current: float = 0.0
    active_power: float = 0.0
    active_energy: float = 0.0
    frequency: float = 50.0
    power_factor: float = 0.0


class CalculateRequest(BaseModel):
    device_id: str = Field(default=DEVICE_ID)
    reading: ReadingPayload
    save: bool = Field(
        default=False, description="Also persist the result to derived_values"
    )


class TriggerRequest(BaseModel):
    device_id: str = Field(default=DEVICE_ID)


# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PowerSense Pro %s starting up (device=%s)", VERSION, DEVICE_ID)
    logger.info(
        "Supabase connected: %s | alert channels: %s",
        db.is_connected(),
        alerts_mod.channel_status(),
    )

    if ENABLE_SCHEDULER:
        try:
            # start_scheduler also trains any untrained model and runs the
            # engine once so the dashboard has data immediately.
            sched.start_scheduler(DEVICE_ID)
        except Exception as exc:
            logger.error("Scheduler failed to start: %s", exc)
    else:
        logger.info("ENABLE_SCHEDULER=false -- API only, no background jobs")
        sched.get_models(DEVICE_ID)

    yield

    logger.info("PowerSense Pro shutting down")
    sched.shutdown_scheduler()


app = FastAPI(
    title="PowerSense Pro",
    description=(
        "Industrial IoT energy monitoring pipeline: 22 derived electrical "
        "metrics, rule-based fault classification, anomaly detection, "
        "24-hour energy forecasting and motor RUL prediction."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Never leak a stack trace to the dashboard; always return clean JSON."""
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "PowerSense Pro",
        "version": VERSION,
        "status": "running",
        "device_id": DEVICE_ID,
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/devices",
            "/derived/{device_id}/latest",
            "/derived/{device_id}?hours=24",
            "/predictions/{device_id}",
            "/predictions/{device_id}/{type}",
            "/summary/{device_id}",
            "/alerts/{device_id}?resolved=false",
            "POST /calculate",
            "POST /trigger",
            "POST /models/upload",
            "/models/{device_id}/status",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": _now(),
        "version": VERSION,
        "database_connected": db.is_connected(),
        "scheduler_running": sched.get_scheduler() is not None,
        "jobs": sched.job_status(),
    }


@app.get("/devices")
def devices() -> Dict[str, Any]:
    found = db.get_devices()
    if DEVICE_ID not in found:
        found = found + [DEVICE_ID]
    return {"devices": found, "default": DEVICE_ID, "count": len(found)}


# ---------------------------------------------------------------------------
# derived values
# ---------------------------------------------------------------------------
@app.get("/derived/{device_id}/latest")
def derived_latest(device_id: str) -> Dict[str, Any]:
    row = db.get_latest_derived(device_id)
    if row:
        return row

    # Nothing stored yet -- compute live from the newest raw reading so the
    # dashboard is never blank on a fresh deploy.
    readings = db.get_latest_readings(device_id, limit=25)
    if not readings:
        raise HTTPException(
            status_code=404,
            detail=f"No derived values or readings found for device '{device_id}'",
        )
    ordered = list(reversed(readings))
    bundle = sched.get_models(device_id)
    computed = calculate_derived_values(ordered[-1], bundle.config, ordered)
    computed["_live_computed"] = True
    return computed


@app.get("/derived/{device_id}")
def derived_history(
    device_id: str,
    hours: int = Query(default=24, ge=1, le=720, description="Look-back window"),
) -> Dict[str, Any]:
    rows = db.get_derived_hours(device_id, hours=hours)
    return {
        "device_id": device_id,
        "hours": hours,
        "count": len(rows),
        "data": rows,
    }


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
@app.get("/predictions/{device_id}")
def predictions_all(device_id: str) -> Dict[str, Any]:
    latest = db.get_latest_predictions_by_type(device_id)
    return {
        "device_id": device_id,
        "timestamp": _now(),
        "types": list(latest.keys()),
        "predictions": latest,
    }


@app.get("/predictions/{device_id}/{prediction_type}")
def predictions_by_type(
    device_id: str,
    prediction_type: str,
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    if prediction_type not in VALID_PREDICTION_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown prediction type '{prediction_type}'. "
                f"Valid types: {', '.join(VALID_PREDICTION_TYPES)}"
            ),
        )
    rows = db.get_predictions(device_id, prediction_type, limit=limit)
    return {
        "device_id": device_id,
        "prediction_type": prediction_type,
        "count": len(rows),
        "predictions": rows,
    }


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
@app.get("/summary/{device_id}")
def summary(device_id: str) -> Dict[str, Any]:
    """One call that fills the whole dashboard."""
    derived = db.get_latest_derived(device_id)

    if derived is None:
        readings = db.get_latest_readings(device_id, limit=25)
        if readings:
            ordered = list(reversed(readings))
            bundle = sched.get_models(device_id)
            derived = calculate_derived_values(ordered[-1], bundle.config, ordered)
            derived["_live_computed"] = True

    predictions = db.get_latest_predictions_by_type(device_id)
    active_alerts = db.get_alerts(device_id, resolved=False, limit=50)
    bundle = sched.get_models(device_id)

    return {
        "device_id": device_id,
        "machine_name": bundle.config.machine_name,
        "timestamp": _now(),
        "has_data": derived is not None,
        "derived": derived,
        "predictions": predictions,
        "active_alerts": active_alerts,
        "alert_count": len(active_alerts),
        "critical_alert_count": sum(
            1 for a in active_alerts if str(a.get("level", "")).lower() == "critical"
        ),
        "health": bundle.health.get_status(device_id),
        "models": {
            "anomaly": bundle.anomaly.status(),
            "forecast": bundle.forecast.status(),
        },
        "config": bundle.config.as_dict(),
    }


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------
@app.get("/alerts/{device_id}")
def alerts_list(
    device_id: str,
    resolved: Optional[bool] = Query(
        default=False, description="false=open, true=resolved, omit via all=true for both"
    ),
    all: bool = Query(default=False, description="Return resolved and unresolved"),
    limit: int = Query(default=100, ge=1, le=500),
) -> Dict[str, Any]:
    rows = db.get_alerts(device_id, resolved=None if all else resolved, limit=limit)
    return {
        "device_id": device_id,
        "resolved_filter": None if all else resolved,
        "count": len(rows),
        "alerts": rows,
        "channels": alerts_mod.channel_status(),
    }


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
@app.post("/calculate")
def calculate(payload: CalculateRequest) -> Dict[str, Any]:
    """Compute derived values for an ad-hoc reading, without waiting for a job."""
    device_id = payload.device_id or DEVICE_ID
    bundle = sched.get_models(device_id)

    reading = payload.reading.model_dump() if hasattr(payload.reading, "model_dump") else payload.reading.dict()
    reading["device_id"] = device_id

    # Pull recent history so the stability index is meaningful rather than 1.0.
    history = list(reversed(db.get_latest_readings(device_id, limit=25)))
    history.append(reading)

    derived = calculate_derived_values(reading, bundle.config, history)

    anomaly: Dict[str, Any] = {"status": "not_trained"}
    if bundle.anomaly.is_trained():
        anomaly = bundle.anomaly.predict(derived)

    fault = classify_fault(
        reading, derived, float(anomaly.get("anomaly_score") or 0.0)
    )

    saved = False
    if payload.save:
        saved = db.insert_derived_values(device_id, derived)

    return {
        "device_id": device_id,
        "timestamp": _now(),
        "saved": saved,
        "derived": derived,
        "anomaly": anomaly,
        "fault": fault,
    }


@app.post("/trigger")
def trigger(payload: TriggerRequest) -> Dict[str, Any]:
    """Run the full pipeline once, right now."""
    device_id = payload.device_id or DEVICE_ID
    result = sched.run_full_pipeline(device_id)
    if result.get("engine", {}).get("status") == "no_data":
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for device '{device_id}' -- nothing to process",
        )
    return result


@app.post("/retrain/{device_id}")
def retrain(device_id: str) -> Dict[str, Any]:
    """Force a refit of the anomaly and forecast models."""
    return sched.run_retrain(device_id)


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    device_id: str = Form(default=""),
    x_upload_key: Optional[str] = Header(default=None, alias="X-Upload-Key"),
) -> Dict[str, Any]:
    """Accept a trained model/scaler file and drop it into ./models/trained/.

    Matches the upload_models.py training-companion script: multipart field
    `file`, plus an informational `device_id` form field. Only .pt and .pkl
    are accepted. The saved filename is whatever the uploader sent (sanitised
    to a bare basename) -- it's expected to already follow the training
    script's naming, e.g. anomaly_hackathon_machine.pt or anomaly_scaler.pkl.

    After saving, every currently-loaded device's models are asked to reload
    so an upload takes effect immediately, without a redeploy.
    """
    if UPLOAD_API_KEY and x_upload_key != UPLOAD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Upload-Key header")

    safe_name = os.path.basename((file.filename or "").strip())
    ext = os.path.splitext(safe_name)[1].lower()
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_MODEL_EXTENSIONS)} files are accepted, got '{ext or '(none)'}'",
        )

    os.makedirs(MODEL_UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(MODEL_UPLOAD_DIR, safe_name)

    size = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_MODEL_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (limit {MAX_MODEL_UPLOAD_BYTES // (1024*1024)} MB)",
                    )
                out.write(chunk)
    except HTTPException:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    except Exception as exc:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")
    finally:
        await file.close()

    logger.info("Model uploaded: %s (%d bytes, device_id=%s)", safe_name, size, device_id or "unspecified")

    # Hot-reload: every device whose models are already in memory re-checks
    # disk for this file right now, so "upload -> better model active" works
    # without waiting for the next restart or nightly retrain.
    reload_results: Dict[str, Any] = {}
    for did in sched.registry_devices():
        bundle = sched.get_models(did)
        try:
            bundle.anomaly.load(bundle.anomaly.model_path)
            bundle.forecast.load(bundle.forecast.model_path)
            reload_results[did] = {
                "anomaly_backend": bundle.anomaly.status()["backend"],
                "forecast_backend": bundle.forecast.status()["backend"],
            }
        except Exception as exc:
            logger.warning("Post-upload reload failed for %s: %s", did, exc)
            reload_results[did] = {"error": str(exc)}

    return {
        "saved": safe_name,
        "path": dest_path,
        "size_bytes": size,
        "reloaded": reload_results,
    }


@app.post("/archive/{device_id}")
def archive(
    device_id: str,
    date: Optional[str] = Query(
        default=None, description="YYYY-MM-DD; defaults to yesterday"
    ),
) -> Dict[str, Any]:
    """Aggregate one day into daily_summaries on demand."""
    result = sched.run_daily_archive(device_id, target_date=date)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@app.get("/summaries/{device_id}")
def summaries(
    device_id: str, limit: int = Query(default=30, ge=1, le=365)
) -> Dict[str, Any]:
    rows = db.get_daily_summaries(device_id, limit=limit)
    return {"device_id": device_id, "count": len(rows), "summaries": rows}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
@app.get("/models/{device_id}/status")
def models_status(device_id: str) -> Dict[str, Any]:
    bundle = sched.get_models(device_id)
    anomaly = bundle.anomaly.status()
    forecast = bundle.forecast.status()
    health_points = bundle.health.history_points(device_id)

    last_trained = max(
        [t for t in (anomaly.get("last_trained"), forecast.get("last_trained")) if t],
        default=None,
    )

    return {
        "device_id": device_id,
        "anomaly_trained": bool(anomaly.get("trained")),
        "forecast_trained": bool(forecast.get("trained")),
        "health_data_points": health_points,
        "health_rul_ready": health_points >= 10,
        "last_trained": last_trained,
        "anomaly": anomaly,
        "forecast": forecast,
        "scheduler_running": sched.get_scheduler() is not None,
        "jobs": sched.job_status(),
    }


@app.get("/health/{device_id}")
def health_status(device_id: str) -> Dict[str, Any]:
    """Motor health score, trend and RUL projection."""
    bundle = sched.get_models(device_id)
    if bundle.health.history_points(device_id) == 0:
        bundle.health.seed_history(device_id, db.get_derived_history(device_id, limit=500))
    return bundle.health.get_status(device_id)


@app.get("/forecast/{device_id}")
def forecast_now(device_id: str) -> Dict[str, Any]:
    """Live 24-hour forecast (does not wait for the 30-minute job)."""
    bundle = sched.get_models(device_id)
    if not bundle.forecast.is_trained():
        sched.train_forecast(device_id)
    recent = list(reversed(db.get_latest_readings(device_id, limit=288)))
    return bundle.forecast.predict_24h(recent)
