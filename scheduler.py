"""
PowerSense Pro -- APScheduler orchestration.

Job cadence
-----------
  every 30 s   run_engine_job      derived values from the newest reading
  every 5 min  run_anomaly_job     IsolationForest scoring
               run_fault_job       rule-based classification + alerting
  every 30 min run_forecast_job    24 h energy forecast
  every 6 h    run_health_job      health trend + RUL
  daily 01:00  run_daily_archive   aggregate yesterday into daily_summaries
               run_daily_report    email the summary
               run_retrain         refit anomaly + forecast models

Model instances live in a per-device registry so the FastAPI endpoints and the
background jobs share exactly one copy of each model (the health trend is
in-memory, so two copies would disagree).
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

import alerts as alerts_mod
import db
from engine import DEFAULT_CONFIG, MachineConfig, calculate_derived_values, summarize_readings
from models.anomaly import AnomalyDetector
from models.fault import classify_fault
from models.forecast import EnergyForecaster
from models.health import MotorHealthModel

logger = logging.getLogger("powersense.scheduler")

DEVICE_ID = os.environ.get("DEVICE_ID", "hackathon_machine")
MACHINE_NAME = os.environ.get("MACHINE_NAME", "Demo Machine")

ENGINE_INTERVAL_SECONDS = 30
ANOMALY_INTERVAL_MINUTES = 5
FORECAST_INTERVAL_MINUTES = 30
HEALTH_INTERVAL_HOURS = 6
DAILY_HOUR = 1

# History windows
ENGINE_HISTORY = 25        # readings used for the stability index
ANOMALY_TRAIN_ROWS = 500
FORECAST_TRAIN_ROWS = 2000
HEALTH_HISTORY_ROWS = 500


def build_config(device_id: str = DEVICE_ID) -> MachineConfig:
    """Machine config with env-var overrides applied."""

    def _env(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    return MachineConfig(
        device_id=device_id,
        machine_name=MACHINE_NAME,
        rated_current=_env("RATED_CURRENT", DEFAULT_CONFIG.rated_current),
        rated_power=_env("RATED_POWER", DEFAULT_CONFIG.rated_power),
        nominal_voltage=_env("NOMINAL_VOLTAGE", DEFAULT_CONFIG.nominal_voltage),
        cable_resistance=_env("CABLE_RESISTANCE", DEFAULT_CONFIG.cable_resistance),
        tariff_rate=_env("TARIFF_RATE", DEFAULT_CONFIG.tariff_rate),
        idle_threshold_pct=_env("IDLE_THRESHOLD_PCT", DEFAULT_CONFIG.idle_threshold_pct),
        rated_pf=_env("RATED_PF", DEFAULT_CONFIG.rated_pf),
        shift_hours=_env("SHIFT_HOURS", DEFAULT_CONFIG.shift_hours),
        india_grid_factor=_env("INDIA_GRID_FACTOR", DEFAULT_CONFIG.india_grid_factor),
        phase_type=os.environ.get("PHASE_TYPE", DEFAULT_CONFIG.phase_type),
    )


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------
class DeviceModels:
    """One bundle of models + config per device."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.config = build_config(device_id)
        self.anomaly = AnomalyDetector(device_id)
        self.forecast = EnergyForecaster(device_id)
        self.health = MotorHealthModel()
        self.last_engine_run: Optional[str] = None
        self.last_error: Optional[str] = None


_registry: Dict[str, DeviceModels] = {}


def get_models(device_id: str = DEVICE_ID) -> DeviceModels:
    """Fetch (creating on first use) the model bundle for a device."""
    bundle = _registry.get(device_id)
    if bundle is None:
        bundle = DeviceModels(device_id)
        _registry[device_id] = bundle
        logger.info("Initialised model registry for %s", device_id)
    return bundle


def registry_devices() -> List[str]:
    return list(_registry.keys())


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
def run_engine_job(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Fetch the newest reading, compute derived values, store the row."""
    bundle = get_models(device_id)
    try:
        history = db.get_latest_readings(device_id, limit=ENGINE_HISTORY)
        if not history:
            logger.info("engine: no readings for %s yet", device_id)
            return {"status": "no_data", "device_id": device_id}

        # db returns newest-first; the engine wants oldest-first history.
        ordered = list(reversed(history))
        latest = ordered[-1]

        derived = calculate_derived_values(
            reading=latest, config=bundle.config, history=ordered
        )
        saved = db.insert_derived_values(device_id, derived)
        bundle.last_engine_run = datetime.now(timezone.utc).isoformat()

        # Feed the health trend on every engine pass so the RUL regression has
        # data even before the 6-hourly health job first runs.
        bundle.health.update_trend(device_id, derived.get("motor_health_score", 0.0))

        logger.info(
            "engine: %s state=%s power=%.1fW pf=%.2f health=%.1f saved=%s",
            device_id,
            derived.get("machine_state"),
            float(derived.get("active_power") or 0),
            float(derived.get("power_factor") or 0),
            float(derived.get("motor_health_score") or 0),
            saved,
        )
        return {
            "status": "ok",
            "device_id": device_id,
            "saved": saved,
            "derived": derived,
        }
    except Exception as exc:
        bundle.last_error = str(exc)
        logger.exception("engine job failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def run_anomaly_job(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Score the latest operating point for anomalousness."""
    bundle = get_models(device_id)
    try:
        if not bundle.anomaly.is_trained():
            trained = train_anomaly(device_id)
            if not bundle.anomaly.is_trained():
                return {
                    "status": "not_trained",
                    "device_id": device_id,
                    "training": trained,
                }

        latest = db.get_latest_derived(device_id)
        if not latest:
            # No stored derived row yet -- derive one on the fly.
            engine_result = run_engine_job(device_id)
            latest = engine_result.get("derived")
        if not latest:
            return {"status": "no_data", "device_id": device_id}

        prediction = bundle.anomaly.predict(latest)
        db.insert_prediction(
            device_id=device_id,
            prediction_type="anomaly",
            value=prediction.get("anomaly_score"),
            confidence=prediction.get("confidence"),
            label=prediction.get("label"),
            details=prediction,
        )
        logger.info(
            "anomaly: %s score=%.3f label=%s",
            device_id,
            float(prediction.get("anomaly_score") or 0),
            prediction.get("label"),
        )
        return {"status": "ok", "device_id": device_id, "prediction": prediction}
    except Exception as exc:
        bundle.last_error = str(exc)
        logger.exception("anomaly job failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def run_fault_job(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Classify faults on the latest derived row and raise alerts."""
    bundle = get_models(device_id)
    try:
        readings = db.get_latest_readings(device_id, limit=ENGINE_HISTORY)
        if not readings:
            return {"status": "no_data", "device_id": device_id}

        ordered = list(reversed(readings))
        latest_reading = ordered[-1]
        # Recompute derived values so the internal "_" statistics the fault
        # rules need (CVs, PF average) are present -- the DB row drops them.
        derived = calculate_derived_values(
            reading=latest_reading, config=bundle.config, history=ordered
        )

        anomaly_score = 0.0
        if bundle.anomaly.is_trained():
            anomaly_score = float(
                bundle.anomaly.predict(derived).get("anomaly_score") or 0.0
            )

        fault = classify_fault(latest_reading, derived, anomaly_score)

        db.insert_prediction(
            device_id=device_id,
            prediction_type="fault",
            value=float(len(fault.get("all_faults") or [])),
            confidence=fault.get("confidence"),
            label=fault.get("fault_type"),
            details=fault,
        )

        alert_result = alerts_mod.check_and_send_alerts(device_id, derived, fault)

        # Close open alerts for conditions that have cleared.
        active_codes = {
            f.get("fault_type") for f in (fault.get("all_faults") or [])
        }
        if fault.get("fault_type") == "NORMAL":
            for row in db.get_alerts(device_id, resolved=False, limit=50):
                code = row.get("code")
                if code and code not in active_codes:
                    db.resolve_alert(row.get("id"))

        logger.info(
            "fault: %s type=%s severity=%s alerted=%s",
            device_id,
            fault.get("fault_type"),
            fault.get("severity"),
            alert_result.get("whatsapp", {}).get("sent"),
        )
        return {
            "status": "ok",
            "device_id": device_id,
            "fault": fault,
            "alert": alert_result,
        }
    except Exception as exc:
        bundle.last_error = str(exc)
        logger.exception("fault job failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def run_forecast_job(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Produce and store the 24 h energy forecast."""
    bundle = get_models(device_id)
    try:
        # A loaded BiLSTM needs a real 48-hour window, not the last few
        # minutes: derived_values (written every 30s by run_engine_job) is
        # used instead of raw 5s-cadence energy_readings so 48+ hours fits
        # comfortably under db.py's 20k-row query cap (energy_readings at 5s
        # cadence would need ~35k rows for the same window and get silently
        # truncated short of "now"). Falls back to raw readings only for the
        # cold-start case where derived_values doesn't have anything yet.
        ordered = db.get_derived_hours(device_id, hours=52)  # already oldest-first
        if not ordered:
            recent = db.get_latest_readings(device_id, limit=288)
            if not recent:
                return {"status": "no_data", "device_id": device_id}
            ordered = list(reversed(recent))

        if not bundle.forecast.is_trained():
            train_forecast(device_id)
        if not bundle.forecast.is_trained():
            return {"status": "not_trained", "device_id": device_id}

        prediction = bundle.forecast.predict_24h(ordered)
        db.insert_prediction(
            device_id=device_id,
            prediction_type="forecast",
            value=prediction.get("total_kwh_forecast"),
            confidence=prediction.get("confidence"),
            label=f"peak_hour_{prediction.get('peak_hour')}",
            details=prediction,
        )
        logger.info(
            "forecast: %s total=%.2fkWh cost=Rs%.0f peak_hour=%s",
            device_id,
            float(prediction.get("total_kwh_forecast") or 0),
            float(prediction.get("estimated_cost") or 0),
            prediction.get("peak_hour"),
        )
        return {"status": "ok", "device_id": device_id, "prediction": prediction}
    except Exception as exc:
        bundle.last_error = str(exc)
        logger.exception("forecast job failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def run_health_job(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Refresh the health trend from storage and project RUL."""
    bundle = get_models(device_id)
    try:
        rows = db.get_derived_history(device_id, limit=HEALTH_HISTORY_ROWS)
        if rows:
            bundle.health.seed_history(device_id, rows)
        elif bundle.health.history_points(device_id) == 0:
            return {"status": "no_data", "device_id": device_id}

        status = bundle.health.get_status(device_id)
        rul = status.get("rul", {})

        db.insert_prediction(
            device_id=device_id,
            prediction_type="health",
            value=status.get("current_health_score"),
            confidence=rul.get("confidence"),
            label=status.get("current_health_label") or rul.get("trend_direction"),
            details=status,
        )
        logger.info(
            "health: %s score=%s trend=%s rul_days=%s",
            device_id,
            status.get("current_health_score"),
            rul.get("trend_direction"),
            rul.get("rul_days"),
        )
        return {"status": "ok", "device_id": device_id, "health": status}
    except Exception as exc:
        bundle.last_error = str(exc)
        logger.exception("health job failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def run_daily_archive(
    device_id: str = DEVICE_ID, target_date: Optional[str] = None
) -> Dict[str, Any]:
    """Aggregate a full day of readings into daily_summaries."""
    bundle = get_models(device_id)
    try:
        if target_date:
            day = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        else:
            day = datetime.now(timezone.utc) - timedelta(days=1)

        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        date_str = start.date().isoformat()

        rows = db.get_readings_between(device_id, start.isoformat(), end.isoformat())
        if not rows:
            logger.info("archive: no readings for %s on %s", device_id, date_str)
            return {"status": "no_data", "device_id": device_id, "date": date_str}

        summary = summarize_readings(rows, bundle.config)
        saved = db.archive_daily_summary(device_id, date_str, summary)

        logger.info(
            "archive: %s %s -> %.3f kWh over %s readings (saved=%s)",
            device_id,
            date_str,
            summary.get("total_energy", 0.0),
            summary.get("reading_count", 0),
            saved,
        )
        return {
            "status": "ok",
            "device_id": device_id,
            "date": date_str,
            "saved": saved,
            "summary": summary,
        }
    except Exception as exc:
        bundle.last_error = str(exc)
        logger.exception("daily archive failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def run_daily_report(
    device_id: str = DEVICE_ID, target_date: Optional[str] = None
) -> Dict[str, Any]:
    """Email yesterday's summary."""
    try:
        date_str = target_date or (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).date().isoformat()

        summary = db.get_daily_summary(device_id, date_str)
        if not summary:
            archive = run_daily_archive(device_id, date_str)
            summary = archive.get("summary")
        if not summary:
            return {"status": "no_data", "device_id": device_id, "date": date_str}

        result = alerts_mod.send_email_daily_report(device_id, date_str, summary)
        logger.info(
            "report: %s %s email_sent=%s", device_id, date_str, result.get("sent")
        )
        return {
            "status": "ok",
            "device_id": device_id,
            "date": date_str,
            "email": result,
        }
    except Exception as exc:
        logger.exception("daily report failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_anomaly(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Train the anomaly model on stored derived values (raw as fallback)."""
    bundle = get_models(device_id)
    try:
        rows = db.get_derived_history(device_id, limit=ANOMALY_TRAIN_ROWS)
        source = "derived_values"

        if len(rows) < 30:
            # Cold start: derive features from raw readings so the model can
            # train on day one instead of waiting for derived rows to pile up.
            raw = list(reversed(db.get_latest_readings(device_id, limit=ANOMALY_TRAIN_ROWS)))
            if len(raw) < 30:
                return {
                    "status": "insufficient_data",
                    "device_id": device_id,
                    "rows": len(raw),
                    "message": "Need at least 30 readings to train the anomaly model.",
                }
            rows = [
                calculate_derived_values(raw[i], bundle.config, raw[max(0, i - 20) : i + 1])
                for i in range(len(raw))
            ]
            source = "derived_on_the_fly"

        result = bundle.anomaly.train(pd.DataFrame(rows))
        result["source"] = source
        result["device_id"] = device_id
        logger.info("anomaly training (%s): %s", device_id, result.get("status"))
        return result
    except Exception as exc:
        logger.exception("anomaly training failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def train_forecast(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Train the forecaster on the raw reading history."""
    bundle = get_models(device_id)
    try:
        rows = list(reversed(db.get_latest_readings(device_id, limit=FORECAST_TRAIN_ROWS)))
        if not rows:
            return {
                "status": "no_data",
                "device_id": device_id,
                "message": "No readings available to train the forecaster.",
            }
        result = bundle.forecast.train(pd.DataFrame(rows))
        result["device_id"] = device_id
        logger.info("forecast training (%s): %s", device_id, result.get("status"))
        return result
    except Exception as exc:
        logger.exception("forecast training failed for %s: %s", device_id, exc)
        return {"status": "error", "device_id": device_id, "message": str(exc)}


def run_retrain(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Nightly refit of both learned models."""
    logger.info("retrain: starting for %s", device_id)
    return {
        "status": "ok",
        "device_id": device_id,
        "anomaly": train_anomaly(device_id),
        "forecast": train_forecast(device_id),
    }


def train_if_needed(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Train whichever models are not ready yet, and seed the health trend."""
    bundle = get_models(device_id)
    out: Dict[str, Any] = {"device_id": device_id}

    if bundle.anomaly.is_trained():
        out["anomaly"] = {"status": "already_trained", **bundle.anomaly.status()}
    else:
        out["anomaly"] = train_anomaly(device_id)

    if bundle.forecast.is_trained():
        out["forecast"] = {"status": "already_trained", **bundle.forecast.status()}
    else:
        out["forecast"] = train_forecast(device_id)

    seeded = bundle.health.seed_history(
        device_id, db.get_derived_history(device_id, limit=HEALTH_HISTORY_ROWS)
    )
    out["health"] = {"status": "ok", "seeded_points": seeded}
    return out


# ---------------------------------------------------------------------------
# full pipeline (used by POST /trigger)
# ---------------------------------------------------------------------------
def run_full_pipeline(device_id: str = DEVICE_ID) -> Dict[str, Any]:
    """Run every stage once, in order. Safe to call on demand."""
    logger.info("pipeline: manual full run for %s", device_id)
    engine = run_engine_job(device_id)
    anomaly = run_anomaly_job(device_id)
    fault = run_fault_job(device_id)
    forecast = run_forecast_job(device_id)
    health = run_health_job(device_id)
    return {
        "status": "ok",
        "device_id": device_id,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "anomaly": anomaly,
        "fault": fault,
        "forecast": forecast,
        "health": health,
    }


# ---------------------------------------------------------------------------
# scheduler lifecycle
# ---------------------------------------------------------------------------
_scheduler = None


def get_scheduler():
    return _scheduler


def _run_5min(device_id: str) -> None:
    """The 5-minute slot runs anomaly scoring and then fault classification."""
    run_anomaly_job(device_id)
    run_fault_job(device_id)


def _run_daily(device_id: str) -> None:
    """The 01:00 slot archives, reports, then retrains."""
    run_daily_archive(device_id)
    run_daily_report(device_id)
    run_retrain(device_id)


def start_scheduler(device_id: str = DEVICE_ID):
    """Create, configure and start the AsyncIOScheduler.

    Startup sequence: build the model registry, train anything untrained, run
    the engine once so the dashboard has fresh data immediately, then register
    the recurring jobs.
    """
    global _scheduler

    if _scheduler is not None:
        logger.info("Scheduler already running")
        return _scheduler

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    bundle = get_models(device_id)

    # Explicitly (re)load both models from ./models/trained/ before deciding
    # whether training is needed -- a PyTorch model uploaded via
    # POST /models/upload before this process started is picked up here.
    # __init__ already calls load() once when the registry entry is first
    # created, so this is normally a no-op re-check; it's kept explicit and
    # visible (rather than relying on that implicit construction-time call)
    # so startup logs always show which backend is active for each model.
    bundle.anomaly.load(bundle.anomaly.model_path)
    bundle.forecast.load(bundle.forecast.model_path)

    anomaly_status = bundle.anomaly.status()
    forecast_status = bundle.forecast.status()
    print(
        f"[startup] anomaly model  : backend={anomaly_status['backend']:<16} "
        f"trained={anomaly_status['trained']}"
    )
    print(
        f"[startup] forecast model : backend={forecast_status['backend']:<16} "
        f"trained={forecast_status['trained']}"
    )
    logger.info(
        "Startup model backends for %s: anomaly=%s forecast=%s",
        device_id, anomaly_status["backend"], forecast_status["backend"],
    )

    try:
        startup = train_if_needed(device_id)
        logger.info(
            "startup training: anomaly=%s forecast=%s",
            startup.get("anomaly", {}).get("status"),
            startup.get("forecast", {}).get("status"),
        )
    except Exception as exc:
        logger.error("startup training failed: %s", exc)

    try:
        run_engine_job(device_id)
    except Exception as exc:
        logger.error("startup engine run failed: %s", exc)

    scheduler = AsyncIOScheduler(
        timezone="UTC",
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 60},
    )

    scheduler.add_job(
        run_engine_job,
        IntervalTrigger(seconds=ENGINE_INTERVAL_SECONDS),
        args=[device_id],
        id="engine",
        name="Derived values engine (30s)",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_5min,
        IntervalTrigger(minutes=ANOMALY_INTERVAL_MINUTES),
        args=[device_id],
        id="anomaly_fault",
        name="Anomaly + fault detection (5min)",
        replace_existing=True,
    )
    scheduler.add_job(
        run_forecast_job,
        IntervalTrigger(minutes=FORECAST_INTERVAL_MINUTES),
        args=[device_id],
        id="forecast",
        name="Energy forecast (30min)",
        replace_existing=True,
    )
    scheduler.add_job(
        run_health_job,
        IntervalTrigger(hours=HEALTH_INTERVAL_HOURS),
        args=[device_id],
        id="health",
        name="Health + RUL (6h)",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_daily,
        CronTrigger(hour=DAILY_HOUR, minute=0),
        args=[device_id],
        id="daily",
        name="Daily archive + report + retrain (01:00 UTC)",
        replace_existing=True,
    )

    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Scheduler started for %s with %s jobs", device_id, len(scheduler.get_jobs())
    )
    return scheduler


def shutdown_scheduler() -> None:
    """Stop the scheduler without waiting for running jobs to finish."""
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
    except Exception as exc:
        logger.warning("Scheduler shutdown error: %s", exc)
    finally:
        _scheduler = None


def job_status() -> List[Dict[str, Any]]:
    """Registered jobs and their next fire times."""
    if _scheduler is None:
        return []
    out = []
    for job in _scheduler.get_jobs():
        next_run = getattr(job, "next_run_time", None)
        out.append(
            {
                "id": job.id,
                "name": job.name,
                "trigger": str(job.trigger),
                "next_run": next_run.isoformat() if next_run else None,
            }
        )
    return out


if __name__ == "__main__":
    # Standalone worker mode: run the scheduler without the API.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    async def _main() -> None:
        start_scheduler(DEVICE_ID)
        while True:
            await asyncio.sleep(3600)

    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, SystemExit):
        shutdown_scheduler()
