"""
PowerSense Pro -- Supabase database layer.

Every public function in this module is defensive: a network failure, a missing
table or a bad credential never raises out of here. Reads return an empty list
or None, writes return False. The pipeline must keep running even when the
database is briefly unreachable (Render free tier cold starts, Supabase
maintenance windows, flaky site wifi at a demo).
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("powersense.db")

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional at runtime
    pass

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get(
    "SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_KEY", "")
)

# --- table names -----------------------------------------------------------
T_READINGS = "energy_readings"
T_DERIVED = "derived_values"
T_PREDICTIONS = "predictions"
T_ALERTS = "alerts"
T_SUMMARIES = "daily_summaries"

# Columns that actually exist on derived_values. The engine returns extra
# internal keys (prefixed with "_") used by the fault classifier; those are
# stripped before the insert so Supabase never rejects the row.
DERIVED_COLUMNS = (
    "device_id",
    "timestamp",
    "machine_name",
    "voltage",
    "current",
    "active_power",
    "active_energy",
    "frequency",
    "power_factor",
    "apparent_power",
    "reactive_power",
    "pf_status",
    "cable_loss",
    "reactive_energy_est",
    "apparent_energy_est",
    "energy_gap",
    "cost_per_hour",
    "daily_cost_projected",
    "monthly_cost_projected",
    "idle_cost_per_hour",
    "pf_penalty_pct",
    "pf_incentive_pct",
    "pf_monthly_impact",
    "motor_efficiency_est",
    "load_factor",
    "thd_estimate",
    "thermal_stress_index",
    "voltage_deviation_pct",
    "voltage_status",
    "machine_state",
    "utilization_pct",
    "carbon_footprint_kg",
    "energy_intensity",
    "motor_health_score",
    "motor_health_label",
    "power_stability_index",
    "specific_energy",
    "alerts",
)

SUMMARY_COLUMNS = (
    "date",
    "device_id",
    "avg_power",
    "max_power",
    "min_power",
    "total_energy",
    "avg_pf",
    "min_pf",
    "avg_voltage",
    "avg_current",
    "avg_frequency",
    "idle_percent",
    "reading_count",
)

_client = None
_client_init_attempted = False


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
def get_client():
    """Return a cached Supabase client, or None if it cannot be created."""
    global _client, _client_init_attempted

    if _client is not None:
        return _client
    if _client_init_attempted:
        return None

    _client_init_attempted = True

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.error(
            "SUPABASE_URL / SUPABASE_SERVICE_KEY are not set -- database layer "
            "is running in offline mode (all reads return empty)."
        )
        return None

    try:
        from supabase import create_client

        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialised for %s", SUPABASE_URL)
        return _client
    except Exception as exc:
        logger.error("Could not initialise Supabase client: %s", exc)
        return None


def is_connected() -> bool:
    """True when a Supabase client could be built."""
    return get_client() is not None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> Any:
    """Make a value JSON/Postgres safe (NaN and inf are not valid JSON)."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    # numpy scalars arrive from pandas aggregations
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list)):
        try:
            return _clean(value.item())
        except Exception:
            return value
    return value


def _project(data: Dict[str, Any], columns) -> Dict[str, Any]:
    """Keep only real table columns, cleaning values on the way through."""
    return {k: _clean(v) for k, v in data.items() if k in columns}


# ---------------------------------------------------------------------------
# energy_readings
# ---------------------------------------------------------------------------
def get_latest_readings(device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Newest-first list of raw readings for a device. Empty list on failure."""
    client = get_client()
    if client is None:
        return []
    try:
        res = (
            client.table(T_READINGS)
            .select("*")
            .eq("device_id", device_id)
            .order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(res.data or [])
    except Exception as exc:
        logger.error("get_latest_readings(%s) failed: %s", device_id, exc)
        return []


def get_latest_reading(device_id: str) -> Optional[Dict[str, Any]]:
    """Single most recent raw reading, or None."""
    rows = get_latest_readings(device_id, limit=1)
    return rows[0] if rows else None


def get_readings_hours(device_id: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Oldest-first readings from the last N hours. Empty list on failure."""
    client = get_client()
    if client is None:
        return []
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=float(hours))).isoformat()
        res = (
            client.table(T_READINGS)
            .select("*")
            .eq("device_id", device_id)
            .gte("created_at", since)
            .order("created_at", desc=False)
            .limit(20000)
            .execute()
        )
        return list(res.data or [])
    except Exception as exc:
        logger.error("get_readings_hours(%s, %sh) failed: %s", device_id, hours, exc)
        return []


def get_readings_between(
    device_id: str, start_iso: str, end_iso: str
) -> List[Dict[str, Any]]:
    """Oldest-first readings in an explicit ISO timestamp window."""
    client = get_client()
    if client is None:
        return []
    try:
        res = (
            client.table(T_READINGS)
            .select("*")
            .eq("device_id", device_id)
            .gte("created_at", start_iso)
            .lt("created_at", end_iso)
            .order("created_at", desc=False)
            .limit(50000)
            .execute()
        )
        return list(res.data or [])
    except Exception as exc:
        logger.error("get_readings_between(%s) failed: %s", device_id, exc)
        return []


def get_devices() -> List[str]:
    """Distinct device_ids seen in energy_readings."""
    client = get_client()
    if client is None:
        return []
    try:
        res = (
            client.table(T_READINGS)
            .select("device_id")
            .order("created_at", desc=True)
            .limit(5000)
            .execute()
        )
        seen: List[str] = []
        for row in res.data or []:
            did = row.get("device_id")
            if did and did not in seen:
                seen.append(did)
        return seen
    except Exception as exc:
        logger.error("get_devices() failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# derived_values
# ---------------------------------------------------------------------------
def insert_derived_values(device_id: str, data_dict: Dict[str, Any]) -> bool:
    """Upsert one derived_values row. Returns True on success.

    Upsert, not insert, on the (device_id, timestamp) unique constraint: the
    engine job runs every 30s regardless of whether a new raw reading has
    actually arrived, so a sensor that stalls or posts slower than that
    (dead battery, lost wifi, a quiet overnight gap) makes the same
    reading's timestamp get reprocessed repeatedly -- a plain insert would
    hit a duplicate-key error on every single tick until new data shows up,
    which is normal IoT behaviour, not a failure worth logging as one.
    """
    client = get_client()
    if client is None:
        return False

    payload = _project(dict(data_dict), DERIVED_COLUMNS)
    payload["device_id"] = device_id
    payload.setdefault("timestamp", _utc_now_iso())

    # `alerts` is a text-ish summary column -- normalise lists to a string.
    if isinstance(payload.get("alerts"), (list, tuple)):
        payload["alerts"] = ", ".join(str(a) for a in payload["alerts"])

    try:
        client.table(T_DERIVED).upsert(payload, on_conflict="device_id,timestamp").execute()
        return True
    except Exception as exc:
        logger.error("insert_derived_values(%s) failed: %s", device_id, exc)
        return False


def get_latest_derived(device_id: str) -> Optional[Dict[str, Any]]:
    """Most recent derived_values row, or None."""
    client = get_client()
    if client is None:
        return None
    try:
        res = (
            client.table(T_DERIVED)
            .select("*")
            .eq("device_id", device_id)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as exc:
        logger.error("get_latest_derived(%s) failed: %s", device_id, exc)
        return None


def get_derived_hours(device_id: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Oldest-first derived_values rows from the last N hours."""
    client = get_client()
    if client is None:
        return []
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=float(hours))).isoformat()
        res = (
            client.table(T_DERIVED)
            .select("*")
            .eq("device_id", device_id)
            .gte("timestamp", since)
            .order("timestamp", desc=False)
            .limit(20000)
            .execute()
        )
        return list(res.data or [])
    except Exception as exc:
        logger.error("get_derived_hours(%s) failed: %s", device_id, exc)
        return []


def get_derived_history(device_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Oldest-first window of the most recent N derived_values rows."""
    client = get_client()
    if client is None:
        return []
    try:
        res = (
            client.table(T_DERIVED)
            .select("*")
            .eq("device_id", device_id)
            .order("timestamp", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        rows = list(res.data or [])
        rows.reverse()
        return rows
    except Exception as exc:
        logger.error("get_derived_history(%s) failed: %s", device_id, exc)
        return []


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
def insert_prediction(
    device_id: str,
    prediction_type: str,
    value: Optional[float] = None,
    confidence: Optional[float] = None,
    label: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """Insert one row into predictions. Returns True on success."""
    client = get_client()
    if client is None:
        return False

    payload = {
        "device_id": device_id,
        "prediction_type": prediction_type,
        "value": _clean(float(value)) if value is not None else None,
        "confidence": _clean(float(confidence)) if confidence is not None else None,
        "label": label,
        "details": _json_safe(details or {}),
    }
    try:
        client.table(T_PREDICTIONS).insert(payload).execute()
        return True
    except Exception as exc:
        logger.error(
            "insert_prediction(%s, %s) failed: %s", device_id, prediction_type, exc
        )
        return False


def get_predictions(
    device_id: str, prediction_type: Optional[str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """Newest-first predictions, optionally filtered by type."""
    client = get_client()
    if client is None:
        return []
    try:
        query = client.table(T_PREDICTIONS).select("*").eq("device_id", device_id)
        if prediction_type:
            query = query.eq("prediction_type", prediction_type)
        res = (
            query.order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(res.data or [])
    except Exception as exc:
        logger.error("get_predictions(%s) failed: %s", device_id, exc)
        return []


def get_latest_prediction(
    device_id: str, prediction_type: str
) -> Optional[Dict[str, Any]]:
    """Most recent prediction of one type, or None."""
    rows = get_predictions(device_id, prediction_type, limit=1)
    return rows[0] if rows else None


def get_latest_predictions_by_type(device_id: str) -> Dict[str, Any]:
    """Map of prediction_type -> most recent prediction row for that type."""
    out: Dict[str, Any] = {}
    for row in get_predictions(device_id, limit=200):
        ptype = row.get("prediction_type")
        if ptype and ptype not in out:
            out[ptype] = row
    return out


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------
def insert_alert(
    device_id: str,
    level: str,
    code: str,
    message: str,
    action: str = "",
) -> bool:
    """Insert one unresolved alert row. Returns True on success."""
    client = get_client()
    if client is None:
        return False

    payload = {
        "device_id": device_id,
        "level": level,
        "code": code,
        "message": message,
        "action": action,
        "resolved": False,
    }
    try:
        client.table(T_ALERTS).insert(payload).execute()
        return True
    except Exception as exc:
        logger.error("insert_alert(%s, %s) failed: %s", device_id, code, exc)
        return False


def get_alerts(
    device_id: str, resolved: Optional[bool] = False, limit: int = 100
) -> List[Dict[str, Any]]:
    """Newest-first alerts. Pass resolved=None for every alert."""
    client = get_client()
    if client is None:
        return []
    try:
        query = client.table(T_ALERTS).select("*").eq("device_id", device_id)
        if resolved is not None:
            query = query.eq("resolved", bool(resolved))
        res = (
            query.order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(res.data or [])
    except Exception as exc:
        logger.error("get_alerts(%s) failed: %s", device_id, exc)
        return []


def resolve_alert(alert_id: Any) -> bool:
    """Mark a single alert resolved."""
    client = get_client()
    if client is None:
        return False
    try:
        (
            client.table(T_ALERTS)
            .update({"resolved": True, "resolved_at": _utc_now_iso()})
            .eq("id", alert_id)
            .execute()
        )
        return True
    except Exception as exc:
        logger.error("resolve_alert(%s) failed: %s", alert_id, exc)
        return False


def resolve_alerts_by_code(device_id: str, code: str) -> bool:
    """Mark every open alert with a given code resolved (fault cleared)."""
    client = get_client()
    if client is None:
        return False
    try:
        (
            client.table(T_ALERTS)
            .update({"resolved": True, "resolved_at": _utc_now_iso()})
            .eq("device_id", device_id)
            .eq("code", code)
            .eq("resolved", False)
            .execute()
        )
        return True
    except Exception as exc:
        logger.error("resolve_alerts_by_code(%s, %s) failed: %s", device_id, code, exc)
        return False


def recent_alert_exists(device_id: str, code: str, within_minutes: int = 30) -> bool:
    """True when an alert with this code was already raised recently.

    Used for de-duplication so a stuck fault does not spam WhatsApp.
    """
    client = get_client()
    if client is None:
        return False
    try:
        since = (
            datetime.now(timezone.utc) - timedelta(minutes=float(within_minutes))
        ).isoformat()
        res = (
            client.table(T_ALERTS)
            .select("id")
            .eq("device_id", device_id)
            .eq("code", code)
            .gte("created_at", since)
            .limit(1)
            .execute()
        )
        return bool(res.data)
    except Exception as exc:
        logger.error("recent_alert_exists(%s, %s) failed: %s", device_id, code, exc)
        return False


# ---------------------------------------------------------------------------
# daily_summaries
# ---------------------------------------------------------------------------
def archive_daily_summary(
    device_id: str, date: str, summary_dict: Dict[str, Any]
) -> bool:
    """Upsert one daily_summaries row on the UNIQUE(date, device_id) key."""
    client = get_client()
    if client is None:
        return False

    payload = _project(dict(summary_dict), SUMMARY_COLUMNS)
    payload["device_id"] = device_id
    payload["date"] = date

    try:
        (
            client.table(T_SUMMARIES)
            .upsert(payload, on_conflict="date,device_id")
            .execute()
        )
        return True
    except Exception as exc:
        logger.error("archive_daily_summary(%s, %s) failed: %s", device_id, date, exc)
        return False


def get_daily_summaries(device_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Newest-first daily summaries."""
    client = get_client()
    if client is None:
        return []
    try:
        res = (
            client.table(T_SUMMARIES)
            .select("*")
            .eq("device_id", device_id)
            .order("date", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(res.data or [])
    except Exception as exc:
        logger.error("get_daily_summaries(%s) failed: %s", device_id, exc)
        return []


def get_daily_summary(device_id: str, date: str) -> Optional[Dict[str, Any]]:
    """One daily summary by date, or None."""
    client = get_client()
    if client is None:
        return None
    try:
        res = (
            client.table(T_SUMMARIES)
            .select("*")
            .eq("device_id", device_id)
            .eq("date", date)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as exc:
        logger.error("get_daily_summary(%s, %s) failed: %s", device_id, date, exc)
        return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / NaN / datetime values into JSONB-safe data."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return _json_safe(obj.tolist())
        except Exception:
            return str(obj)
    return str(obj)
