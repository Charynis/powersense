"""
PowerSense Pro -- derived values engine.

Takes one raw PZEM-004T reading (voltage, current, active_power,
active_energy, frequency, power_factor) plus an optional history window and
produces the full set of 22+ derived electrical, cost, health and operational
metrics that the dashboard, the fault classifier and the ML models consume.

The engine is pure: no I/O, no globals, no exceptions for bad input. A reading
with missing or zero fields produces a complete dict with safe values, because
an industrial sensor that drops a field must not take the pipeline down.
"""

import math
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Working days per month used for the monthly cost projection.
WORKING_DAYS_PER_MONTH = 26

# Power below this (W) means the machine is electrically off, not idle.
OFF_POWER_THRESHOLD_W = 10.0

# Number of trailing power samples used for the stability index.
STABILITY_WINDOW = 20


@dataclass
class MachineConfig:
    """Nameplate and tariff configuration for one monitored machine."""

    device_id: str = "hackathon_machine"
    machine_name: str = "Demo Machine"
    rated_current: float = 10.0          # A
    rated_power: float = 2000.0          # W
    nominal_voltage: float = 230.0       # V
    cable_resistance: float = 0.5        # ohm
    tariff_rate: float = 9.0             # Rs / kWh
    idle_threshold_pct: float = 8.0      # % of rated power
    rated_pf: float = 0.85
    shift_hours: float = 8.0             # productive hours per day
    india_grid_factor: float = 0.82      # kg CO2 / kWh
    phase_type: str = "single_phase"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = MachineConfig(
    device_id="hackathon_machine",
    machine_name="Demo Machine",
    rated_current=10.0,
    rated_power=2000.0,
    nominal_voltage=230.0,
    cable_resistance=0.5,
    tariff_rate=9.0,
    idle_threshold_pct=8.0,
    rated_pf=0.85,
    shift_hours=8.0,
    india_grid_factor=0.82,
    phase_type="single_phase",
)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _f(value: Any, default: float = 0.0) -> float:
    """Coerce anything to a finite float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 3) -> float:
    try:
        out = round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    # avoid -0.0 in JSON payloads
    return 0.0 if out == 0 else out


# ---------------------------------------------------------------------------
# individual metric blocks
# ---------------------------------------------------------------------------
def classify_pf_status(power_factor: float) -> str:
    """Power factor quality band."""
    if power_factor > 0.95:
        return "excellent"
    if power_factor > 0.85:
        return "good"
    if power_factor > 0.70:
        return "poor"
    return "critical"


def classify_voltage_status(deviation_pct: float) -> str:
    """Supply voltage band relative to nominal."""
    dev = abs(deviation_pct)
    if dev <= 5.0:
        return "normal"
    if dev <= 10.0:
        return "warning"
    return "critical"


def classify_machine_state(
    active_power: float, load_factor: float, idle_threshold_pct: float
) -> str:
    """Operational state derived from absolute power and load factor."""
    if active_power < OFF_POWER_THRESHOLD_W:
        return "off"
    if load_factor < idle_threshold_pct:
        return "idle"
    if load_factor < 30.0:
        return "light_load"
    if load_factor < 80.0:
        return "normal"
    if load_factor < 100.0:
        return "heavy_load"
    return "overload"


def classify_health_label(score: float) -> str:
    """Health score band."""
    if score > 85.0:
        return "optimal"
    if score > 70.0:
        return "good"
    if score > 50.0:
        return "moderate"
    if score > 30.0:
        return "poor"
    return "critical"


def _history_power_series(history: Optional[List[Dict[str, Any]]]) -> List[float]:
    """Extract the trailing active_power series from a history window."""
    if not history:
        return []
    series: List[float] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        value = row.get("active_power")
        if value is None:
            continue
        series.append(_f(value))
    return series[-STABILITY_WINDOW:]


def _history_series(history: Optional[List[Dict[str, Any]]], key: str, window: int):
    """Extract a trailing numeric series for an arbitrary key."""
    if not history:
        return []
    series: List[float] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if value is None:
            continue
        series.append(_f(value))
    return series[-window:]


def compute_power_stability_index(history: Optional[List[Dict[str, Any]]]) -> float:
    """1.0 = rock steady load, 0.0 = wildly fluctuating.

    Defined as 1 - coefficient of variation over the last 20 power readings.
    Returns 1.0 when there is no usable history, so a cold start never looks
    like a fault.
    """
    series = _history_power_series(history)
    if len(series) < 2:
        return 1.0

    mean = sum(series) / len(series)
    if mean <= 0:
        return 1.0
    try:
        std = statistics.pstdev(series)
    except statistics.StatisticsError:
        return 1.0
    return _clamp(1.0 - (std / mean), 0.0, 1.0)


def compute_motor_health_score(
    power_factor: float,
    thermal_stress_index: float,
    voltage_deviation_pct: float,
    power_stability_index: float,
    load_factor: float,
) -> Dict[str, float]:
    """Weighted composite health score plus its component breakdown.

    Weights: PF 0.30, thermal 0.25, voltage 0.20, stability 0.15, load 0.10.
    """
    # Power factor: full marks at 0.95+, zero at 0.
    pf_score = _clamp((power_factor / 0.95) * 100.0, 0.0, 100.0)

    # Thermal: full marks up to 70% of rated thermal stress, zero at 170%.
    thermal_score = _clamp(100.0 - max(0.0, thermal_stress_index - 70.0), 0.0, 100.0)

    # Voltage: full marks on nominal, zero at 10% deviation either way.
    voltage_score = _clamp(100.0 - abs(voltage_deviation_pct) * 10.0, 0.0, 100.0)

    # Stability: direct mapping of the stability index.
    stability_score = _clamp(power_stability_index * 100.0, 0.0, 100.0)

    # Load: a motor is happiest between 60% and 100% of rated load. Running
    # far under-loaded wastes magnetising current; running over-loaded cooks it.
    if load_factor > 100.0:
        load_score = _clamp(100.0 - (load_factor - 100.0) * 2.5, 0.0, 100.0)
    elif load_factor < 60.0:
        load_score = _clamp(100.0 - (60.0 - load_factor) * 0.8, 0.0, 100.0)
    else:
        load_score = 100.0

    total = (
        pf_score * 0.30
        + thermal_score * 0.25
        + voltage_score * 0.20
        + stability_score * 0.15
        + load_score * 0.10
    )

    return {
        "score": _clamp(total, 0.0, 100.0),
        "pf_score": pf_score,
        "thermal_score": thermal_score,
        "voltage_score": voltage_score,
        "stability_score": stability_score,
        "load_score": load_score,
    }


def _collect_inline_alerts(d: Dict[str, Any], config: MachineConfig) -> List[str]:
    """Short human-readable flags stored on the derived_values row itself.

    This is a quick dashboard badge list -- the real fault classification lives
    in models/fault.py.
    """
    flags: List[str] = []
    if d["voltage_status"] == "critical":
        flags.append("VOLTAGE_CRITICAL")
    elif d["voltage_status"] == "warning":
        flags.append("VOLTAGE_WARNING")
    if d["pf_status"] in ("poor", "critical") and d["machine_state"] not in (
        "off",
        "idle",
    ):
        flags.append("PF_LOW")
    if d["machine_state"] == "overload":
        flags.append("OVERLOAD")
    if d["thermal_stress_index"] > 110.0:
        flags.append("THERMAL_STRESS")
    if d["machine_state"] == "idle":
        flags.append("IDLE")
    if d["power_stability_index"] < 0.80 and d["machine_state"] not in ("off", "idle"):
        flags.append("UNSTABLE_LOAD")
    if d["motor_health_label"] in ("poor", "critical"):
        flags.append("HEALTH_" + d["motor_health_label"].upper())
    return flags


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def calculate_derived_values(
    reading: Dict[str, Any],
    config: MachineConfig = DEFAULT_CONFIG,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute every derived metric for one reading.

    Args:
        reading: raw row from energy_readings (or an equivalent dict).
        config:  machine nameplate / tariff configuration.
        history: optional oldest-first list of previous readings (raw or
                 derived). Used for the stability index and the internal
                 statistics the fault classifier needs.

    Returns:
        A dict containing the echoed raw reading, all derived values, the
        health score breakdown, and internal "_"-prefixed statistics. Keys
        prefixed with "_" are never written to Supabase (db.py filters them).
    """
    reading = reading if isinstance(reading, dict) else {}
    config = config or DEFAULT_CONFIG

    # --- raw inputs --------------------------------------------------------
    voltage = _f(reading.get("voltage"))
    current = _f(reading.get("current"))
    active_power = _f(reading.get("active_power"))
    active_energy = _f(reading.get("active_energy"))
    frequency = _f(reading.get("frequency"))
    power_factor = _clamp(_f(reading.get("power_factor")), 0.0, 1.0)

    timestamp = (
        reading.get("created_at")
        or reading.get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )

    # Guard against divide-by-zero on a misconfigured machine.
    rated_power = config.rated_power if config.rated_power > 0 else 1.0
    rated_current = config.rated_current if config.rated_current > 0 else 1.0
    nominal_voltage = config.nominal_voltage if config.nominal_voltage > 0 else 230.0

    # --- POWER -------------------------------------------------------------
    apparent_power = voltage * current  # VA

    # Reactive power from the power triangle. Clamp the radicand: a PZEM can
    # report an active power slightly above V*I because the three values are
    # sampled with independent rounding.
    radicand = max(0.0, apparent_power**2 - active_power**2)
    reactive_power = math.sqrt(radicand)  # VAR

    pf_status = classify_pf_status(power_factor)
    cable_loss = (current**2) * config.cable_resistance  # W

    # --- ENERGY ------------------------------------------------------------
    if power_factor > 0.0:
        phi = math.acos(_clamp(power_factor, 0.0, 1.0))
        reactive_energy_est = active_energy * math.tan(phi)  # kVARh
        apparent_energy_est = active_energy / power_factor  # kVAh
    else:
        reactive_energy_est = 0.0
        apparent_energy_est = 0.0
    energy_gap = apparent_energy_est - active_energy

    # --- COST (Rs) ---------------------------------------------------------
    power_kw = active_power / 1000.0
    cost_per_hour = power_kw * config.tariff_rate
    daily_cost_projected = cost_per_hour * config.shift_hours
    monthly_cost_projected = daily_cost_projected * WORKING_DAYS_PER_MONTH
    idle_cost_per_hour = (
        (rated_power * config.idle_threshold_pct / 100.0) / 1000.0
    ) * config.tariff_rate

    if power_factor < 0.75:
        pf_penalty_pct = 2.0
    elif power_factor < 0.85:
        pf_penalty_pct = 1.0
    else:
        pf_penalty_pct = 0.0

    if power_factor > 0.95:
        pf_incentive_pct = 1.0
    elif power_factor > 0.90:
        pf_incentive_pct = 0.5
    else:
        pf_incentive_pct = 0.0

    pf_monthly_impact = (
        monthly_cost_projected * (pf_penalty_pct - pf_incentive_pct) / 100.0
    )

    # --- HEALTH ------------------------------------------------------------
    motor_efficiency_est = (
        (active_power / apparent_power) * 100.0 if apparent_power > 0 else 0.0
    )
    motor_efficiency_est = _clamp(motor_efficiency_est, 0.0, 100.0)

    load_factor = (active_power / rated_power) * 100.0

    if power_factor > 0.0:
        thd_radicand = max(0.0, (1.0 / (power_factor**2)) - 1.0)
        thd_estimate = _clamp(math.sqrt(thd_radicand) * 100.0, 0.0, 100.0)
    else:
        thd_estimate = 0.0

    thermal_stress_index = ((current / rated_current) ** 2) * 100.0
    voltage_deviation_pct = ((voltage - nominal_voltage) / nominal_voltage) * 100.0
    voltage_status = classify_voltage_status(voltage_deviation_pct)

    # --- OPERATIONAL -------------------------------------------------------
    machine_state = classify_machine_state(
        active_power, load_factor, config.idle_threshold_pct
    )
    utilization_pct = _clamp((active_power / rated_power) * 100.0, 0.0, 100.0)
    carbon_footprint_kg = power_kw * config.india_grid_factor
    energy_intensity = active_power / 1000.0

    # --- COMPOSITE ---------------------------------------------------------
    power_stability_index = compute_power_stability_index(history)
    health = compute_motor_health_score(
        power_factor=power_factor,
        thermal_stress_index=thermal_stress_index,
        voltage_deviation_pct=voltage_deviation_pct,
        power_stability_index=power_stability_index,
        load_factor=load_factor,
    )
    motor_health_score = health["score"]
    motor_health_label = classify_health_label(motor_health_score)
    specific_energy = active_power / 1000.0

    # --- internal statistics for the fault classifier ----------------------
    power_series = _history_power_series(history)
    current_series = _history_series(history, "current", STABILITY_WINDOW)
    pf_series = _history_series(history, "power_factor", STABILITY_WINDOW)
    voltage_series = _history_series(history, "voltage", STABILITY_WINDOW)

    def _cv(series: List[float]) -> float:
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        if mean <= 0:
            return 0.0
        try:
            return statistics.pstdev(series) / mean
        except statistics.StatisticsError:
            return 0.0

    power_mean = sum(power_series) / len(power_series) if power_series else active_power
    pf_recent_avg = sum(pf_series) / len(pf_series) if pf_series else power_factor

    out: Dict[str, Any] = {
        # --- identity / echoed raw reading ---
        "device_id": reading.get("device_id") or config.device_id,
        "machine_name": config.machine_name,
        "timestamp": timestamp,
        "voltage": _round(voltage),
        "current": _round(current),
        "active_power": _round(active_power),
        "active_energy": _round(active_energy, 4),
        "frequency": _round(frequency),
        "power_factor": _round(power_factor),
        # --- power ---
        "apparent_power": _round(apparent_power),
        "reactive_power": _round(reactive_power),
        "pf_status": pf_status,
        "cable_loss": _round(cable_loss),
        # --- energy ---
        "reactive_energy_est": _round(reactive_energy_est, 4),
        "apparent_energy_est": _round(apparent_energy_est, 4),
        "energy_gap": _round(energy_gap, 4),
        # --- cost ---
        "cost_per_hour": _round(cost_per_hour, 2),
        "daily_cost_projected": _round(daily_cost_projected, 2),
        "monthly_cost_projected": _round(monthly_cost_projected, 2),
        "idle_cost_per_hour": _round(idle_cost_per_hour, 2),
        "pf_penalty_pct": _round(pf_penalty_pct, 2),
        "pf_incentive_pct": _round(pf_incentive_pct, 2),
        "pf_monthly_impact": _round(pf_monthly_impact, 2),
        # --- health ---
        "motor_efficiency_est": _round(motor_efficiency_est, 2),
        "load_factor": _round(load_factor, 2),
        "thd_estimate": _round(thd_estimate, 2),
        "thermal_stress_index": _round(thermal_stress_index, 2),
        "voltage_deviation_pct": _round(voltage_deviation_pct, 2),
        "voltage_status": voltage_status,
        # --- operational ---
        "machine_state": machine_state,
        "utilization_pct": _round(utilization_pct, 2),
        "carbon_footprint_kg": _round(carbon_footprint_kg, 4),
        "energy_intensity": _round(energy_intensity, 4),
        # --- composite ---
        "motor_health_score": _round(motor_health_score, 2),
        "motor_health_label": motor_health_label,
        "power_stability_index": _round(power_stability_index, 4),
        "specific_energy": _round(specific_energy, 4),
        # --- health score breakdown (also mirrored under "_health_breakdown") ---
        "health_breakdown": {
            "pf_score": _round(health["pf_score"], 2),
            "thermal_score": _round(health["thermal_score"], 2),
            "voltage_score": _round(health["voltage_score"], 2),
            "stability_score": _round(health["stability_score"], 2),
            "load_score": _round(health["load_score"], 2),
            "weights": {
                "pf": 0.30,
                "thermal": 0.25,
                "voltage": 0.20,
                "stability": 0.15,
                "load": 0.10,
            },
        },
        # --- internal stats (filtered out before the DB insert) ---
        "_history_len": len(power_series),
        "_power_mean": _round(power_mean, 3),
        "_power_cv": _round(_cv(power_series), 4),
        "_current_cv": _round(_cv(current_series), 4),
        "_voltage_cv": _round(_cv(voltage_series), 4),
        "_pf_recent_avg": _round(pf_recent_avg, 4),
        "_config": config.as_dict(),
    }

    out["alerts"] = ", ".join(_collect_inline_alerts(out, config))
    out["alert_flags"] = _collect_inline_alerts(out, config)
    return out


def summarize_readings(
    readings: List[Dict[str, Any]], config: MachineConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """Aggregate a list of raw readings into a daily_summaries payload."""
    config = config or DEFAULT_CONFIG
    rows = [r for r in (readings or []) if isinstance(r, dict)]
    if not rows:
        return {
            "avg_power": 0.0,
            "max_power": 0.0,
            "min_power": 0.0,
            "total_energy": 0.0,
            "avg_pf": 0.0,
            "min_pf": 0.0,
            "avg_voltage": 0.0,
            "avg_current": 0.0,
            "avg_frequency": 0.0,
            "idle_percent": 0.0,
            "reading_count": 0,
        }

    powers = [_f(r.get("active_power")) for r in rows]
    pfs = [_clamp(_f(r.get("power_factor")), 0.0, 1.0) for r in rows]
    voltages = [_f(r.get("voltage")) for r in rows]
    currents = [_f(r.get("current")) for r in rows]
    freqs = [_f(r.get("frequency")) for r in rows]
    energies = [_f(r.get("active_energy")) for r in rows]

    rated_power = config.rated_power if config.rated_power > 0 else 1.0
    idle_cutoff = rated_power * config.idle_threshold_pct / 100.0
    idle_count = sum(
        1 for p in powers if p < idle_cutoff
    )

    # active_energy is a monotonically increasing cumulative counter on the
    # PZEM, so the day's consumption is the span -- unless the counter was
    # reset mid-window, in which case fall back to the final value.
    energy_span = max(energies) - min(energies)
    total_energy = energy_span if energy_span >= 0 else max(energies)

    return {
        "avg_power": _round(sum(powers) / len(powers), 3),
        "max_power": _round(max(powers), 3),
        "min_power": _round(min(powers), 3),
        "total_energy": _round(total_energy, 4),
        "avg_pf": _round(sum(pfs) / len(pfs), 4),
        "min_pf": _round(min(pfs), 4),
        "avg_voltage": _round(sum(voltages) / len(voltages), 3),
        "avg_current": _round(sum(currents) / len(currents), 3),
        "avg_frequency": _round(sum(freqs) / len(freqs), 3),
        "idle_percent": _round(idle_count / len(rows) * 100.0, 2),
        "reading_count": len(rows),
    }
