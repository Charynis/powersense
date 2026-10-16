"""
PowerSense Pro -- rule-based fault classifier.

Pure deterministic rules over the derived values. No training, no neural
network, no model file: it works on the very first reading, which is exactly
what you want for a fault layer that has to be explainable to a plant
operator ("why did you stop my machine?" -> the evidence list).

Twelve conditions are evaluated. The highest-severity match wins, but every
matched condition's evidence is returned so the dashboard can show the full
picture.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("powersense.fault")

NOMINAL_VOLTAGE = 230.0

# --- thresholds ------------------------------------------------------------
OVERLOAD_LOAD_FACTOR = 115.0
THERMAL_STRESS_LIMIT = 130.0
VOLTAGE_SAG_LIMIT = NOMINAL_VOLTAGE * 0.90      # 207 V
VOLTAGE_SWELL_LIMIT = NOMINAL_VOLTAGE * 1.10    # 253 V
FLUCTUATION_STABILITY = 0.80
PF_LOW_LIMIT = 0.70
BEARING_STABILITY = 0.75
BEARING_MIN_LOAD = 30.0
BEARING_CV = 0.15
WINDING_PF_DROP_PCT = 15.0
LOOSENESS_STABILITY = 0.70
LOOSENESS_CURRENT_CV = 0.12
TOOL_WEAR_LOAD = 105.0
TOOL_WEAR_STABILITY = 0.85
TOOL_WEAR_PF = 0.80
IDLE_WASTE_SECONDS = 600.0                      # 10 minutes
ANOMALY_SCORE_THRESHOLD = 1.0                   # normalised; 2x => 2.0

SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# Per-device idle run tracking: device_id -> epoch seconds when idle started.
_idle_since: Dict[str, float] = {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _track_idle(device_id: str, machine_state: str) -> float:
    """Return how many seconds the machine has been continuously idle."""
    now = time.time()
    if machine_state == "idle":
        started = _idle_since.get(device_id)
        if started is None:
            _idle_since[device_id] = now
            return 0.0
        return max(0.0, now - started)
    _idle_since.pop(device_id, None)
    return 0.0


def reset_idle_tracking(device_id: Optional[str] = None) -> None:
    """Clear idle-run state (used by tests and by manual pipeline triggers)."""
    if device_id is None:
        _idle_since.clear()
    else:
        _idle_since.pop(device_id, None)


def classify_fault(
    reading: Dict[str, Any],
    derived: Dict[str, Any],
    anomaly_score: float = 0.0,
) -> Dict[str, Any]:
    """Classify the machine's fault state from one reading + its derived values.

    Args:
        reading:       raw energy_readings row.
        derived:       output of engine.calculate_derived_values().
        anomaly_score: normalised anomaly score from the AnomalyDetector
                       (1.0 == at threshold).

    Returns:
        dict with fault_type, severity, confidence, evidence, action, urgency,
        plus `all_faults` listing every condition that matched.
    """
    reading = reading if isinstance(reading, dict) else {}
    derived = derived if isinstance(derived, dict) else {}

    device_id = (
        derived.get("device_id") or reading.get("device_id") or "hackathon_machine"
    )

    voltage = _f(derived.get("voltage", reading.get("voltage")))
    current = _f(derived.get("current", reading.get("current")))
    active_power = _f(derived.get("active_power", reading.get("active_power")))
    power_factor = _f(derived.get("power_factor", reading.get("power_factor")))

    load_factor = _f(derived.get("load_factor"))
    thermal_stress_index = _f(derived.get("thermal_stress_index"))
    stability = _f(derived.get("power_stability_index"), 1.0)
    machine_state = derived.get("machine_state") or "off"

    power_cv = _f(derived.get("_power_cv"))
    current_cv = _f(derived.get("_current_cv"))
    pf_recent_avg = _f(derived.get("_pf_recent_avg"), power_factor)
    history_len = int(_f(derived.get("_history_len")))

    anomaly_score = _f(anomaly_score)
    idle_seconds = _track_idle(device_id, machine_state)

    # A machine that is switched off cannot have a mechanical fault. Report
    # cleanly instead of flagging zero-power noise.
    if machine_state == "off":
        return {
            "fault_type": "NORMAL",
            "severity": "none",
            "confidence": 0.99,
            "evidence": [
                f"Machine is off (active power {active_power:.1f} W < 10 W)."
            ],
            "action": "Continue normal operation",
            "urgency": "none",
            "all_faults": [],
            "device_id": device_id,
            "anomaly_score": round(anomaly_score, 4),
        }

    candidates: List[Dict[str, Any]] = []

    def add(
        fault_type: str,
        severity: str,
        confidence: float,
        evidence: List[str],
        action: str,
        urgency: str,
    ) -> None:
        candidates.append(
            {
                "fault_type": fault_type,
                "severity": severity,
                "confidence": round(max(0.0, min(1.0, confidence)), 3),
                "evidence": evidence,
                "action": action,
                "urgency": urgency,
            }
        )

    # 1. OVERLOAD -----------------------------------------------------------
    if load_factor > OVERLOAD_LOAD_FACTOR:
        add(
            "OVERLOAD",
            "critical",
            0.97,
            [
                f"Load factor {load_factor:.1f}% exceeds the "
                f"{OVERLOAD_LOAD_FACTOR:.0f}% overload limit.",
                f"Drawing {active_power:.0f} W against the rated load.",
            ],
            "Reduce the load or stop the machine now. Sustained overload will "
            "trip protection and shorten winding life.",
            "immediate",
        )

    # 2. THERMAL_STRESS -----------------------------------------------------
    if thermal_stress_index > THERMAL_STRESS_LIMIT:
        add(
            "THERMAL_STRESS",
            "critical",
            0.95,
            [
                f"Thermal stress index {thermal_stress_index:.1f}% is above the "
                f"{THERMAL_STRESS_LIMIT:.0f}% limit.",
                f"Current {current:.2f} A is well above the nameplate rating "
                f"(I^2 heating scales with the square of current).",
            ],
            "Stop the machine and let it cool. Check ventilation, cooling fan "
            "and ambient temperature before restarting.",
            "immediate",
        )

    # 3. VOLTAGE_SAG --------------------------------------------------------
    if 0.0 < voltage < VOLTAGE_SAG_LIMIT:
        add(
            "VOLTAGE_SAG",
            "high",
            0.94,
            [
                f"Supply voltage {voltage:.1f} V is below the "
                f"{VOLTAGE_SAG_LIMIT:.0f} V sag limit (-10% of nominal).",
                "Under-voltage raises current draw for the same mechanical "
                "load and overheats the windings.",
            ],
            "Check the incoming supply, cable sizing and connections. Contact "
            "the utility if the sag is upstream.",
            "immediate",
        )

    # 4. VOLTAGE_SWELL ------------------------------------------------------
    if voltage > VOLTAGE_SWELL_LIMIT:
        add(
            "VOLTAGE_SWELL",
            "high",
            0.94,
            [
                f"Supply voltage {voltage:.1f} V is above the "
                f"{VOLTAGE_SWELL_LIMIT:.0f} V swell limit (+10% of nominal).",
                "Over-voltage stresses insulation and saturates the motor core.",
            ],
            "Check the transformer tap setting and the upstream regulator. "
            "Consider a voltage stabiliser.",
            "immediate",
        )

    # 5. VOLTAGE_FLUCTUATION ------------------------------------------------
    if stability < FLUCTUATION_STABILITY and voltage > 0.0:
        add(
            "VOLTAGE_FLUCTUATION",
            "medium",
            0.80,
            [
                f"Power stability index {stability:.2f} is below "
                f"{FLUCTUATION_STABILITY:.2f}.",
                f"Power varies by {power_cv * 100:.1f}% around a mean of "
                f"{_f(derived.get('_power_mean')):.0f} W.",
            ],
            "Inspect the supply quality and shared-circuit loads. Log the "
            "pattern over a full shift before acting.",
            "monitor",
        )

    # 6. POWER_FACTOR_LOW ---------------------------------------------------
    if power_factor < PF_LOW_LIMIT:
        add(
            "POWER_FACTOR_LOW",
            "medium",
            0.92,
            [
                f"Power factor {power_factor:.2f} is below the "
                f"{PF_LOW_LIMIT:.2f} limit.",
                f"Reactive power {_f(derived.get('reactive_power')):.0f} VAR is "
                "being drawn without doing useful work.",
                f"Estimated monthly tariff impact: Rs "
                f"{_f(derived.get('pf_monthly_impact')):.0f}.",
            ],
            "Install or service power factor correction capacitors. Check for "
            "a chronically under-loaded motor.",
            "week",
        )

    # 7. BEARING_WEAR -------------------------------------------------------
    if (
        stability < BEARING_STABILITY
        and load_factor > BEARING_MIN_LOAD
        and power_cv > BEARING_CV
    ):
        add(
            "BEARING_WEAR",
            "high",
            0.84,
            [
                f"Stability {stability:.2f} below {BEARING_STABILITY:.2f} while "
                f"loaded at {load_factor:.1f}%.",
                f"Power coefficient of variation {power_cv:.3f} exceeds "
                f"{BEARING_CV:.2f}, the signature of rising friction torque.",
                f"Based on {history_len} trailing samples.",
            ],
            "Schedule a bearing inspection: listen for growl, check grease and "
            "shaft play. Order replacement bearings.",
            "week",
        )

    # 8. WINDING_DEGRADATION ------------------------------------------------
    if pf_recent_avg > 0.0 and history_len >= 5:
        pf_drop_pct = (pf_recent_avg - power_factor) / pf_recent_avg * 100.0
        if pf_drop_pct > WINDING_PF_DROP_PCT:
            add(
                "WINDING_DEGRADATION",
                "high",
                0.82,
                [
                    f"Power factor fell {pf_drop_pct:.1f}% from a recent average "
                    f"of {pf_recent_avg:.2f} to {power_factor:.2f}.",
                    "A sudden PF collapse at steady load points to insulation "
                    "or winding deterioration.",
                ],
                "Perform an insulation resistance (megger) test and a winding "
                "resistance balance check at the next stop.",
                "week",
            )

    # 9. MECHANICAL_LOOSENESS -----------------------------------------------
    if stability < LOOSENESS_STABILITY and current_cv > LOOSENESS_CURRENT_CV:
        add(
            "MECHANICAL_LOOSENESS",
            "medium",
            0.78,
            [
                f"Stability {stability:.2f} below {LOOSENESS_STABILITY:.2f}.",
                f"Current fluctuating at a coefficient of variation of "
                f"{current_cv:.3f}.",
                "Erratic current with an erratic load is typical of a loose "
                "mount, coupling or belt.",
            ],
            "Check foundation bolts, coupling alignment and belt tension.",
            "week",
        )

    # 10. BLADE_TOOL_WEAR ---------------------------------------------------
    if (
        load_factor > TOOL_WEAR_LOAD
        and stability > TOOL_WEAR_STABILITY
        and power_factor > TOOL_WEAR_PF
    ):
        add(
            "BLADE_TOOL_WEAR",
            "medium",
            0.80,
            [
                f"Load factor {load_factor:.1f}% above {TOOL_WEAR_LOAD:.0f}% but "
                f"the load is steady (stability {stability:.2f}).",
                f"Power factor {power_factor:.2f} is healthy, so the motor is "
                "fine -- the extra work is at the tool.",
                "A steady rise in power at good PF means a dull blade or tool "
                "cutting harder for the same output.",
            ],
            "Inspect and replace the cutting tool or blade. Verify feed rate "
            "and material.",
            "week",
        )

    # 11. IDLE_WASTE --------------------------------------------------------
    if machine_state == "idle" and idle_seconds > IDLE_WASTE_SECONDS:
        wasted_hours = idle_seconds / 3600.0
        idle_cost_hr = _f(derived.get("idle_cost_per_hour"))
        add(
            "IDLE_WASTE",
            "low",
            0.90,
            [
                f"Machine has been idle for {idle_seconds / 60.0:.0f} minutes "
                f"(limit {IDLE_WASTE_SECONDS / 60.0:.0f} min).",
                f"Idle burn rate Rs {idle_cost_hr:.2f}/hour -- about Rs "
                f"{idle_cost_hr * wasted_hours:.2f} wasted so far.",
                f"Current draw {active_power:.0f} W with no useful output.",
            ],
            "Switch the machine off between jobs, or enable an auto-standby "
            "timer.",
            "monitor",
        )

    # 12. UNKNOWN_ANOMALY ---------------------------------------------------
    if not candidates and anomaly_score > 2.0 * ANOMALY_SCORE_THRESHOLD:
        add(
            "UNKNOWN_ANOMALY",
            "medium",
            0.65,
            [
                f"Anomaly score {anomaly_score:.2f} is more than twice the "
                f"detection threshold.",
                "No deterministic rule matched -- the operating point simply "
                "does not resemble the learned normal behaviour.",
            ],
            "Inspect the machine manually and note any change in process, "
            "material or setup. Label this event so the model can learn it.",
            "monitor",
        )

    # --- pick the winner ---------------------------------------------------
    if not candidates:
        return {
            "fault_type": "NORMAL",
            "severity": "none",
            "confidence": 0.95,
            "evidence": [
                f"All {12} fault rules clear.",
                f"PF {power_factor:.2f}, load {load_factor:.1f}%, voltage "
                f"{voltage:.1f} V, stability {stability:.2f}.",
            ],
            "action": "Continue normal operation",
            "urgency": "none",
            "all_faults": [],
            "device_id": device_id,
            "anomaly_score": round(anomaly_score, 4),
        }

    candidates.sort(
        key=lambda c: (SEVERITY_ORDER.get(c["severity"], 0), c["confidence"]),
        reverse=True,
    )
    primary = dict(candidates[0])

    # Surface every matched condition's evidence on the primary fault, so a
    # single alert carries the whole story.
    all_evidence = list(primary["evidence"])
    for other in candidates[1:]:
        all_evidence.append(
            f"[also {other['fault_type']} / {other['severity']}] "
            + other["evidence"][0]
        )

    primary["evidence"] = all_evidence
    primary["all_faults"] = [
        {
            "fault_type": c["fault_type"],
            "severity": c["severity"],
            "confidence": c["confidence"],
            "evidence": c["evidence"],
            "action": c["action"],
            "urgency": c["urgency"],
        }
        for c in candidates
    ]
    primary["device_id"] = device_id
    primary["anomaly_score"] = round(anomaly_score, 4)
    return primary
