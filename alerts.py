"""
PowerSense Pro -- alerting (Twilio WhatsApp + SendGrid email).

Both channels are optional. When credentials are absent the functions log and
return a "skipped" result instead of raising, and the alert is still written to
the `alerts` table so the dashboard shows it. Nothing here is allowed to take
the pipeline down.

De-duplication is two-layered:
  1. an in-process cooldown cache (fast, survives the 30 s job cadence)
  2. a DB check for a recent alert with the same code (survives restarts)
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import db

logger = logging.getLogger("powersense.alerts")

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# --- Twilio WhatsApp -------------------------------------------------------
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_WHATSAPP = os.environ.get(
    "TWILIO_FROM_WHATSAPP", "whatsapp:+14155238886"
)
ALERT_WHATSAPP_TO = os.environ.get("ALERT_WHATSAPP_TO", "")

# --- SendGrid email --------------------------------------------------------
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "alerts@powersensepro.com")

# --- behaviour -------------------------------------------------------------
ALERT_COOLDOWN_SECONDS = float(os.environ.get("ALERT_COOLDOWN_SECONDS", "1800"))

# Severities that justify waking somebody up on WhatsApp.
WHATSAPP_SEVERITIES = ("high", "critical")

SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "none": "🟢",
}

# (device_id, code) -> epoch seconds of the last send
_last_sent: Dict[str, float] = {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def whatsapp_configured() -> bool:
    return bool(
        TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_WHATSAPP and ALERT_WHATSAPP_TO
    )


def email_configured() -> bool:
    return bool(SENDGRID_API_KEY and ALERT_EMAIL_TO and ALERT_EMAIL_FROM)


def channel_status() -> Dict[str, Any]:
    """Which notification channels are usable in this deployment."""
    return {
        "whatsapp_configured": whatsapp_configured(),
        "email_configured": email_configured(),
        "whatsapp_to": ALERT_WHATSAPP_TO or None,
        "email_to": ALERT_EMAIL_TO or None,
        "cooldown_seconds": ALERT_COOLDOWN_SECONDS,
    }


def reset_cooldowns(device_id: Optional[str] = None) -> None:
    """Clear the in-process cooldown cache (manual re-trigger, tests)."""
    if device_id is None:
        _last_sent.clear()
        return
    for key in [k for k in _last_sent if k.startswith(f"{device_id}|")]:
        _last_sent.pop(key, None)


def _in_cooldown(device_id: str, code: str) -> bool:
    key = f"{device_id}|{code}"
    last = _last_sent.get(key)
    if last is None:
        return False
    return (time.time() - last) < ALERT_COOLDOWN_SECONDS


def _mark_sent(device_id: str, code: str) -> None:
    _last_sent[f"{device_id}|{code}"] = time.time()


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------
def send_whatsapp_alert(
    device_id: str,
    fault_type: str,
    severity: str,
    message: str,
    action: str,
) -> Dict[str, Any]:
    """Send one WhatsApp alert through Twilio.

    Returns a dict with `sent` plus either `sid` or `reason`/`error`.
    """
    if not whatsapp_configured():
        logger.info(
            "WhatsApp not configured -- skipping alert %s/%s", device_id, fault_type
        )
        return {"sent": False, "channel": "whatsapp", "reason": "not_configured"}

    emoji = SEVERITY_EMOJI.get(str(severity).lower(), "⚡")
    body = (
        "⚡ PowerSense Alert\n"
        f"Device: {device_id}\n"
        f"Fault: {fault_type}\n"
        f"Severity: {emoji} {str(severity).upper()}\n"
        f"{message}\n"
        f"Action: {action}"
    )

    try:
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            from_=TWILIO_FROM_WHATSAPP,
            to=ALERT_WHATSAPP_TO,
            body=body,
        )
        logger.info(
            "WhatsApp alert sent for %s/%s (sid=%s)", device_id, fault_type, msg.sid
        )
        return {
            "sent": True,
            "channel": "whatsapp",
            "sid": msg.sid,
            "to": ALERT_WHATSAPP_TO,
            "body": body,
        }
    except Exception as exc:
        logger.error("WhatsApp send failed for %s/%s: %s", device_id, fault_type, exc)
        return {"sent": False, "channel": "whatsapp", "error": str(exc), "body": body}


# ---------------------------------------------------------------------------
# Email daily report
# ---------------------------------------------------------------------------
def _summary_rows_html(summary: Dict[str, Any]) -> str:
    tariff = _f(os.environ.get("TARIFF_RATE", "9.0"), 9.0)
    grid_factor = _f(os.environ.get("INDIA_GRID_FACTOR", "0.82"), 0.82)
    total_energy = _f(summary.get("total_energy"))

    rows = [
        ("Total energy", f"{total_energy:.3f} kWh"),
        ("Energy cost", f"&#8377; {total_energy * tariff:,.2f}"),
        ("Carbon footprint", f"{total_energy * grid_factor:.3f} kg CO&#8322;"),
        ("Average power", f"{_f(summary.get('avg_power')):.1f} W"),
        ("Peak power", f"{_f(summary.get('max_power')):.1f} W"),
        ("Minimum power", f"{_f(summary.get('min_power')):.1f} W"),
        ("Average power factor", f"{_f(summary.get('avg_pf')):.3f}"),
        ("Worst power factor", f"{_f(summary.get('min_pf')):.3f}"),
        ("Average voltage", f"{_f(summary.get('avg_voltage')):.1f} V"),
        ("Average current", f"{_f(summary.get('avg_current')):.2f} A"),
        ("Average frequency", f"{_f(summary.get('avg_frequency')):.2f} Hz"),
        ("Idle time", f"{_f(summary.get('idle_percent')):.1f} %"),
        ("Readings collected", f"{int(_f(summary.get('reading_count')))}"),
    ]
    return "".join(
        "<tr>"
        f'<td style="padding:8px 14px;border-bottom:1px solid #e5e7eb;color:#374151;">{label}</td>'
        f'<td style="padding:8px 14px;border-bottom:1px solid #e5e7eb;color:#111827;'
        f'font-weight:600;text-align:right;">{value}</td>'
        "</tr>"
        for label, value in rows
    )


def build_daily_report_html(
    device_id: str,
    date: str,
    summary_dict: Dict[str, Any],
    alerts_list: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render the daily report email body."""
    alerts_list = alerts_list or []
    if alerts_list:
        alert_items = "".join(
            "<li style=\"margin-bottom:6px;color:#374151;\">"
            f"<strong>{a.get('level', '?').upper()}</strong> "
            f"{a.get('code', '')} &mdash; {a.get('message', '')}"
            "</li>"
            for a in alerts_list[:10]
        )
        alerts_block = (
            '<h3 style="margin:24px 0 8px;font-size:15px;color:#111827;">'
            f"Alerts raised ({len(alerts_list)})</h3>"
            f'<ul style="margin:0;padding-left:20px;font-size:13px;">{alert_items}</ul>'
        )
    else:
        alerts_block = (
            '<p style="margin:24px 0 0;padding:12px 14px;background:#ecfdf5;'
            'border-radius:8px;color:#065f46;font-size:13px;">'
            "No alerts were raised in this period.</p>"
        )

    return f"""<!doctype html>
<html>
<body style="margin:0;padding:24px;background:#f3f4f6;
             font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:620px;margin:0 auto;background:#ffffff;border-radius:12px;
              overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <div style="background:#0f172a;padding:20px 24px;">
      <h1 style="margin:0;font-size:18px;color:#ffffff;">&#9889; PowerSense Pro</h1>
      <p style="margin:4px 0 0;font-size:13px;color:#94a3b8;">
        Daily energy report &mdash; {date}
      </p>
    </div>
    <div style="padding:24px;">
      <p style="margin:0 0 16px;font-size:14px;color:#374151;">
        Summary for device <strong>{device_id}</strong> on <strong>{date}</strong>.
      </p>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        {_summary_rows_html(summary_dict)}
      </table>
      {alerts_block}
      <p style="margin:24px 0 0;font-size:11px;color:#9ca3af;">
        Generated automatically by the PowerSense Pro pipeline at
        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.
      </p>
    </div>
  </div>
</body>
</html>"""


def send_email_daily_report(
    device_id: str,
    date: str,
    summary_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Send the daily HTML energy summary through SendGrid."""
    if not email_configured():
        logger.info("SendGrid not configured -- skipping daily report for %s", device_id)
        return {"sent": False, "channel": "email", "reason": "not_configured"}

    summary_dict = summary_dict or {}
    alerts_list = db.get_alerts(device_id, resolved=None, limit=20)
    html = build_daily_report_html(device_id, date, summary_dict, alerts_list)

    total_energy = _f(summary_dict.get("total_energy"))
    tariff = _f(os.environ.get("TARIFF_RATE", "9.0"), 9.0)
    subject = (
        f"PowerSense {date}: {total_energy:.2f} kWh / "
        f"Rs {total_energy * tariff:,.0f} - {device_id}"
    )

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        mail = Mail(
            from_email=ALERT_EMAIL_FROM,
            to_emails=ALERT_EMAIL_TO,
            subject=subject,
            html_content=html,
        )
        client = SendGridAPIClient(SENDGRID_API_KEY)
        response = client.send(mail)
        logger.info(
            "Daily report emailed for %s (status %s)", device_id, response.status_code
        )
        return {
            "sent": True,
            "channel": "email",
            "status_code": response.status_code,
            "to": ALERT_EMAIL_TO,
            "subject": subject,
        }
    except Exception as exc:
        logger.error("Daily report email failed for %s: %s", device_id, exc)
        return {"sent": False, "channel": "email", "error": str(exc), "subject": subject}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def check_and_send_alerts(
    device_id: str,
    derived_values: Dict[str, Any],
    fault_prediction: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether an alert is warranted, log it, and notify if severe.

    Every non-NORMAL fault is written to the `alerts` table (subject to
    de-duplication). WhatsApp fires only for `high` and `critical`.
    """
    derived_values = derived_values if isinstance(derived_values, dict) else {}
    fault_prediction = fault_prediction if isinstance(fault_prediction, dict) else {}

    fault_type = str(fault_prediction.get("fault_type") or "NORMAL")
    severity = str(fault_prediction.get("severity") or "none").lower()
    action = str(fault_prediction.get("action") or "Monitor the machine.")
    evidence = fault_prediction.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    result: Dict[str, Any] = {
        "device_id": device_id,
        "fault_type": fault_type,
        "severity": severity,
        "logged": False,
        "whatsapp": {"sent": False, "reason": "not_attempted"},
        "deduplicated": False,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    # Nothing wrong -> close any open alerts for faults that have cleared.
    if fault_type == "NORMAL" or severity in ("none", ""):
        result["reason"] = "no_fault"
        return result

    # --- de-duplication ---------------------------------------------------
    if _in_cooldown(device_id, fault_type):
        result["deduplicated"] = True
        result["reason"] = "in_cooldown"
        return result

    if db.recent_alert_exists(
        device_id, fault_type, within_minutes=int(ALERT_COOLDOWN_SECONDS // 60) or 1
    ):
        _mark_sent(device_id, fault_type)
        result["deduplicated"] = True
        result["reason"] = "recent_db_alert"
        return result

    # --- build the message ------------------------------------------------
    headline = evidence[0] if evidence else f"{fault_type} detected."
    context = (
        f"PF {_f(derived_values.get('power_factor')):.2f} | "
        f"load {_f(derived_values.get('load_factor')):.0f}% | "
        f"{_f(derived_values.get('voltage')):.0f} V | "
        f"{_f(derived_values.get('active_power')):.0f} W | "
        f"health {_f(derived_values.get('motor_health_score')):.0f}/100"
    )
    message = f"{headline} ({context})"

    # --- log to Supabase ---------------------------------------------------
    result["logged"] = db.insert_alert(
        device_id=device_id,
        level=severity,
        code=fault_type,
        message=message,
        action=action,
    )
    result["message"] = message

    # --- notify ------------------------------------------------------------
    if severity in WHATSAPP_SEVERITIES:
        result["whatsapp"] = send_whatsapp_alert(
            device_id=device_id,
            fault_type=fault_type,
            severity=severity,
            message=message,
            action=action,
        )
    else:
        result["whatsapp"] = {
            "sent": False,
            "reason": f"severity '{severity}' below WhatsApp threshold",
        }

    _mark_sent(device_id, fault_type)
    return result
