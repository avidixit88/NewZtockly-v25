import smtplib
from email.message import EmailMessage
from typing import Optional, Dict, Any

def send_email_alert(
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    subject: str,
    body: str,
) -> None:
    """Send a simple plaintext email via SMTP (Gmail app-password compatible)."""
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def format_alert_email(payload: Dict[str, Any]) -> str:
    """Create a human-readable email body from an alert payload dict."""
    # The app's internal row/payload objects sometimes use TitleCase keys
    # (e.g., "Symbol", "Bias", "Score", "AsOf") while other paths use
    # lowercase keys ("symbol", "bias", ...). Normalize reads so emails
    # never come through as "None".
    def g(*keys, default=None):
        for k in keys:
            if k in payload and payload.get(k) is not None:
                return payload.get(k)
        return default

    lines = []
    lines.append(f"Time: {g('time','Time','as_of','as_of','asof','AsOf')}")
    lines.append(f"Symbol: {g('symbol','Symbol')}")
    lines.append(
        f"Bias: {g('bias','Bias')}   Tier: {g('tier','Tier','stage','Stage')}   Score: {g('score','Score')}   Session: {g('session','Session')}"
    )
    lines.append("")
    lines.append(f"Last: {g('last','Last')}")
    lines.append(f"Entry (limit): {g('entry_limit','Entry','entry')}")
    lines.append(f"Chase line: {g('entry_chase_line','Chase','chase')}")

    # RIDE / continuation fields (if present)
    br = g('break_trigger','BreakTrigger','breakTrigger')
    pb = g('pullback_entry','PullbackEntry','pullback_entry')
    if pb is not None:
        lines.append(f"Pullback entry: {pb}")
    if br is not None:
        lines.append(f"Break trigger: {br}")

    lines.append(f"Stop: {g('stop','Stop')}")
    lines.append(f"TP0: {g('tp0','TP0')}")
    # Support both canonical keys (tp1/tp2) and internal short keys (t1/t2)
    lines.append(f"TP1: {g('tp1','TP1','t1','T1')}")
    lines.append(f"TP2: {g('tp2','TP2','t2','T2')}")
    lines.append(f"TP3: {g('tp3','TP3','t3','T3')}")
    lines.append(f"ETA TP0 (min): {g('eta_tp0_min','ETA TP0 (min)')}")
    lines.append("")
    why = g('why','Why', default="") or ""
    lines.append("Why:")
    lines.append(str(why))
    lines.append("")
    extras = g('extras','Extras', default={}) or {}
    if extras:
        lines.append("Diagnostics:")
        for k in ["vwap_logic","session_vwap_include_premarket","atr_pct","baseline_atr_pct","atr_ref_pct","atr_score_scale","htf_bias","liquidity_phase"]:
            if k in extras:
                lines.append(f"- {k}: {extras.get(k)}")
    return "\n".join(lines)
