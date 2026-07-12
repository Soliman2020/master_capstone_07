"""SOC intake node: stage an incident -> redacted text for the planner.

P6's intake was OCR-driven (image -> pytesseract -> vision fallback). P7's
is event-driven: the caller puts an ``incident_id`` in ``domain_state`` and
the intake node reads that incident from the fused incidents parquet,
redacts PII with the policy's patterns, and surfaces a short text summary
as ``redacted_text`` for the planner. No OCR, no image path.

Same factory signature as P6's make_intake_node so build_graph wires it the
same way: ``make_intake_node(policy, audit, llm=None, ...)``. The ``llm`` and
``ocr_threshold`` args are accepted for signature parity but unused here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT / "src"
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from governance.audit import AuditLogger
from governance.pii import redact
from governance.policy import Policy

from src.utils.io import read_parquet


def _incident_text(incident_id: str) -> str:
    """Build a short analyst-facing description of the incident for the planner."""
    df = read_parquet("incidents")
    row = df[df["incident_id"] == incident_id]
    if row.empty:
        raise ValueError(
            f"incident {incident_id} not found in incidents.parquet — "
            "run src/fusion/incidents.py first."
        )
    r = row.iloc[0]
    return (
        f"Incident {r['incident_id']} ({r['incident_type']}), "
        f"risk_band={r['risk_band']}, risk_score={r['risk_score']}, "
        f"zone={r['zone_id']}. "
        f"Linked events: {r['linked_event_ids'] or 'none'}; "
        f"linked logs: {r['linked_log_ids'] or 'none'}."
    )


def make_intake_node(policy: Policy, audit: AuditLogger, llm: Any = None,
                     ocr_threshold: int = 65):
    """Factory: returns the intake node function. Signature mirrors P6 so
    build_graph wiring is identical. ``llm`` and ``ocr_threshold`` are unused
    (kept for parity) — P7 intake is event-driven, not OCR-driven.
    """

    def intake(state: dict) -> dict:
        # Allow the caller to pre-supply redacted_text (scripted scenarios /
        # tests that don't need a real incident). Otherwise stage the incident.
        user_text = state.get("redacted_text", "")
        if not user_text:
            incident_id = state.get("domain_state", {}).get("incident_id")
            if not incident_id:
                audit.log_decision(
                    turn_id=state.get("turn_id", ""), node="ingest",
                    decision="intake_no_incident",
                    rationale="domain_state has no incident_id",
                )
                # No incident staged -> empty text; planner will see nothing
                # and the turn ends cleanly.
                return {"redacted_text": "", "messages": state.get("messages", [])}
            user_text = _incident_text(incident_id)
            audit.log_decision(
                turn_id=state.get("turn_id", ""), node="ingest",
                decision="intake_staged_incident",
                rationale=f"incident_id={incident_id}",
            )

        redacted = redact(user_text, policy)
        return {"redacted_text": redacted, "messages": state.get("messages", [])}

    return intake


if __name__ == "__main__":
    # Self-check: stage the first incident, redact a fake PII string.
    import tempfile

    pol = Policy.from_dict({
        "domain": "self_test",
        "pii_redaction": {"enabled": True, "patterns": [
            {"name": "email", "regex": r"[\w.+-]+@[\w-]+\.[\w.-]+",
             "replacement": "[EMAIL-REDACTED]"}]},
        "actions": [],
    })
    with tempfile.TemporaryDirectory() as d:
        audit = AuditLogger(Path(d) / "audit.jsonl")
        node = make_intake_node(pol, audit)
        out = node({
            "turn_id": "t1", "messages": [],
            "redacted_text": "Contact analyst@corp.com about INC-000001",
            "domain_state": {},
        })
        assert "[EMAIL-REDACTED]" in out["redacted_text"], out
        assert "analyst@corp.com" not in out["redacted_text"], out
        print("domain/intake_node.py self-check OK")