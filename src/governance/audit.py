"""Hash-chained JSONL audit log.

Each line is one event (a tool call, a reviewer decision, a block, or a human
approval). Every line stores the hash of the previous line, so if someone
edits a past entry the chain breaks and verify_chain() catches it.

This is domain-agnostic — it just records whatever action/args/node the graph
passes in. The SOC copilot uses it for the tool-call + reviewer-decision trail.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

GENESIS_HASH = hashlib.sha256(b"genesis-p6-audit").hexdigest()


class ChainBrokenError(Exception):
    """Raised when verify_chain finds a hash that doesn't match."""


def _canonical_json(obj: dict) -> str:
    # Sort keys + no spaces so the hash is stable regardless of dict order.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AuditLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Start a fresh chain, or continue from the last line if the file exists.
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            self._prev_hash = json.loads(lines[-1])["this_hash"]
            self._seq = len(lines)
        else:
            self._prev_hash = GENESIS_HASH
            self._seq = 0

    def _append(self, record: dict) -> str:
        record["seq"] = self._seq
        record["ts"] = _utc_now()
        record["prev_hash"] = self._prev_hash
        # Hash everything except this_hash itself.
        this_hash = hashlib.sha256(
            (self._prev_hash + _canonical_json(record)).encode("utf-8")
        ).hexdigest()
        record["this_hash"] = this_hash
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq += 1
        self._prev_hash = this_hash
        return this_hash

    # Four named entry points so the log distinguishes event kinds.
    def log_call(self, *, turn_id, action, args, tool, result_summary, actor="worker"):
        return self._append({"turn_id": turn_id, "node": "worker_dispatch", "kind": "call",
                             "actor": actor, "action": action, "args": args or {},
                             "tool": tool, "result_summary": result_summary})

    def log_decision(self, *, turn_id, node, decision, rationale="", actor="system"):
        return self._append({"turn_id": turn_id, "node": node, "kind": "decision",
                             "actor": actor, "decision": decision, "rationale": rationale})

    def log_block(self, *, turn_id, action, args, violations, block_reason, actor="reviewer"):
        return self._append({"turn_id": turn_id, "node": "reviewer", "kind": "block",
                             "actor": actor, "action": action, "args": args or {},
                             "violations": violations, "block_reason": block_reason})

    def log_human_approval(self, *, turn_id, action, approver, granted, note=""):
        return self._append({"turn_id": turn_id, "node": "human_approval",
                             "kind": "human_approval", "actor": approver,
                             "action": action, "granted": granted, "note": note})

    def verify_chain(self) -> bool:
        """Recompute every hash from genesis. Raise ChainBrokenError on mismatch."""
        if not self.path.exists():
            return True
        prev = GENESIS_HASH
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                this_hash = record.pop("this_hash")
                expected = hashlib.sha256(
                    (prev + _canonical_json(record)).encode("utf-8")
                ).hexdigest()
                if expected != this_hash:
                    raise ChainBrokenError(
                        f"chain broken at seq={record.get('seq')}: "
                        f"expected {expected}, got {this_hash}"
                    )
                prev = this_hash
        return True

    def read_all(self) -> list[dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    # Self-check: build a small chain, verify it, tamper, expect a break.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        log = AuditLogger(Path(d) / "audit.jsonl")
        log.log_decision(turn_id="t1", node="planner", decision="plan_issued")
        log.log_block(turn_id="t1", action="demo.forbidden", args={"x": 1},
                      violations=["action_not_allowed"], block_reason="outside authority")
        log.log_call(turn_id="t1", action="demo.read", args={"q": "z"},
                     tool="demo.read", result_summary="ok")
        assert log.verify_chain() is True

        # Tamper with the middle line.
        path = Path(d) / "audit.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[1])
        tampered["block_reason"] = "tampered"
        lines[1] = json.dumps(tampered, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            log.verify_chain()
        except ChainBrokenError as e:
            print(f"audit.py self-check OK ({e})")
        else:
            raise AssertionError("tampering was not detected")