"""Risk scoring + incident finalization.

Pure functions. `score_candidate` turns a raw rule candidate into a
candidate with risk_score in [0, 100] and a risk_band label. The
incident generator (`incidents.py`) materializes the candidates to
incident rows and writes them to Parquet.

Risk band policy (per the P3 lesson: report the trade-off, don't hide it):
  critical >= 80   (forces human review)
  high     >= 60
  medium   >= 40
  low      <  40
"""
from __future__ import annotations

# Each rule has a base risk contribution. Composite score is base +
# size bonus + confidence bonus, capped at 100.
RULE_BASE_RISK = {
    "intrusion_restricted": 70,
    "repeated_denials":     55,
    "cross_anomaly":        60,
    "tailgate_door":        65,
}

# Bonus per additional linked event/log, capped.
LINK_BONUS_PER = 3
LINK_BONUS_CAP = 15

# Confidence bonus only applies to the intrusion rule (others don't
# have a confidence value in the candidate).
CONFIDENCE_BONUS_CAP = 10


def score_candidate(
    candidate: dict,
    events_lookup: dict[str, float] | None = None,
) -> dict:
    """Return a new dict with risk_score + risk_band added.

    `events_lookup`: optional {event_id: confidence_score} for rules
    that include surveillance events. The intrusion rule pulls its
    confidence from here; the cross-anomaly rule averages its
    linked_event_ids' confidences.
    """
    rule = candidate["rule"]
    base = RULE_BASE_RISK.get(rule, 40)

    # Size bonus: more linked evidence -> higher score.
    n_links = len(candidate.get("linked_event_ids", [])) + len(candidate.get("linked_log_ids", []))
    size_bonus = min(n_links * LINK_BONUS_PER, LINK_BONUS_CAP)

    # Confidence bonus: for rules that touch surveillance events, use
    # the mean confidence of linked events (clipped to [0, 1]).
    conf_bonus = 0.0
    if events_lookup:
        confs = [
            events_lookup[eid]
            for eid in candidate.get("linked_event_ids", [])
            if eid in events_lookup
        ]
        if confs:
            mean_conf = sum(confs) / len(confs)
            conf_bonus = min(mean_conf * CONFIDENCE_BONUS_CAP, CONFIDENCE_BONUS_CAP)

    score = min(int(base + size_bonus + conf_bonus), 100)
    return {
        **candidate,
        "risk_score": score,
        "risk_band": _band(score),
    }


def _band(score: int) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 40:
        return "medium"
    return "low"
