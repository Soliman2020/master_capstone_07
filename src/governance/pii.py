"""PII redactor driven by the policy's redaction_patterns.

Domain-agnostic: it just applies whatever regex patterns the policy carries.
One policy may list ssn/email/phone; another may list badge/employee IDs. The
redactor doesn't care — it runs the patterns and returns the scrubbed text.
"""

from __future__ import annotations

import re

from .policy import Policy


def redact(text: str, policy: Policy) -> str:
    """Apply every redaction pattern in the policy, in order."""
    if not policy.redaction_enabled or not text:
        return text
    out = text
    for p in policy.redaction_patterns:
        out = re.sub(p.regex, p.replacement, out)
    return out