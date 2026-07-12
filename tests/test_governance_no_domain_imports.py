"""Guard: governance/ must stay domain-agnostic.

A domain layer lifts ``src/governance`` verbatim and swaps in its own
``src/domain`` (the SOC copilot uses a security-operations domain). If any
governance module imports from ``domain`` or hardcodes a domain string
(tenant/lease/rent/maintenance/evict/lockout/ledger/unit_id/pytesseract), the
transfer breaks silently. This test fails loudly instead.

This is the single most important guard for the domain-agnostic-spine
claim: governance contributes its *method*, not domain code.
"""

import ast
import sys
from pathlib import Path

GOVERNANCE_DIR = Path(__file__).resolve().parents[1] / "src" / "governance"

# Strings that would leak a property-management domain into governance.
# Each is checked as a standalone token where it appears, to avoid false
# positives from substrings (e.g. "unit" inside "community"). We use a simple
# word-boundary regex on the source text.
FORBIDDEN_LITERALS = [
    "tenant", "lease", "rent", "maintenance", "evict", "lockout",
    "ledger", "unit_id", "pytesseract",
]

# Modules governance is allowed to import from (stdlib + langchain + own pkg).
ALLOWED_IMPORT_ROOTS = {
    "json", "hashlib", "os", "re", "sqlite3", "statistics", "dataclasses",
    "pathlib", "typing", "datetime", "tempfile", "langchain_core", "langgraph",
    "yaml", "base64", "PIL",
}


def _governance_files():
    return [p for p in GOVERNANCE_DIR.glob("*.py") if p.name != "__init__.py"]


def test_no_domain_imports():
    """governance/*.py must not import from src.domain (or a bare 'domain')."""
    for path in _governance_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [n.name.split(".")[0] for n in node.names]
                is_relative = False
            elif isinstance(node, ast.ImportFrom):
                # Relative imports (.audit etc.) are intra-package and fine.
                if node.level and node.level > 0:
                    continue
                if node.module is None:
                    continue
                roots = [node.module.split(".")[0]]
                is_relative = False
            else:
                continue
            for r in roots:
                assert r != "domain", f"{path.name} imports forbidden 'domain'"
                if r and r not in ALLOWED_IMPORT_ROOTS and not r.startswith("langchain") \
                        and r not in {"__future__"}:
                    # Unknown root — could be a domain sneaking in. Flag it.
                    raise AssertionError(f"{path.name} imports non-allowlisted root '{r}'")


def test_no_domain_string_literals():
    """governance/*.py must not contain domain-specific string literals.

    We check only string *constants used as values* — actual string literals
    in code — not module/class/function docstrings. A docstring that says
    "no 'tenant' strings here" is documentation and should be allowed; a
    string literal "tenant" used as a key or action name is a leak.
    """
    import re

    def _docstring_nodes(tree):
        """Return the set of line numbers of all docstring constants."""
        ds = set()
        for parent in ast.walk(tree):
            body = getattr(parent, "body", None)
            if body and isinstance(body, list) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    ds.add(first.value.lineno)
        return ds

    for path in _governance_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.lineno in docstrings:
                    continue  # it's a docstring — allowed
                for lit in FORBIDDEN_LITERALS:
                    if re.search(r"\b" + re.escape(lit) + r"\b", node.value, re.IGNORECASE):
                        raise AssertionError(
                            f"{path.name}:{node.lineno} string literal leaks '{lit}': {node.value!r}"
                        )


def test_governance_dir_exists():
    assert GOVERNANCE_DIR.is_dir(), f"governance dir not found at {GOVERNANCE_DIR}"
    assert len(_governance_files()) > 0, "no governance modules found"