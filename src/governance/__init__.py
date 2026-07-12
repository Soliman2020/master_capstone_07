"""Domain-agnostic governance spine.

This package MUST NOT import from ``src.domain`` and MUST NOT contain
domain-specific string literals (tenant/lease/rent/maintenance/evict/...).
``tests/test_governance_no_domain_imports.py`` enforces this so swapping in
a domain package (e.g. the SOC ``domain/``) cannot break silently.
"""