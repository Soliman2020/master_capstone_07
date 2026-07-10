"""Single source of truth for P7 determinism + schema version.

`SEED = 42` is non-negotiable: any generator, fusion rule, or KB loader
that needs randomness MUST import this constant. Changing the seed
invalidates every test fixture and every reported metric.
"""
SEED = 42
SCHEMA_VERSION = "p7-v0.1-vertical-slice"
