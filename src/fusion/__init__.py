"""Fusion package: rules-first incident detection.

pure functions, no classes. Each rule takes the events +
logs dataframes and returns candidate incident rows. The orchestrator
(risk_scorer / incident generator) decides which candidates become
real incidents and assigns risk_score + risk_band.
"""
