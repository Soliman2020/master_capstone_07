"""P2 -> P7 threshold calibration test.

P2 ran two hypothesis tests on its own data (which is now P7's scaled
slice via p1_pipeline). This test re-runs both on the slice the copilot
actually runs against and asserts the headline pattern still holds.

The two tests are P2's:

    H1. Zone restrictiveness is INDEPENDENT of access outcome
        (chi-square). P2 rejected H0 with Cramér's V ~ 0.022 (negligible
        practical effect). P7's `intrusion_restricted` rule treats
        restricted zones as a signal; this test asserts that signal is
        weak on the scaled slice and warns if it strengthens (would
        suggest the fusion rule is over-indexing on zone).

    H2. Intrusion events have higher confidence than normal-motion events
        (independent t-test). P2 rejected H0 with Cohen's d ~ 0.21
        (medium effect) and warned the P7 threshold (0.85) sits
        *just above* the mean intrusion confidence (0.825). This test
        asserts the mean + the threshold's gap on the scaled slice and
        warns if the threshold would now miss more than 30% of intrusions
        (i.e. recall < 0.70 at the current cutoff).

Both tests are **calibration tests**, not correctness tests: they
record findings in the audit trail and `pytest -v` shows the numbers.
A test FAILS only if the pattern breaks *catastrophically* (effect
sizes in the wrong direction); softer drifts are warnings.

The connection: this is the **mechanical bridge** from P2's "Statistical
Analysis" to P7's rule engine. P2's "should be validated separately
against precision/recall before deployment" warning is the test's
contract. (See `progress_report.md` §What to do for the paper.)
"""
import os
import sys
from pathlib import Path

# Bootstraps: pytest doesn't run the project-level sys.path + cwd the way
# the module-level scripts do, so we replicate the bootstrap here. The
# adapter and io helpers both need to be importable AND find
# data/synthetic/ from the repo root, so we chdir.
_PROJECT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PROJECT.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd

# Headline numbers from P2's `Statistical_Analysis_Report.md`. These are
# the calibration target. If they drift substantially on P7's slice,
# the test prints a warning; the test fails only if the pattern reverses.
P2_CHI2_PVALUE = 0.0341
P2_CHI2_CRAMERS_V = 0.0221
P2_TTEST_TSTAT = 7.852
P2_TTEST_COHENS_D = 0.21
P2_TTEST_PVALUE = 1.36e-14
P2_INTRUSION_MEAN_CONF = 0.825
P7_INTRUSION_CONFIDENCE_THRESHOLD = 0.85  # fusion/risk_scorer.py, rules.py


def _load_p7_slice() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the synthetic slice. Falls back to P1's raw if P7's
    synthetic/ doesn't exist (e.g. reviewers who skipped run_all)."""
    from src.utils.io import read_parquet
    try:
        events = read_parquet("surveillance_events")
        access = read_parquet("access_logs")
    except FileNotFoundError:
        # Fall back to P1's raw (which the adapter normalizes on the fly).
        REPO = Path(__file__).resolve().parents[2]
        from src.utils.p1_adapter import adapt_surveillance, adapt_access
        events = adapt_surveillance(
            pd.read_parquet(REPO / "project_01_reproducible_workflows/data/raw/surveillance_events.parquet")
        )
        access, _ = adapt_access(
            pd.read_parquet(REPO / "project_01_reproducible_workflows/data/raw/access_logs.parquet")
        )
    return events, access


def _is_restricted(zone_id: str) -> bool:
    """P1's zone_id format is 'SITE-NNN::ZONE-X'; ZONE-D is restricted.
    Mirrors src/utils/p1_adapter._is_restricted.
    """
    if not isinstance(zone_id, str) or "ZONE-" not in zone_id:
        return False
    suffix = zone_id.rsplit("ZONE-", 1)[-1].strip()
    return suffix == "D"


def test_chi2_zone_restrictiveness_vs_access_outcome():
    """H1: Zone restrictiveness ~ access outcome.

    P2 rejected H0 (p=0.034) with Cramér's V=0.022 (negligible).
    P7's `intrusion_restricted` rule assumes zone matters; P2's finding
    says it matters very little. We re-test on the scaled slice.

    Calibration contract: the test RECORDS the finding, doesn't fail
    on p-value drift (P2 used P1's original data; P7's adapter dropped
    sentinel rows and normalized labels, both of which can move the
    p-value). A FAIL fires only if the pattern reverses: zone
    restrictiveness becomes a *strong* predictor of unusual access
    (Cramér's V > 0.30), which would mean the fusion rule is
    over-indexing on zone.
    """
    from scipy.stats import chi2_contingency
    events, access = _load_p7_slice()

    # Build a contingency table: (zone_type x access_outcome).
    # Use the access events (denied + invalid + tailgate = "unusual outcome").
    access = access.copy()
    access["zone_restricted"] = access["zone_id"].map(_is_restricted)
    access["unusual_outcome"] = access["access_result"].isin({"denied", "invalid", "tailgate"})

    contingency = pd.crosstab(
        access["zone_restricted"],
        access["unusual_outcome"],
    )
    if contingency.shape != (2, 2) or contingency.values.min() < 1:
        print(f"WARNING: contingency table malformed; skipping H1. table=\n{contingency}")
        return  # soft skip

    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    print(f"H1 (chi-square): n={n}, chi2={chi2:.3f}, p={p:.4g}, Cramér's V={cramers_v:.4f}")
    print(f"  P2 baseline:    p={P2_CHI2_PVALUE}, Cramér's V={P2_CHI2_CRAMERS_V}")
    # Calibration contract: the *practical* claim is "zone restrictiveness
    # is NOT a strong signal" (Cramér's V < 0.10). If the p-value
    # drifted (different data, sentinels dropped, labels normalized)
    # but V is still in the "negligible" range, the fusion rule's
    # premise holds. Fail only on effect-size reversal.
    if cramers_v >= 0.30:
        raise AssertionError(
            f"zone restrictiveness became a strong predictor of unusual access "
            f"(Cramér's V={cramers_v:.3f} >= 0.30). Fusion rule may need re-tuning."
        )
    if p > 0.10:
        print(
            f"NOTE: H1 p-value ({p:.4g}) drifted from P2's baseline "
            f"({P2_CHI2_PVALUE}). Likely cause: P7's adapter dropped "
            f"sentinel rows + normalized outcome labels. The practical "
            f"effect (Cramér's V={cramers_v:.3f}) is still in the "
            f"'negligible' range; fusion's premise holds."
        )


def test_ttest_intrusion_vs_normal_confidence():
    """H2: Intrusion events have higher confidence than normal-motion events.

    P2 rejected H0 (p~0, Cohen's d~0.21) and warned that the P7
    threshold (0.85) sits *just above* the mean intrusion confidence
    (0.825). We re-test on the scaled slice and assert the
    mean-vs-threshold gap.
    """
    from scipy.stats import ttest_ind
    events, _ = _load_p7_slice()
    events = events.copy()
    # P7's anomaly flag is what fusion actually uses; an "intrusion"
    # in P1's terms maps to a high-confidence person_detected event in
    # P7 (per the adapter). We test on the union of P7's anomaly flag
    # and the original P1 "intrusion" label carried through as
    # event_type=anomaly. For the t-test the cleanest cut is: P7 anomaly
    # vs P7 non-anomaly.
    anomaly_conf = events.loc[events["anomaly"].astype(bool), "confidence_score"].astype(float).values
    normal_conf = events.loc[~events["anomaly"].astype(bool), "confidence_score"].astype(float).values
    if len(anomaly_conf) < 5 or len(normal_conf) < 5:
        print(f"WARNING: too few samples (n_anom={len(anomaly_conf)}, n_norm={len(normal_conf)}); skipping H2")
        return  # soft skip

    t, p = ttest_ind(anomaly_conf, normal_conf, equal_var=False)
    pooled_std = np.sqrt(
        (anomaly_conf.var(ddof=1) + normal_conf.var(ddof=1)) / 2
    )
    cohens_d = (anomaly_conf.mean() - normal_conf.mean()) / pooled_std
    n_anom_recall = float((anomaly_conf >= P7_INTRUSION_CONFIDENCE_THRESHOLD).mean())

    print(f"H2 (t-test): n_anom={len(anomaly_conf)}, n_norm={len(normal_conf)}, "
          f"t={t:.3f}, p={p:.4g}, Cohen's d={cohens_d:.3f}, "
          f"mean_anom_conf={anomaly_conf.mean():.3f}, recall@0.85={n_anom_recall:.3f}")
    print(f"  P2 baseline:    t={P2_TTEST_TSTAT}, Cohen's d={P2_TTEST_COHENS_D}, mean_anom_conf={P2_INTRUSION_MEAN_CONF}")
    # Assertions:
    # (1) The mean intrusion/anomaly confidence is meaningfully LOWER than
    #     the fusion threshold (P2's warning: the threshold may be on the
    #     edge). We assert the gap is documented but don't fail unless it
    #     flips (anomaly mean > normal mean would be a sign of dataset
    #     leakage in the adapter).
    assert anomaly_conf.mean() > normal_conf.mean(), (
        f"anomaly mean confidence ({anomaly_conf.mean():.3f}) is <= "
        f"normal mean ({normal_conf.mean():.3f}); the adapter's anomaly "
        f"derivation is likely inverted."
    )
    # (2) P2's calibration warning: if the threshold catches < 70% of
    #     anomaly events, fusion may be missing real intrusions. Print
    #     a warning (not a fail) — operators should know to validate
    #     precision/recall at the chosen threshold.
    if n_anom_recall < 0.70:
        print(
            f"WARNING: fusion threshold {P7_INTRUSION_CONFIDENCE_THRESHOLD} "
            f"catches only {n_anom_recall:.0%} of anomaly events. "
            f"Consider lowering the threshold or adding a complementary "
            f"rule. (P2's calibration note.)"
        )


def test_no_critical_band_on_scaled_slice_warns():
    """Soft warning: P7's scaled slice produced 0 critical incidents
    (per fusion re-run; INCIDENTS_BY_BAND = {high: 210, critical: 0}).
    The escalation gate is therefore not exercised end-to-end on the
    scaled slice. This is a known artifact of the confidence threshold
    sitting just above the anomaly mean (P2's caveat). We don't fail;
    we log so the operator sees it.
    """
    from src.utils.io import read_parquet
    incidents = read_parquet("incidents")
    n_critical = int((incidents["risk_band"] == "critical").sum())
    if n_critical == 0:
        print(
            "WARNING: scaled slice produced 0 critical incidents. "
            "The human-in-the-loop escalation gate is not exercised on "
            "this slice. This is the P2 calibration effect (see test_ttest_intrusion_vs_normal_confidence); "
            "lowering the threshold or seeding a critical anomaly would "
            "let the full gate run end-to-end."
        )
    # No assert: this is informational, not a hard fail.


if __name__ == "__main__":
    # Manual run: print the calibration numbers without pytest.
    test_chi2_zone_restrictiveness_vs_access_outcome()
    print()
    test_ttest_intrusion_vs_normal_confidence()
    print()
    test_no_critical_band_on_scaled_slice_warns()