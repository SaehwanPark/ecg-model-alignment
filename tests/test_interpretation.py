"""Tests for Stage 11 Research Interpretation Module.

Verifies:
- Alignment strength classification and threshold boundaries.
- Within-A residual risk gradient assessments.
- Discordance quadrant characterization and relative risk metrics.
- Separation of statistical incremental value from clinical bedside utility.
- Pretraining contamination audit and scientific claim language validation.
- Technical failure rate cataloging and cohort completeness tracking.
- Prespecified vs. post-hoc analytical registry structure.
- External validation justification logic and multi-center recommendations.
- Full synthesis object construction and markdown report generation.
"""

import pytest

from ecg_alignment.analysis import (
  BootstrapConfidenceInterval,
  DiscordanceQuadrant,
  DiscordanceResult,
  GlobalAlignmentResult,
  IncrementalInformationResult,
  NestedModelEvaluation,
  PerformanceComparisonResult,
  PrimaryAnalysisResult,
  QuantileStratumResult,
  StratifiedAnalysisResult,
  StratifiedCategoryResult,
)
from ecg_alignment.interpretation import (
  ALIGNMENT_STRONG_MIN,
  ALIGNMENT_WEAK_MAX,
  MEANINGFUL_GRADIENT_RATIO_MIN,
  AlignmentInterpretation,
  AlignmentStrength,
  AnalysisRegistry,
  CategoryGradientAssessment,
  ClinicalUtilityDistinction,
  ContaminationAudit,
  DiscordanceInterpretationSummary,
  ExternalValidationRecommendation,
  ResearchInterpretationSynthesis,
  TechnicalFailureSummary,
  ValidationRecommendationStatus,
  WithinAGradientsSummary,
  assess_within_a_gradients,
  audit_pretraining_contamination,
  build_analysis_registry,
  classify_alignment_strength,
  evaluate_clinical_utility_distinction,
  generate_research_interpretation_markdown,
  interpret_discordant_groups,
  interpret_global_alignment,
  summarize_technical_failure_rates,
  synthesize_external_validation_recommendation,
  synthesize_research_interpretation,
  verify_scientific_claims_language,
)


# -----------------------------------------------------------------------------
# Unit Tests: Alignment Strength Classification
# -----------------------------------------------------------------------------


def test_classify_alignment_strength_boundaries() -> None:
  """Test alignment strength classification across boundary values."""
  # Weak cases
  strength, narrative = classify_alignment_strength(0.15)
  assert strength == AlignmentStrength.WEAK
  assert "weak alignment" in narrative

  strength, _ = classify_alignment_strength(-0.25)
  assert strength == AlignmentStrength.WEAK

  # Moderate cases
  strength, narrative = classify_alignment_strength(0.30)
  assert strength == AlignmentStrength.MODERATE
  assert "moderate alignment" in narrative

  strength, narrative = classify_alignment_strength(0.512)
  assert strength == AlignmentStrength.MODERATE
  assert "0.512" in narrative

  strength, _ = classify_alignment_strength(0.699)
  assert strength == AlignmentStrength.MODERATE

  # Strong cases
  strength, narrative = classify_alignment_strength(0.70)
  assert strength == AlignmentStrength.STRONG
  assert "strong alignment" in narrative

  strength, narrative = classify_alignment_strength(0.85)
  assert strength == AlignmentStrength.STRONG
  assert "predominantly recapitulates" in narrative


def test_interpret_global_alignment() -> None:
  """Test global alignment interpretation from GlobalAlignmentResult."""
  dummy_res = GlobalAlignmentResult(
    spearman_rho=0.512,
    spearman_pvalue=1e-16,
    pearson_r=0.494,
    pearson_pvalue=1e-15,
    a_grid=(0.0, 10.0, 20.0),
    expected_b_smooth=(0.02, 0.06, 0.12),
    risk_surface_a_bins=(0.0, 10.0, 20.0),
    risk_surface_b_bins=(0.0, 0.05, 0.10),
    risk_surface_matrix=((0.01, 0.03), (0.04, 0.09)),
  )

  interp = interpret_global_alignment(dummy_res)
  assert isinstance(interp, AlignmentInterpretation)
  assert interp.spearman_rho == 0.512
  assert interp.strength == AlignmentStrength.MODERATE
  assert "0.512" in interp.narrative


# -----------------------------------------------------------------------------
# Unit Tests: Within-A Residual Risk Gradients
# -----------------------------------------------------------------------------


def test_assess_within_a_gradients() -> None:
  """Test evaluation of Model B residual risk gradients within Model A strata."""
  q_normal = (
    QuantileStratumResult("Tertile 1", 5000, 91, 0.0182, 0.0, 0.03),
    QuantileStratumResult("Tertile 2", 5000, 156, 0.0312, 0.03, 0.05),
    QuantileStratumResult("Tertile 3", 5000, 260, 0.0520, 0.05, 0.20),
  )
  cat_normal = StratifiedCategoryResult(
    category="Normal",
    n_total=15000,
    n_events=507,
    event_rate=0.0338,
    model_b_mean=0.035,
    model_b_std=0.020,
    model_b_median=0.032,
    model_b_q25=0.019,
    model_b_q75=0.052,
    model_b_auroc=0.7412,
    model_b_auprc=0.1205,
    b_quantiles=q_normal,
  )

  q_infarct = (
    QuantileStratumResult("Tertile 1", 1000, 75, 0.075, 0.0, 0.06),
    QuantileStratumResult("Tertile 2", 1000, 116, 0.116, 0.06, 0.11),
    QuantileStratumResult("Tertile 3", 1000, 177, 0.177, 0.11, 0.35),
  )
  cat_infarct = StratifiedCategoryResult(
    category="Probable Infarction",
    n_total=3000,
    n_events=368,
    event_rate=0.1227,
    model_b_mean=0.105,
    model_b_std=0.060,
    model_b_median=0.098,
    model_b_q25=0.062,
    model_b_q75=0.154,
    model_b_auroc=0.7028,
    model_b_auprc=0.2850,
    b_quantiles=q_infarct,
  )

  summary = assess_within_a_gradients((cat_normal, cat_infarct))
  assert isinstance(summary, WithinAGradientsSummary)
  assert len(summary.category_assessments) == 2
  assert summary.all_categories_meaningful is True

  normal_eval = summary.category_assessments[0]
  assert normal_eval.category_name == "Normal"
  assert normal_eval.gradient_ratio > 2.80
  assert normal_eval.is_meaningful is True
  assert normal_eval.risk_difference_t3_t1 > 0.03

  infarct_eval = summary.category_assessments[1]
  assert infarct_eval.category_name == "Probable Infarction"
  assert infarct_eval.gradient_ratio > 2.30
  assert infarct_eval.is_meaningful is True


# -----------------------------------------------------------------------------
# Unit Tests: Discordance Interpretation
# -----------------------------------------------------------------------------


def test_interpret_discordant_groups() -> None:
  """Test interpretation and clinical characterization of discordance quadrants."""
  quadrants = (
    DiscordanceQuadrant("Q1", "A-low / B-low", 14482, 442, BootstrapConfidenceInterval(0.0305, 0.0278, 0.0333)),
    DiscordanceQuadrant("Q2", "A-low / B-high", 9587, 688, BootstrapConfidenceInterval(0.0718, 0.0667, 0.0769)),
    DiscordanceQuadrant("Q3", "A-high / B-low", 1646, 98, BootstrapConfidenceInterval(0.0595, 0.0482, 0.0712)),
    DiscordanceQuadrant("Q4", "A-high / B-high", 6541, 614, BootstrapConfidenceInterval(0.0939, 0.0868, 0.1010)),
  )
  dummy_discordance = DiscordanceResult(
    a_threshold=15.0,
    b_threshold=0.048,
    threshold_method="median",
    quadrants=quadrants,
    risk_diff_alow_bhigh_vs_alow_blow=BootstrapConfidenceInterval(0.0413, 0.0354, 0.0471),
    risk_ratio_alow_bhigh_vs_alow_blow=BootstrapConfidenceInterval(2.35, 2.10, 2.63),
    risk_diff_ahigh_bhigh_vs_ahigh_blow=BootstrapConfidenceInterval(0.0344, 0.0218, 0.0470),
  )

  summary = interpret_discordant_groups(dummy_discordance)
  assert isinstance(summary, DiscordanceInterpretationSummary)
  assert len(summary.quadrants) == 4
  assert summary.occult_relative_risk == 2.35
  assert summary.occult_risk_difference == 0.0413
  assert "Quadrant 2" in summary.occult_risk_narrative
  assert "Quadrant 3" in summary.pseudo_high_risk_narrative
  assert "occult risk" in summary.quadrants[1].clinical_interpretation.lower()


# -----------------------------------------------------------------------------
# Unit Tests: Statistical vs Clinical Utility Distinction
# -----------------------------------------------------------------------------


def test_evaluate_clinical_utility_distinction() -> None:
  """Test the formal boundary separating statistical value from clinical utility."""
  distinction = evaluate_clinical_utility_distinction()
  assert isinstance(distinction, ClinicalUtilityDistinction)
  assert distinction.lrt_stat > 300.0
  assert distinction.delta_auroc > 0.08
  assert "RESEARCH BOUNDARY" in distinction.formal_distinction_statement
  assert "does NOT equate to clinical bedside utility" in distinction.clinical_utility_caveat
  assert len(distinction.missing_elements_for_deployment) >= 4
  assert any("Decision curve analysis" in s for s in distinction.missing_elements_for_deployment)
  assert any("Prospective clinical trial" in s for s in distinction.missing_elements_for_deployment)


# -----------------------------------------------------------------------------
# Unit Tests: Pretraining Contamination Audit & Claims Linter
# -----------------------------------------------------------------------------


def test_audit_pretraining_contamination() -> None:
  """Test contamination audit records for foundation and traditional models."""
  audit = audit_pretraining_contamination()
  assert isinstance(audit, ContaminationAudit)
  assert audit.has_any_contamination is True
  assert audit.firewall_verified is True
  assert len(audit.records) == 3

  dbeta = audit.records[0]
  assert dbeta.model_name.startswith("D-BETA")
  assert dbeta.has_mimic_contamination is True
  assert dbeta.approved_scientific_label == "In-Domain Representation Probing"
  assert dbeta.prohibited_scientific_label == "Independent External Validation"

  ciis = audit.records[2]
  assert "CIIS" in ciis.model_name
  assert ciis.has_mimic_contamination is False


def test_verify_scientific_claims_language() -> None:
  """Test detection of disallowed external validation claims in research text."""
  clean_text = (
    "This study presents an in-domain representation probe of D-BETA on MIMIC-IV-ECG. "
    "Future work will evaluate on independent external cohorts."
  )
  is_clean, prohibited = verify_scientific_claims_language(clean_text)
  assert is_clean is True
  assert len(prohibited) == 0

  contaminated_text = (
    "We report the definitive external validation of D-BETA on a completely unseen dataset."
  )
  is_clean, prohibited = verify_scientific_claims_language(contaminated_text)
  assert is_clean is False
  assert "external validation" in prohibited
  assert "completely unseen dataset" in prohibited


# -----------------------------------------------------------------------------
# Unit Tests: Technical Failure Rates & Analysis Registry
# -----------------------------------------------------------------------------


def test_summarize_technical_failure_rates() -> None:
  """Test technical failure rate and data completeness tracking."""
  summary = summarize_technical_failure_rates(
    total_waveforms=10000,
    model_a_failures=8,
    model_b_failures=0,
    joint_cohort_size=9992,
  )
  assert isinstance(summary, TechnicalFailureSummary)
  assert summary.model_a_failure_rate == 0.0008
  assert summary.model_b_failure_rate == 0.0
  assert summary.cohort_completeness_rate == 0.9992
  assert "99.92%" in summary.narrative


def test_build_analysis_registry() -> None:
  """Test construction of prespecified vs post-hoc analytical registry."""
  registry = build_analysis_registry()
  assert isinstance(registry, AnalysisRegistry)
  assert registry.n_prespecified >= 8
  assert registry.n_post_hoc >= 5
  assert all(item.classification == "prespecified" for item in registry.prespecified_items)
  assert all(item.classification == "post_hoc" for item in registry.post_hoc_items)


# -----------------------------------------------------------------------------
# Unit Tests: External Validation Recommendation
# -----------------------------------------------------------------------------


def test_synthesize_external_validation_recommendation() -> None:
  """Test external validation decision logic and cohort recommendations."""
  rec = synthesize_external_validation_recommendation()
  assert isinstance(rec, ExternalValidationRecommendation)
  assert rec.status == ValidationRecommendationStatus.JUSTIFIED
  assert len(rec.justification_points) >= 3
  assert any("PTB-XL" in c for c in rec.recommended_cohorts)
  assert any("Decision Curve Analysis" in d for d in rec.recommended_design_elements)


# -----------------------------------------------------------------------------
# End-to-End Synthesis & Markdown Generation Tests
# -----------------------------------------------------------------------------


def test_synthesize_research_interpretation_and_markdown() -> None:
  """Test full research interpretation synthesis and report generation."""
  import polars as pl
  from ecg_alignment.analysis import run_primary_analysis
  from ecg_alignment.cli import simulate_cohort_predictions

  cohort_df = pl.DataFrame({
    "subject_id": list(range(100)),
    "study_id": list(range(100, 200)),
    "split": ["dev"] * 60 + ["val"] * 20 + ["test"] * 20,
  })
  unified_table, _ = simulate_cohort_predictions(cohort_df, seed=42)
  primary_result = run_primary_analysis(unified_table, n_bootstraps=20, random_seed=42)

  # 1. Successful synthesis from empirical result
  synthesis = synthesize_research_interpretation(primary_result=primary_result)
  assert isinstance(synthesis, ResearchInterpretationSynthesis)
  assert synthesis.contamination_audit.has_any_contamination is True
  assert synthesis.external_validation_recommendation.status in (
    ValidationRecommendationStatus.JUSTIFIED,
    ValidationRecommendationStatus.INCONCLUSIVE,
    ValidationRecommendationStatus.NOT_JUSTIFIED,
  )

  markdown = generate_research_interpretation_markdown(synthesis)
  assert isinstance(markdown, str)
  assert "# Stage 11 Research Report" in markdown
  assert "mermaid" in markdown
  assert "Quadrant 2" in markdown
  assert "RESEARCH BOUNDARY" in markdown
  assert "DISCLOSURE AUDIT" in markdown
  assert "Prespecified Components" in markdown
  assert "Post-Hoc & Exploratory Components" in markdown
  assert "External Validation" in markdown
  assert "Stage 11 is complete." in markdown

  # 2. Refuses None primary_result (fails closed against hardcoded static fallbacks)
  with pytest.raises(ValueError, match="requires a valid PrimaryAnalysisResult"):
    # pyright: ignore[reportArgumentType]
    synthesize_research_interpretation(primary_result=None)  # type: ignore
