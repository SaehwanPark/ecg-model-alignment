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
  GlobalPerformanceResult,
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
  assert "a-low / b-high" in summary.quadrants[1].clinical_interpretation.lower()
  assert "A < 15.0" in summary.quadrants[1].description
  assert "B >= 0.048" in summary.quadrants[1].description
  assert "A >= 15.0" in summary.quadrants[2].description
  assert "B < 0.048" in summary.quadrants[2].description


# -----------------------------------------------------------------------------
# Unit Tests: Statistical vs Clinical Utility Distinction
# -----------------------------------------------------------------------------


def test_evaluate_clinical_utility_distinction() -> None:
  """Test the formal boundary separating statistical value from clinical utility."""
  # 1. Default (no empirical input)
  distinction_default = evaluate_clinical_utility_distinction()
  assert isinstance(distinction_default, ClinicalUtilityDistinction)
  assert distinction_default.lrt_stat == 0.0
  assert "RESEARCH BOUNDARY" in distinction_default.formal_distinction_statement
  assert "does NOT equate to clinical bedside utility" in distinction_default.clinical_utility_caveat
  assert len(distinction_default.missing_elements_for_deployment) >= 4

  # 2. With positive empirical inputs
  dummy_inc = IncrementalInformationResult(
    model_a_only=NestedModelEvaluation("Model A", "y ~ splines(A)", -120.0, 246.0, 252.0, 0.50, 0.70, 0.08),
    model_b_only=NestedModelEvaluation("Model B", "y ~ B", -110.0, 224.0, 230.0, 0.45, 0.76, 0.07),
    model_combined=NestedModelEvaluation("Model A + B", "y ~ splines(A) + B", -95.0, 198.0, 208.0, 0.40, 0.78, 0.06),
    lrt_statistic=350.0,
    lrt_degrees_of_freedom=1,
    lrt_pvalue=1e-20,
    auroc_improvement=BootstrapConfidenceInterval(0.08, 0.06, 0.10),
    brier_improvement=BootstrapConfidenceInterval(0.02, 0.01, 0.03),
    held_out_loss_reduction=0.10,
    held_out_loss_reduction_ci=BootstrapConfidenceInterval(0.10, 0.08, 0.12),
  )
  dummy_comp = PerformanceComparisonResult(
    delta_auroc=BootstrapConfidenceInterval(0.08, 0.06, 0.10),
    delta_auprc=BootstrapConfidenceInterval(0.05, 0.03, 0.07),
    delta_brier=BootstrapConfidenceInterval(0.02, 0.01, 0.03),
    p_value_auroc_diff=1e-15,
  )
  distinction_emp = evaluate_clinical_utility_distinction(dummy_inc, dummy_comp)
  assert distinction_emp.lrt_stat == 350.0
  assert distinction_emp.delta_auroc == 0.08
  assert "p < 10^-15" in distinction_emp.statistical_summary

  # 3. Test that incremental metric takes priority over marginal metric
  inc_diff = IncrementalInformationResult(
    model_a_only=NestedModelEvaluation("Model A", "y ~ splines(A)", -120.0, 246.0, 252.0, 0.50, 0.70, 0.08),
    model_b_only=NestedModelEvaluation("Model B", "y ~ B", -110.0, 224.0, 230.0, 0.45, 0.76, 0.07),
    model_combined=NestedModelEvaluation("Model A + B", "y ~ splines(A) + B", -95.0, 198.0, 208.0, 0.40, 0.78, 0.06),
    lrt_statistic=12.0,
    lrt_degrees_of_freedom=1,
    lrt_pvalue=0.08,  # Non-significant LRT should not block affirmative when CI lower > 0
    auroc_improvement=BootstrapConfidenceInterval(0.04, 0.01, 0.07),  # Strictly positive incremental
    brier_improvement=BootstrapConfidenceInterval(0.01, 0.005, 0.02),
    held_out_loss_reduction=0.03,
  )
  comp_marginal = PerformanceComparisonResult(
    delta_auroc=BootstrapConfidenceInterval(0.12, 0.09, 0.15),  # Marginal B vs A
    delta_auprc=BootstrapConfidenceInterval(0.08, 0.05, 0.11),
    delta_brier=BootstrapConfidenceInterval(0.03, 0.01, 0.05),
    p_value_auroc_diff=1e-5,
  )
  dist_prio = evaluate_clinical_utility_distinction(inc_diff, comp_marginal)
  assert dist_prio.delta_auroc == 0.04  # Incremental AUROC, not marginal 0.12
  assert "Incremental Delta AUROC = 0.0400" in dist_prio.statistical_summary


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
  # 1. No input -> INCONCLUSIVE
  rec_none = synthesize_external_validation_recommendation()
  assert isinstance(rec_none, ExternalValidationRecommendation)
  assert rec_none.status == ValidationRecommendationStatus.INCONCLUSIVE
  assert len(rec_none.justification_points) >= 2

  # 2. Positive empirical input -> JUSTIFIED
  dummy_quadrants = (
    DiscordanceQuadrant("Q1", "A-low / B-low", 100, 3, BootstrapConfidenceInterval(0.03, 0.01, 0.05)),
    DiscordanceQuadrant("Q2", "A-low / B-high", 100, 10, BootstrapConfidenceInterval(0.10, 0.06, 0.14)),
    DiscordanceQuadrant("Q3", "A-high / B-low", 100, 4, BootstrapConfidenceInterval(0.04, 0.02, 0.06)),
    DiscordanceQuadrant("Q4", "A-high / B-high", 100, 12, BootstrapConfidenceInterval(0.12, 0.08, 0.16)),
  )
  dummy_disc = DiscordanceResult(
    a_threshold=15.0,
    b_threshold=0.05,
    threshold_method="median",
    quadrants=dummy_quadrants,
    risk_diff_alow_bhigh_vs_alow_blow=BootstrapConfidenceInterval(0.07, 0.02, 0.12),
    risk_ratio_alow_bhigh_vs_alow_blow=BootstrapConfidenceInterval(3.33, 1.50, 6.00),
    risk_diff_ahigh_bhigh_vs_ahigh_blow=BootstrapConfidenceInterval(0.08, 0.03, 0.13),
  )
  dummy_primary = PrimaryAnalysisResult(
    n_patients=400,
    n_events=29,
    event_rate=0.0725,
    alignment=GlobalAlignmentResult(0.50, 1e-10, 0.48, 1e-9, (0.0,), (0.0,), (0.0,), (0.0,), ((0.0,),)),
    stratified=StratifiedAnalysisResult((), overall_event_rate=0.0725),
    discordance=dummy_disc,
    incremental=IncrementalInformationResult(
      model_a_only=NestedModelEvaluation("Model A", "y ~ splines(A)", -120.0, 246.0, 252.0, 0.50, 0.70, 0.08),
      model_b_only=NestedModelEvaluation("Model B", "y ~ B", -110.0, 224.0, 230.0, 0.45, 0.76, 0.07),
      model_combined=NestedModelEvaluation("Model A + B", "y ~ splines(A) + B", -95.0, 198.0, 208.0, 0.40, 0.78, 0.06),
      lrt_statistic=50.0,
      lrt_degrees_of_freedom=1,
      lrt_pvalue=1e-12,
      auroc_improvement=BootstrapConfidenceInterval(0.08, 0.04, 0.12),
      brier_improvement=BootstrapConfidenceInterval(0.02, 0.01, 0.03),
      held_out_loss_reduction=0.10,
    ),
    performance_a=GlobalPerformanceResult("Model A", 400, 29, 0.0725, BootstrapConfidenceInterval(0.70, 0.65, 0.75), BootstrapConfidenceInterval(0.20, 0.15, 0.25), BootstrapConfidenceInterval(0.08, 0.06, 0.10), None, 1.0, 0.0),
    performance_b=GlobalPerformanceResult("Model B", 400, 29, 0.0725, BootstrapConfidenceInterval(0.78, 0.73, 0.83), BootstrapConfidenceInterval(0.25, 0.20, 0.30), BootstrapConfidenceInterval(0.06, 0.04, 0.08), None, 1.0, 0.0),
    comparison=PerformanceComparisonResult(BootstrapConfidenceInterval(0.08, 0.04, 0.12), BootstrapConfidenceInterval(0.05, 0.01, 0.09), BootstrapConfidenceInterval(0.02, 0.01, 0.03), 1e-5),
  )
  rec_pos = synthesize_external_validation_recommendation(dummy_primary)
  assert rec_pos.status == ValidationRecommendationStatus.JUSTIFIED
  assert any("PTB-XL" in c for c in rec_pos.recommended_cohorts)
  assert any("Decision Curve Analysis" in d for d in rec_pos.recommended_design_elements)


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

  markdown = generate_research_interpretation_markdown(synthesis, data_mode="real")
  assert isinstance(markdown, str)
  assert "# Stage 11 Research Report" in markdown
  assert "> **Data Source:** REAL MIMIC-IV-ECG predictions" in markdown
  assert "mermaid" in markdown
  assert f"rho = {synthesis.alignment.spearman_rho:.3f}" in markdown
  assert f"Occult Risk RR = {synthesis.discordance.occult_relative_risk:.2f}x" in markdown
  assert "Quadrant 2" in markdown
  assert "RESEARCH BOUNDARY" in markdown
  assert "DISCLOSURE AUDIT" in markdown
  assert "Prespecified Components" in markdown
  assert "Post-Hoc & Exploratory Components" in markdown
  assert "External Validation" in markdown
  assert "Stage 11 is complete." in markdown

  # 2. Simulation mode provenance header
  markdown_sim = generate_research_interpretation_markdown(synthesis, data_mode="simulation")
  assert "> **Data Source:** SIMULATION — NOT EMPIRICAL RESULTS" in markdown_sim

  # 3. Refuses None primary_result (fails closed against hardcoded static fallbacks)
  with pytest.raises(ValueError, match="requires a valid PrimaryAnalysisResult"):
    # pyright: ignore[reportArgumentType]
    synthesize_research_interpretation(primary_result=None)  # type: ignore


def test_simulation_negative_control_outputs_no_false_affirmative_claims() -> None:
  """Test negative-control property: simulated null data must not generate false affirmative claims."""
  import polars as pl
  from ecg_alignment.analysis import generate_primary_results_markdown, run_primary_analysis
  from ecg_alignment.cli import simulate_cohort_predictions

  cohort_df = pl.DataFrame({
    "subject_id": list(range(200)),
    "study_id": list(range(100, 300)),
    "split": ["dev"] * 120 + ["val"] * 40 + ["test"] * 40,
  })
  unified_table, _ = simulate_cohort_predictions(cohort_df, seed=42)
  primary_result = run_primary_analysis(unified_table, n_bootstraps=20, random_seed=42)
  prim_md = generate_primary_results_markdown(primary_result, data_mode="simulation")

  synthesis = synthesize_research_interpretation(primary_result=primary_result)
  report_md = generate_research_interpretation_markdown(synthesis, data_mode="simulation")

  # Assert no false affirmative claims
  assert "H4 (Incremental Information): Confirmed" not in prim_md
  assert "STRONGLY JUSTIFIED" not in report_md
  assert "undeniable" not in report_md
  assert "large effect size" not in report_md
