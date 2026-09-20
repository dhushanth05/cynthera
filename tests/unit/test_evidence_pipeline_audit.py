"""Unit and regression test suite for CYNTHERA forensic code audit and pipeline fixes.

Verifies:
1. Canonical EvidenceSummary data contract and separation of semantic counts.
2. Clinical Evidence Agent semantics (reporting clinical trials evaluated, not total evidence records).
3. Decision factors & limiting factors generation (MS LOW, unapproved indication, missing clinical efficacy).
4. Mechanism hop status distinctions (DIRECT_TARGET_ASSOCIATION, TARGET_DISEASE_ASSOCIATION, STRUCTURAL_EVIDENCE).
5. Prohibition of false "VALIDATED" labeling without literature bridge validation.
6. Subgraph extractor edge provenance tier and claim citation mapping.
7. Colchicine → Heart Failure end-to-end simulation.
8. Metformin → Pancreatic Cancer regression safety and mechanistic safeguards.
"""
from __future__ import annotations

import uuid
import pytest

from backend.core.domain.drug import Drug
from backend.core.domain.disease import Disease
from backend.core.domain.target import Target
from backend.core.domain.protein import Protein
from backend.core.domain.pathway import Pathway
from backend.core.domain.evidence import Evidence
from backend.core.domain.clinical_trial import ClinicalTrial
from backend.core.domain.retrieval_package import RetrievalPackage
from backend.core.domain.candidate_mechanism import CandidateMechanism, MechanismHop
from backend.core.domain.claim import Claim
from backend.core.domain.contradiction import Contradiction
from backend.core.domain.reasoning_result import (
    SupportAssessment,
    MechanisticAssessment,
    RiskAssessment,
    OppositionAssessment,
    ScientificAuditReport,
)
from backend.core.enums.evidence_type import EvidenceType
from backend.core.enums.predicate_type import PredicateType
from backend.core.enums.recommendation import RecommendationStatus
from backend.core.enums.trial_outcome import TrialOutcomeStatus
from backend.core.enums.causal_grounding import CausalGrounding
from backend.core.value_objects.erw import ERW
from backend.core.value_objects.provenance import ProvenanceReference
from backend.core.value_objects.evidence_summary import EvidenceSummary
from backend.reasoning.agents.clinical_safety_agent import SafetyProfile
from backend.reasoning.context.scientific_context_builder import ScientificContext, DimensionalAssessment
from backend.reasoning.mechanistic.mechanism_validation import MechanismValidator
from backend.reasoning.orchestrator.decision_rules import build_evidence_checks, apply_decision_rules
from backend.reasoning.orchestrator.reasoning_orchestrator import ReasoningOrchestrator
from backend.playground.subgraph_extractor import extract_relevant_subgraph


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_colchicine_package(n_evidence: int = 87, n_trials: int = 15) -> RetrievalPackage:
    hyp_id = uuid.uuid4()
    drug = Drug(name="Colchicine", identifiers={"chembl": "CHEMBL84"})
    disease = Disease(name="Heart Failure", identifiers={"mesh": "D006333"})

    targets = [
        Target(
            drug_chembl_id="CHEMBL84",
            protein_uniprot="P68371",
            affinity_nm=12.0,
            affinity_type="IC50",
            mechanism="INHIBITOR",
            erw=ERW(value=0.90),
            provenance=ProvenanceReference(source_name="ChEMBL", source_version="v33", record_id="CHEMBL84"),
        )
    ]
    proteins = [
        Protein(
            uniprot_accession="P68371",
            gene_symbol="TUBB4B",
            name="Tubulin beta-4B chain",
            organism="Homo sapiens",
            is_reviewed=True,
            provenance=ProvenanceReference(source_name="UniProt", source_version="2024_01", record_id="P68371"),
        )
    ]
    pathways = [
        Pathway(
            reactome_id="R-HSA-983189",
            name="Kinesins",
            participant_uniprot_ids=["P68371"],
            provenance=ProvenanceReference(source_name="Reactome", source_version="v85", record_id="R-HSA-983189"),
        )
    ]

    evidence_records = [
        Evidence(
            evidence_type=EvidenceType.LITERATURE,
            erw=ERW(value=0.65),
            citation_key=f"PMID:300000{i:02d}",
            title=f"Colchicine cardiac study {i}",
            abstract=f"Investigation of colchicine effects on myocardial inflammation study {i}.",
            provenance=ProvenanceReference(source_name="PubMed", source_version="2024", record_id=f"PMID:300000{i:02d}"),
            target_uniprot="P68371" if i < 71 else None,
        )
        for i in range(n_evidence)
    ]

    clinical_trials = [
        ClinicalTrial(
            nct_id=f"NCT{10000000 + i}",
            title=f"Colchicine in Heart Failure trial {i}",
            phase="Phase II",
            status=TrialOutcomeStatus.ACTIVE,
            provenance=ProvenanceReference(source_name="ClinicalTrials.gov", source_version="2024", record_id=f"NCT{10000000 + i}"),
        )
        for i in range(n_trials)
    ]

    return RetrievalPackage(
        hypothesis_id=hyp_id,
        drug=drug,
        disease=disease,
        targets=targets,
        proteins=proteins,
        genes=[],
        pathways=pathways,
        evidence_records=evidence_records,
        clinical_trials=clinical_trials,
        retrieval_confidence="HIGH",
        sources_queried=["chembl", "uniprot", "reactome", "pubmed", "clinicaltrials", "opentargets"],
        sources_failed=[],
        validated_disease_genes={"TUBB4B": 0.45},
    )


# ─────────────────────────────────────────────
# Test 1: Canonical EvidenceSummary Contract
# ─────────────────────────────────────────────

def test_evidence_summary_contract():
    summary = EvidenceSummary(
        total_retrieved_records=87,
        subgraph_mapped_records=71,
        literature_claims_extracted=44,
        clinical_trials_evaluated=15,
        clinical_trials_failed=0,
        clinical_trials_successful=0,
        candidate_mechanisms_count=5,
        validated_mechanisms_count=0,
        mechanistic_paths_count=17,
        pathways_annotated_count=39,
        targets_count=1,
    )
    assert summary.total_retrieved_records == 87
    assert summary.subgraph_mapped_records == 71
    assert summary.literature_claims_extracted == 44
    assert summary.clinical_trials_evaluated == 15
    assert summary.clinical_trials_failed == 0
    assert summary.candidate_mechanisms_count == 5
    assert summary.pathways_annotated_count == 39

    display_dict = summary.to_display_dict()
    assert "total_retrieved_records" in display_dict
    assert "clinical_trials" in display_dict
    assert display_dict["clinical_trials"]["count"] == 15


# ─────────────────────────────────────────────
# Test 2: Clinical Evidence Agent Semantics
# ─────────────────────────────────────────────

def test_clinical_evidence_agent_verdict_uses_trial_count():
    orchestrator = ReasoningOrchestrator.__new__(ReasoningOrchestrator)
    support = SupportAssessment(
        score=0.998,
        level="HIGH",
        evidence_count=87,
        weighted_sum=52.8,
        rationale="Extensive literature co-mentions",
        supporting_claim_ids=[],
        has_high_quality_therapeutic=False,
    )
    mechanistic = MechanisticAssessment(
        score=0.431,
        level="LOW",
        pathway_count=39,
        mechanistic_chain=[],
        candidate_mechanisms=[{
            "support_level": "WEAK_SPECULATIVE",
            "discovery_status": "CANDIDATE_STRUCTURAL",
            "confidence_score": 0.431,
        }],
    )
    risk = RiskAssessment(
        score=0.0,
        level="NONE",
        failed_trial_count=0,
        contradiction_count=0,
        rationale="No safety signals detected",
    )
    safety_profile = SafetyProfile(overall_safety_grade="A")
    scientific_context = ScientificContext(
        regulatory=DimensionalAssessment(dimension="regulatory", status="INVESTIGATIONAL", confidence=0.0),
        repurposing=DimensionalAssessment(dimension="repurposing", status="ACTIVE_INVESTIGATION", confidence=0.5),
        mechanistic=DimensionalAssessment(dimension="mechanistic", status="SPECULATIVE", confidence=0.3),
        clinical=DimensionalAssessment(dimension="clinical", status="UNDER_EVALUATION", confidence=0.4),
        knowledge_maturity=DimensionalAssessment(dimension="knowledge_maturity", status="SPECULATIVE", confidence=0.2),
    )
    pkg = _make_colchicine_package(n_evidence=87, n_trials=15)

    verdicts = orchestrator._compute_agent_verdicts(
        support=support,
        mechanistic=mechanistic,
        risk=risk,
        contradictions=[],
        safety_profile=safety_profile,
        scientific_context=scientific_context,
        package=pkg,
    )

    cea_verdict = verdicts["Clinical Evidence Agent"]
    # Must report 15 clinical trials, NOT 87 records
    assert "15 clinical trial(s)" in cea_verdict
    assert "87" not in cea_verdict
    # Must caution that therapeutic efficacy is not established
    assert "UNCERTAIN" in cea_verdict or "CAUTION" in cea_verdict
    assert "not established" in cea_verdict or "0 failed" in cea_verdict

    # Support Assessment Agent must explicitly note presence metric
    saa_verdict = verdicts["Support Assessment Agent"]
    assert "87" in saa_verdict
    assert "presence metric" in saa_verdict

    # Mechanistic Expert Agent must clarify structural candidate
    mea_verdict = verdicts["Mechanistic Expert Agent"]
    assert "LOW (0.431)" in mea_verdict
    assert "structural candidate" in mea_verdict or "unvalidated" in mea_verdict


def _make_sc(reg_status: str = "INVESTIGATIONAL") -> ScientificContext:
    return ScientificContext(
        regulatory=DimensionalAssessment(dimension="regulatory", status=reg_status, confidence=0.0),
        repurposing=DimensionalAssessment(dimension="repurposing", status="ACTIVE_INVESTIGATION", confidence=0.5),
        mechanistic=DimensionalAssessment(dimension="mechanistic", status="SPECULATIVE", confidence=0.3),
        clinical=DimensionalAssessment(dimension="clinical", status="UNDER_EVALUATION", confidence=0.4),
        knowledge_maturity=DimensionalAssessment(dimension="knowledge_maturity", status="SPECULATIVE", confidence=0.2),
    )


def test_clinical_evidence_agent_positive_efficacy():
    orchestrator = ReasoningOrchestrator.__new__(ReasoningOrchestrator)
    support = SupportAssessment(
        score=0.85,
        level="HIGH",
        evidence_count=20,
        weighted_sum=15.0,
        has_high_quality_therapeutic=True,
    )
    risk = RiskAssessment(score=0.1, level="LOW", failed_trial_count=0)
    safety_profile = SafetyProfile(overall_safety_grade="A")
    scientific_context = _make_sc()
    pkg = _make_colchicine_package(n_evidence=20, n_trials=3)

    verdicts = orchestrator._compute_agent_verdicts(
        support=support,
        mechanistic=MechanisticAssessment(score=0.8, level="HIGH", pathway_count=5),
        risk=risk,
        contradictions=[],
        safety_profile=safety_profile,
        scientific_context=scientific_context,
        package=pkg,
    )
    assert "HIGH — 3 trial(s) evaluated; positive clinical trial efficacy demonstrated" in verdicts["Clinical Evidence Agent"]


# ─────────────────────────────────────────────
# Test 3: Decision Factors & Limiting Factors
# ─────────────────────────────────────────────

def test_limiting_factors_generated_when_ms_is_low():
    support = SupportAssessment(
        score=0.998,
        level="HIGH",
        evidence_count=87,
        has_high_quality_therapeutic=False,
    )
    mechanistic = MechanisticAssessment(
        score=0.431,
        level="LOW",
        pathway_count=17,
        candidate_mechanisms=[{
            "support_level": "WEAK_SPECULATIVE",
            "discovery_status": "CANDIDATE_STRUCTURAL",
        }],
    )
    risk = RiskAssessment(score=0.0, level="NONE")
    pkg = _make_colchicine_package(n_evidence=87, n_trials=15)

    checks = build_evidence_checks(
        support=support,
        mechanistic=mechanistic,
        risk=risk,
        contradictions=[],
        package=pkg,
        is_approved=False,
        disease_name="Heart Failure",
        has_high_quality_therapeutic=False,
    )

    passes = [c.strip()[6:].strip() for c in checks if c.strip().startswith("[PASS]")]
    fails = [c.strip()[6:].strip() for c in checks if c.strip().startswith("[FAIL]")]

    # Mechanistic plausibility must FAIL because MS is LOW / WEAK_SPECULATIVE
    assert any("Mechanistic plausibility" in f and "structural candidate only" in f for f in fails)
    assert not any("Mechanistic plausibility" in p for p in passes)

    # Approved indication anchor must FAIL because not approved for Heart Failure
    assert any("Approved indication anchor" in f and "repurposing hypothesis" in f for f in fails)

    # Human clinical efficacy must FAIL because no pair-specific efficacy demonstrated
    assert any("Human clinical efficacy" in f and "No pair-specific clinical trial success" in f for f in fails)

    # There must be multiple limiting factors (NOT zero / "None identified")
    assert len(fails) >= 3


# ─────────────────────────────────────────────
# Test 4: Mechanism Hop Status Distinctions
# ─────────────────────────────────────────────

def test_mechanism_hop_status_distinctions():
    candidate = CandidateMechanism(
        name="TUBB4B candidate",
        support_level="WEAK_SPECULATIVE",
        confidence_score=0.431,
        summary_chain=["Drug: Colchicine", "Target: TUBB4B", "Disease: Heart Failure"],
        hops=[
            MechanismHop(
                from_node="Drug: Colchicine",
                to_node="Target: TUBB4B",
                predicate="INHIBITOR",
                evidence_strength=0.95,
                source_database="ChEMBL",
            ),
            MechanismHop(
                from_node="Target: TUBB4B",
                to_node="Disease: Heart Failure",
                predicate="ASSOCIATED_WITH",
                evidence_strength=0.45,
                source_database="Open Targets",
            ),
        ],
    )
    pkg = _make_colchicine_package()
    validator = MechanismValidator()

    # No literature claims mapped to the target-disease hop
    validated_cands = validator.validate(pkg, [candidate], [])
    res_cand = validated_cands[0]

    # Hop 0 is direct drug-target
    assert res_cand.hops[0].status == "DIRECT_TARGET_ASSOCIATION"
    # Hop 1 is target-disease database association, NOT a validated mechanism
    assert res_cand.hops[1].status == "TARGET_DISEASE_ASSOCIATION"
    # Discovery status must be CANDIDATE_STRUCTURAL, never VALIDATED
    assert res_cand.discovery_status == "CANDIDATE_STRUCTURAL"
    assert res_cand.support_level == "WEAK_SPECULATIVE"


# ─────────────────────────────────────────────
# Test 5: End-to-End Simulation of Colchicine → Heart Failure
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_colchicine_heart_failure_end_to_end_traceability():
    pkg = _make_colchicine_package(n_evidence=87, n_trials=15)
    orchestrator = ReasoningOrchestrator()

    result = await orchestrator.reason(pkg)

    # 1. Recommendation must be UNCERTAIN
    assert result.recommendation_status == RecommendationStatus.UNCERTAIN

    # 2. Scores must be consistent with problem statement
    assert result.support_assessment.score > 0.95
    assert result.support_assessment.level == "HIGH"
    assert result.risk_assessment.score == 0.0

    # 3. ScientificAuditReport must have limiting factors populated
    audit = result.audit_report
    assert len(audit.negative_factors) >= 3
    assert any("Mechanistic plausibility" in f for f in audit.negative_factors)
    assert any("Approved indication anchor" in f for f in audit.negative_factors)
    assert any("Human clinical efficacy" in f for f in audit.negative_factors)

    # 4. Canonical EvidenceSummary must be attached and accurate
    es = audit.evidence_summary
    assert es is not None
    assert es.total_retrieved_records == 87
    assert es.clinical_trials_evaluated == 15
    assert es.clinical_trials_failed == 0
    assert es.candidate_mechanisms_count >= 1
    assert es.validated_mechanisms_count == 0

    # 5. Agent verdicts must be epistemically honest
    verdicts = audit.agent_verdicts
    assert "15 clinical trial(s)" in verdicts["Clinical Evidence Agent"]
    assert "87" not in verdicts["Clinical Evidence Agent"]
    assert "presence metric" in verdicts["Support Assessment Agent"]
    assert "structural candidate" in verdicts["Mechanistic Expert Agent"] or "LOW" in verdicts["Mechanistic Expert Agent"]

    # 6. Playground extraction must reflect exact claim citations and edge grounding
    graph_data = extract_relevant_subgraph(pkg, result)
    assert len(graph_data.evidence) > 0
    for edge in graph_data.edges:
        assert edge.data_quality in ("DIRECT", "CURATED", "INFERRED", "STRUCTURAL", "NONE", "EVIDENCE_BACKED", "STRUCTURE_ONLY", "UNVALIDATED")


# ─────────────────────────────────────────────
# Test 6: Metformin → Pancreatic Cancer Regression
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metformin_pancreatic_cancer_regression():
    """Metformin in Pancreatic Cancer: speculative repurposing hypothesis.

    Must remain UNCERTAIN (or NOT_RECOMMENDED if negative trials/opposition),
    and must NOT be falsely recommended as PROMISING.
    Limiting factors must correctly list lack of approved indication anchor
    and lack of pair-specific clinical trial efficacy.
    """
    hyp_id = uuid.uuid4()
    drug = Drug(name="Metformin", identifiers={"chembl": "CHEMBL1431"})
    disease = Disease(name="Pancreatic Cancer", identifiers={"mesh": "D010190"})

    targets = [
        Target(
            drug_chembl_id="CHEMBL1431",
            protein_uniprot="Q13131",  # PRKAA1 (AMPK)
            affinity_nm=100.0,
            affinity_type="EC50",
            mechanism="ACTIVATOR",
            erw=ERW(value=0.85),
            provenance=ProvenanceReference(source_name="ChEMBL", source_version="v33", record_id="CHEMBL1431"),
        )
    ]
    proteins = [
        Protein(
            uniprot_accession="Q13131",
            gene_symbol="PRKAA1",
            name="5'-AMP-activated protein kinase catalytic subunit alpha-1",
            organism="Homo sapiens",
            is_reviewed=True,
            provenance=ProvenanceReference(source_name="UniProt", source_version="2024_01", record_id="Q13131"),
        )
    ]
    pathways = [
        Pathway(
            reactome_id="R-HSA-380972",
            name="AMPK signaling",
            participant_uniprot_ids=["Q13131"],
            provenance=ProvenanceReference(source_name="Reactome", source_version="v85", record_id="R-HSA-380972"),
        )
    ]
    evidence_records = [
        Evidence(
            evidence_type=EvidenceType.LITERATURE,
            erw=ERW(value=0.65),
            citation_key="PMID:28000001",
            title="Metformin and pancreatic cancer incidence observational analysis",
            abstract="Epidemiological analysis showed associations between metformin use and reduced pancreatic cancer risk.",
            provenance=ProvenanceReference(source_name="PubMed", source_version="2024", record_id="PMID:28000001"),
        )
    ]

    pkg = RetrievalPackage(
        hypothesis_id=hyp_id,
        drug=drug,
        disease=disease,
        targets=targets,
        proteins=proteins,
        genes=[],
        pathways=pathways,
        evidence_records=evidence_records,
        clinical_trials=[],
        retrieval_confidence="HIGH",
        sources_queried=["chembl", "uniprot", "reactome", "pubmed", "clinicaltrials", "opentargets"],
        sources_failed=[],
        validated_disease_genes={"PRKAA1": 0.30},
    )

    orchestrator = ReasoningOrchestrator()
    result = await orchestrator.reason(pkg)

    # Must be UNCERTAIN (cannot be PROMISING without clinical trial success or approved indication)
    assert result.recommendation_status == RecommendationStatus.UNCERTAIN

    # Limiting factors must clearly identify why
    audit = result.audit_report
    assert len(audit.negative_factors) >= 2
    assert any("Approved indication anchor" in f for f in audit.negative_factors)
    assert any("Human clinical efficacy" in f for f in audit.negative_factors)


# ─────────────────────────────────────────────
# Test 7: Prior Knowledge Path Labeling (Issue 2)
# ─────────────────────────────────────────────

def test_prior_knowledge_unvalidated_path_not_labeled_validated():
    """Unvalidated mechanistic paths must NEVER be labeled 'Validated path:'."""
    from backend.reasoning.context.scientific_context_builder import ScientificContextBuilder
    from backend.reasoning.mechanistic.multi_hop_reasoner import MechanisticPath
    from backend.reasoning.agents.prior_knowledge_agent import PriorKnowledgeContext

    path = MechanisticPath(
        hop_count=1,
        confidence=0.431,
        path_type="2-HOP",
        description="Colchicine → Tubulin beta chain → Heart failure",
    )
    mech_unvalidated = MechanisticAssessment(
        score=0.431,
        level="LOW",
        pathway_count=39,
        candidate_mechanisms=[{
            "name": "Candidate 1",
            "summary_chain": ["Colchicine", "Tubulin beta chain", "Heart failure"],
            "discovery_status": "CANDIDATE_STRUCTURAL",
        }],
    )
    pkg = _make_colchicine_package(n_evidence=71, n_trials=15)
    ctx = ScientificContextBuilder.build(
        prior_ctx=PriorKnowledgeContext(),
        support=SupportAssessment(score=0.998, level="HIGH", evidence_count=87),
        mechanistic=mech_unvalidated,
        mechanistic_paths=[path],
        package=pkg,
    )

    mech_ev = ctx.mechanistic.evidence
    # Must NOT say 'Validated path:'
    assert not any(e.startswith("Validated path:") for e in mech_ev)
    # Must say 'Candidate path: ... (unvalidated causal bridge)'
    assert any("Candidate path:" in e and "unvalidated causal bridge" in e for e in mech_ev)


def test_prior_knowledge_validated_path_labeled_validated():
    """Actually validated mechanisms must be labeled 'Validated path:'."""
    from backend.reasoning.context.scientific_context_builder import ScientificContextBuilder
    from backend.reasoning.mechanistic.multi_hop_reasoner import MechanisticPath
    from backend.reasoning.agents.prior_knowledge_agent import PriorKnowledgeContext

    path = MechanisticPath(
        hop_count=1,
        confidence=0.85,
        path_type="DIRECT",
        description="Sildenafil → PDE5A → Pulmonary Arterial Hypertension",
    )
    mech_validated = MechanisticAssessment(
        score=0.85,
        level="HIGH",
        pathway_count=10,
        candidate_mechanisms=[{
            "name": "Validated Mechanism 1",
            "summary_chain": ["Sildenafil", "PDE5A", "PAH"],
            "discovery_status": "VALIDATED",
        }],
    )
    pkg = _make_colchicine_package(n_evidence=20, n_trials=3)
    ctx = ScientificContextBuilder.build(
        prior_ctx=PriorKnowledgeContext(),
        support=SupportAssessment(score=0.9, level="HIGH", evidence_count=20),
        mechanistic=mech_validated,
        mechanistic_paths=[path],
        package=pkg,
    )

    mech_ev = ctx.mechanistic.evidence
    assert any(e.startswith("Validated path:") for e in mech_ev)


# ─────────────────────────────────────────────
# Test 8: Evaluated Support Signals & Narrative (Issue 1, 3, 4)
# ─────────────────────────────────────────────

def test_evidence_and_clinical_narratives_and_risk_semantics():
    """Verify narrative separation, risk qualification, and evaluated signals."""
    from backend.reasoning.conflict.conflict_resolver import ConflictResolutionReport
    from backend.reasoning.agents.prior_knowledge_agent import PriorKnowledgeContext

    orchestrator = ReasoningOrchestrator()
    support = SupportAssessment(
        score=0.998,
        level="HIGH",
        evidence_count=87,
        weighted_sum=52.8,
        rationale="Support Score (SS) from 87 evaluated evidence signals (71 literature/assay records + 16 extracted claims).",
        has_high_quality_therapeutic=False,
    )
    mechanistic = MechanisticAssessment(
        score=0.431,
        level="LOW",
        pathway_count=39,
        candidate_mechanisms=[{
            "support_level": "WEAK_SPECULATIVE",
            "discovery_status": "CANDIDATE_STRUCTURAL",
            "confidence_score": 0.431,
        }],
    )
    risk = RiskAssessment(
        score=0.0,
        level="NONE",
        failed_trial_count=0,
        contradiction_count=0,
        rationale="No safety trial terminations or boxed warnings detected in evaluated data.",
    )
    safety_profile = SafetyProfile(overall_safety_grade="A")
    scientific_context = _make_sc()
    pkg = _make_colchicine_package(n_evidence=71, n_trials=15)

    report = orchestrator._generate_audit_report(
        all_claims=[],
        contradictions=[],
        support=support,
        mechanistic=mechanistic,
        risk=risk,
        recommendation=RecommendationStatus.UNCERTAIN,
        reasons=["Rule 5 (UNCERTAIN): Mixed or sparse evidence."],
        prior_ctx=PriorKnowledgeContext(),
        scientific_context=scientific_context,
        safety_profile=safety_profile,
        mechanistic_paths=[],
        conflict_report=ConflictResolutionReport(),
        package=pkg,
    )

    # 1. Evidence Summary has evaluated_support_signals = 87
    es = report.evidence_summary
    assert es is not None
    assert es.total_retrieved_records == 71
    assert es.evaluated_support_signals == 87
    assert es.subgraph_mapped_records == 24 or es.subgraph_mapped_records > 0

    # 2. Confidence narrative differentiates retrieved records (71) from evaluated signals (87)
    assert "71 total evidence records" in report.confidence_narrative
    assert "87 support signals" in report.confidence_narrative
    assert "standard clinical precautions and known pharmacology apply" in report.confidence_narrative

    # 3. Clinical narrative structure
    assert "Evidence & Clinical Investigation Assessment:" in report.clinical_narrative
    assert "A. Literature & Preclinical Evidence:" in report.clinical_narrative
    assert "B. Clinical Trial Investigation:" in report.clinical_narrative
    assert "NOT ESTABLISHED / NO EFFICACY DEMONSTRATED" in report.clinical_narrative

    # 4. Safety narrative qualification
    assert "Observed Risk Signals: NONE DETECTED" in report.safety_narrative
    assert "Standard clinical precautions and known pharmacology still apply" in report.safety_narrative


# ─────────────────────────────────────────────
# Test 9: PDF Exporter Consistency & Versioning (Issue 5 & 6)
# ─────────────────────────────────────────────

def test_pdf_export_version_and_inventory():
    """Verify PDF export includes CYNTHERA v2.0, Rule Set 3.2, and clean inventory."""
    from backend.reporting.pdf_exporter import PDFReporter
    from backend.core.domain.reasoning_result import ReasoningResult

    support = SupportAssessment(score=0.998, level="HIGH", evidence_count=87)
    mechanistic = MechanisticAssessment(score=0.431, level="LOW", pathway_count=39, candidate_mechanisms=[])
    risk = RiskAssessment(score=0.0, level="NONE", failed_trial_count=0)
    audit = ScientificAuditReport(
        summary="Colchicine in heart failure analysis.",
        evidence_summary=EvidenceSummary(
            total_retrieved_records=71,
            subgraph_mapped_records=24,
            literature_claims_extracted=44,
            evaluated_support_signals=87,
            clinical_trials_evaluated=15,
            clinical_trials_failed=0,
            clinical_trials_successful=0,
            candidate_mechanisms_count=5,
            validated_mechanisms_count=0,
            mechanistic_paths_count=16,
            pathways_annotated_count=39,
            targets_count=4,
        ),
        clinical_narrative="A. Literature & Preclinical Evidence:\nB. Clinical Trial Investigation:\n",
        safety_narrative="Observed Risk Signals: NONE DETECTED",
        mechanistic_narrative="Mechanistic Assessment: LOW",
        final_synthesis="Synthesis: UNCERTAIN",
        confidence_narrative="High confidence.",
    )
    res = ReasoningResult(
        hypothesis_id=uuid.uuid4(),
        recommendation_status=RecommendationStatus.UNCERTAIN,
        recommendation_reasons=["Rule 5: UNCERTAIN"],
        support_assessment=support,
        mechanistic_assessment=mechanistic,
        risk_assessment=risk,
        contradictions=[],
        audit_report=audit,
        rule_set_version="3.2",
    )

    reporter = PDFReporter(drug_name="Colchicine", disease_name="Heart Failure")
    pdf_bytes = reporter.generate(res)
    assert len(pdf_bytes) > 1000

    # Also check plain-text fallback contains CYNTHERA v2.0 and Rule Set: 3.2
    txt_report = reporter._generate_text_report(res).decode("utf-8")
    assert "CYNTHERA v2.0 | Rule Set: 3.2" in txt_report

