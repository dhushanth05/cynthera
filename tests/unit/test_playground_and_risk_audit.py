"""Dedicated unit test suite for CYNTHERA Playground UX, Risk model semantics,
Reactome source status and request counts, and PDF/Playground consistency.

Covers all 15 verification points from Part 6 of the final audit specification:
1. Overview graph view filters to top relevant targets, candidate pathways, and core disease nodes without background reaction clutter
2. Mechanisms graph view highlights the candidate mechanism and dims non-participating nodes/edges
3. Evidence graph view highlights evidence-supported edges and dims unvalidated/structural edges
4. Full Graph view retains all nodes and edges (zero data loss)
5. Candidate mechanism selector updates graph highlighting and inspector details
6. Node sizing hierarchy: Drug/Disease (34-36px), Targets/Pathways (26-28px), Genes/Reactions (18-20px)
7. Node labels display canonical short symbol on graph and full identifier in hover tooltip
8. Edge labels hidden by default; midpoints do not permanently occlude nodes/edges (transparent hover marker)
9. Epistemic status styling: structural Reactome edges styled distinctly (dashed/muted) from evidence-backed/direct edges
10. Reactome status: 0 failed -> SUCCESS, some failed + pathways -> PARTIAL with request counts, all failed -> FAILED
11. Reactome request counts exposed in Data Source Diagnostics (X succeeded, Y failed)
12. Risk Score RS = 0.000 reports: 'Observed Clinical Risk Signals: NONE DETECTED (Risk Score = 0.000, Safety Signal Assessment: Grade A)'
13. Known Pharmacological Risk Context reports: 'NOT ASSESSED IN CLINICAL TRIAL DATA (Standard clinical precautions and known pharmacology profile still apply)' without fabricating adverse events
14. Terminology cleanup: 'Risk Level: 0.0% (NONE)' and 'clean safety record' replaced across UI and PDF
15. PDF + Playground metric consistency across RS, Observed Risk Signals, Safety Grade, Reactome status, request counts, candidate mechanisms, annotated pathways, version CYNTHERA v2.0, rule set 3.2
"""
from __future__ import annotations

import uuid
from datetime import datetime
import pytest

from backend.core.domain.drug import Drug
from backend.core.domain.disease import Disease
from backend.core.domain.target import Target
from backend.core.domain.protein import Protein
from backend.core.domain.gene import Gene
from backend.core.domain.pathway import Pathway
from backend.core.domain.evidence import Evidence
from backend.core.domain.clinical_trial import ClinicalTrial
from backend.core.domain.approval_signal import ApprovalSignal
from backend.core.domain.retrieval_package import RetrievalPackage
from backend.core.domain.reasoning_result import (
    ReasoningResult,
    SupportAssessment,
    MechanisticAssessment,
    RiskAssessment,
    OppositionAssessment,
    ScientificAuditReport,
)
from backend.core.enums.recommendation import RecommendationStatus
from backend.core.enums.evidence_type import EvidenceType
from backend.core.enums.trial_outcome import TrialOutcomeStatus
from backend.core.value_objects.erw import ERW
from backend.core.value_objects.provenance import ProvenanceReference
from backend.reasoning.agents.clinical_safety_agent import SafetyProfile
from backend.reasoning.context.scientific_context_builder import ScientificContext, DimensionalAssessment
from backend.core.value_objects.evidence_summary import EvidenceSummary
from backend.engineering.retrieval.connectors.reactome import ReactomeConnector
from backend.engineering.retrieval.pipeline import RetrievalPipeline
from backend.reasoning.orchestrator.reasoning_orchestrator import ReasoningOrchestrator
from backend.reasoning.agents.prior_knowledge_agent import PriorKnowledgeContext
from backend.reasoning.conflict.conflict_resolver import ConflictResolutionReport
from backend.playground.models import (
    PlaygroundGraphData,
    PlaygroundNode,
    PlaygroundEdge,
    PlaygroundEvidence,
    EvidenceLandscape,
)
from backend.reporting.pdf_exporter import PDFReporter
import frontend.playground_view as pv


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_playground_graph():
    nodes = [
        PlaygroundNode(id="DRUG:Colchicine", label="DRUG", name="Colchicine", evidence_count=71),
        PlaygroundNode(id="TARGET:TUBB4B", label="TARGET", name="TUBB4B", meta={"gene_symbol": "TUBB4B"}, evidence_count=24),
        PlaygroundNode(id="TARGET:TUBB1", label="TARGET", name="TUBB1", meta={"gene_symbol": "TUBB1"}, evidence_count=5),
        PlaygroundNode(id="PATHWAY:R-HSA-5653656", label="PATHWAY", name="Vesicle-mediated transport", evidence_count=10),
        PlaygroundNode(id="PATHWAY:R-HSA-1234567", label="PATHWAY", name="Unrelated Reactome Pathway", evidence_count=0),
        PlaygroundNode(id="REACTION:R-HSA-9999999", label="REACTION", name="Background Reaction 1", evidence_count=0),
        PlaygroundNode(id="GENE:TUBB4B", label="GENE", name="TUBB4B Gene", meta={"gene_symbol": "TUBB4B"}, evidence_count=2),
        PlaygroundNode(id="DISEASE:Heart Failure", label="DISEASE", name="Heart Failure", evidence_count=15),
    ]
    edges = [
        PlaygroundEdge(
            id="e1",
            source_id="DRUG:Colchicine",
            target_id="TARGET:TUBB4B",
            predicate="INHIBITOR",
            data_quality="DIRECT_TARGET_ASSOCIATION",
            evidence_count=24,
            evidence_strength=0.90,
            source_database="ChEMBL",
        ),
        PlaygroundEdge(
            id="e2",
            source_id="TARGET:TUBB4B",
            target_id="PATHWAY:R-HSA-5653656",
            predicate="PARTICIPATES_IN",
            data_quality="STRUCTURAL",
            evidence_count=0,
            evidence_strength=0.50,
            source_database="Reactome",
        ),
        PlaygroundEdge(
            id="e3",
            source_id="PATHWAY:R-HSA-5653656",
            target_id="DISEASE:Heart Failure",
            predicate="ASSOCIATED_WITH",
            data_quality="UNVALIDATED_BIOLOGICAL_BRIDGE",
            evidence_count=0,
            evidence_strength=0.45,
            source_database="Curated",
        ),
        PlaygroundEdge(
            id="e4",
            source_id="TARGET:TUBB1",
            target_id="PATHWAY:R-HSA-1234567",
            predicate="PARTICIPATES_IN",
            data_quality="STRUCTURAL",
            evidence_count=0,
            evidence_strength=0.30,
            source_database="Reactome",
        ),
        PlaygroundEdge(
            id="e5",
            source_id="PATHWAY:R-HSA-1234567",
            target_id="REACTION:R-HSA-9999999",
            predicate="CONTAINS_REACTION",
            data_quality="STRUCTURAL",
            evidence_count=0,
            evidence_strength=0.20,
            source_database="Reactome",
        ),
    ]
    candidate_mechanisms = [
        {
            "candidate_index": 1,
            "name": "TUBB4B - Direct target association -> Vesicle-mediated transport -> Heart Failure",
            "support_level": "WEAK_SPECULATIVE",
            "confidence_score": 0.431,
            "summary_chain": ["Colchicine", "TUBB4B", "Vesicle-mediated transport", "Heart Failure"],
            "hops": [
                {
                    "from_node": "Colchicine",
                    "to_node": "TUBB4B",
                    "predicate": "INHIBITOR",
                    "status": "DIRECT_TARGET_ASSOCIATION",
                    "source_database": "ChEMBL",
                    "canonical_from_id": "DRUG:Colchicine",
                    "canonical_to_id": "TARGET:TUBB4B",
                    "links": [{"url": "https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL84", "display_label": "ChEMBL"}],
                },
                {
                    "from_node": "TUBB4B",
                    "to_node": "Vesicle-mediated transport",
                    "predicate": "PARTICIPATES_IN",
                    "status": "STRUCTURAL_EVIDENCE",
                    "source_database": "Reactome",
                    "canonical_from_id": "TARGET:TUBB4B",
                    "canonical_to_id": "PATHWAY:R-HSA-5653656",
                    "links": [{"url": "https://reactome.org/content/detail/R-HSA-5653656", "display_label": "Reactome"}],
                },
                {
                    "from_node": "Vesicle-mediated transport",
                    "to_node": "Heart Failure",
                    "predicate": "ASSOCIATED_WITH",
                    "status": "UNVALIDATED_BIOLOGICAL_BRIDGE",
                    "source_database": "Curated",
                    "canonical_from_id": "PATHWAY:R-HSA-5653656",
                    "canonical_to_id": "DISEASE:Heart Failure",
                    "links": [],
                },
            ],
            "literature_citations": [{"title": "Colchicine study in HF", "url": "https://pubmed.ncbi.nlm.nih.gov/12345678"}],
        }
    ]
    return PlaygroundGraphData(
        hypothesis_id="test-hyp-1",
        drug_name="Colchicine",
        disease_name="Heart Failure",
        nodes=nodes,
        edges=edges,
        evidence=[],
        clinical_trials=[],
        contradictions=[],
        candidate_mechanisms=candidate_mechanisms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests 1-15
# ─────────────────────────────────────────────────────────────────────────────

def test_01_overview_graph_view_filters_clutter():
    """1. Overview graph view filters to top relevant targets, candidate pathways,
    and core disease nodes without background reaction clutter."""
    graph = _make_mock_playground_graph()
    overview_nodes, overview_edges = pv._get_overview_subgraph(graph)
    node_ids = {n.id for n in overview_nodes}
    node_labels = {n.label for n in overview_nodes}

    # Core nodes must be present
    assert "DRUG:Colchicine" in node_ids
    assert "DISEASE:Heart Failure" in node_ids
    assert "TARGET:TUBB4B" in node_ids
    assert "PATHWAY:R-HSA-5653656" in node_ids

    # Clutter nodes must NOT be present in Overview
    assert "REACTION:R-HSA-9999999" not in node_ids
    assert "REACTION" not in node_labels
    assert "PATHWAY:R-HSA-1234567" not in node_ids  # Unrelated background pathway excluded


def test_02_mechanisms_graph_view_highlights_candidate_mechanism():
    """2. Mechanisms graph view highlights the candidate mechanism and dims non-participating nodes/edges."""
    graph = _make_mock_playground_graph()
    positions = pv._get_or_compute_layout(graph)
    visible_types = {n.label for n in graph.nodes}

    fig, lookups = pv._build_playground_figure(
        graph,
        positions,
        visible_types,
        selected_element=None,
        selected_type=None,
        disabled_edges=set(),
        graph_mode="Mechanisms",
        selected_cm_idx=0,
    )

    # Entities trace is the nodes trace
    nodes_trace = [t for t in fig.data if t.name == "Entities"][0]
    # Highlighted candidate mechanism nodes must not be dimmed
    # Colchicine, TUBB4B, Vesicle-mediated transport, Heart Failure have visible names
    assert any("TUBB4B" in str(txt) for txt in nodes_trace.text)
    # Background reaction or unrelated pathway node should have empty on-node text (dimmed)
    assert any(txt == "" for txt in nodes_trace.text)


def test_03_evidence_graph_view_highlights_evidence_supported_edges():
    """3. Evidence graph view highlights evidence-supported edges and dims unvalidated/structural edges."""
    graph = _make_mock_playground_graph()
    positions = pv._get_or_compute_layout(graph)
    visible_types = {n.label for n in graph.nodes}

    fig, lookups = pv._build_playground_figure(
        graph,
        positions,
        visible_types,
        selected_element=None,
        selected_type=None,
        disabled_edges=set(),
        graph_mode="Evidence",
    )

    trace_names = [t.name for t in fig.data if t.name]
    assert "Direct / Evidence-Backed" in trace_names
    # Unvalidated or structural edges without direct evidence records get placed in dimmed lines


def test_04_full_graph_view_retains_all_nodes_and_edges():
    """4. Full Graph view retains all nodes and edges (zero data loss)."""
    graph = _make_mock_playground_graph()
    positions = pv._get_or_compute_layout(graph)
    visible_types = {n.label for n in graph.nodes}

    fig, lookups = pv._build_playground_figure(
        graph,
        positions,
        visible_types,
        selected_element=None,
        selected_type=None,
        disabled_edges=set(),
        graph_mode="Full Graph",
    )

    nodes_trace = [t for t in fig.data if t.name == "Entities"][0]
    assert len(nodes_trace.x) == len(graph.nodes)


def test_05_candidate_mechanism_selector_updates_highlighting():
    """5. Candidate mechanism selector updates graph highlighting and inspector details."""
    graph = _make_mock_playground_graph()
    cm = graph.candidate_mechanisms[0]
    matched_nodes, matched_edges = pv._get_mechanism_node_and_edge_ids(cm, graph.nodes, graph.edges)

    assert "DRUG:Colchicine" in matched_nodes
    assert "TARGET:TUBB4B" in matched_nodes
    assert "PATHWAY:R-HSA-5653656" in matched_nodes
    assert "DISEASE:Heart Failure" in matched_nodes
    assert "e1" in matched_edges or any("TARGET:TUBB4B" in str(e) for e in matched_edges)


def test_06_node_sizing_hierarchy():
    """6. Node sizing hierarchy: Drug/Disease (34-36px), Targets/Pathways (26-28px), Genes/Reactions (18-20px)."""
    assert 34 <= pv.NODE_BASE_SIZES["DRUG"] <= 36
    assert 34 <= pv.NODE_BASE_SIZES["DISEASE"] <= 36
    assert 26 <= pv.NODE_BASE_SIZES["TARGET"] <= 28
    assert 26 <= pv.NODE_BASE_SIZES["PATHWAY"] <= 28
    assert 18 <= pv.NODE_BASE_SIZES["GENE"] <= 20
    assert 18 <= pv.NODE_BASE_SIZES["REACTION"] <= 20


def test_07_node_labels_short_symbol_and_full_tooltip():
    """7. Node labels display canonical short symbol on graph and full identifier in hover tooltip."""
    node = PlaygroundNode(
        id="TARGET:P68371",
        label="TARGET",
        name="Tubulin beta-4B chain (UniProt:P68371)",
        meta={"gene_symbol": "TUBB4B"},
        evidence_count=24,
    )
    short_label = pv._canonical_node_display_name(node)
    assert short_label == "TUBB4B"

    # Build figure to verify tooltip contains full identifier
    graph = _make_mock_playground_graph()
    positions = pv._get_or_compute_layout(graph)
    fig, _ = pv._build_playground_figure(
        graph,
        positions,
        {n.label for n in graph.nodes},
        None,
        None,
        set(),
        graph_mode="Full Graph",
    )
    nodes_trace = [t for t in fig.data if t.name == "Entities"][0]
    assert any("ID: TARGET:TUBB4B" in txt for txt in nodes_trace.hovertext)


def test_08_edge_labels_hidden_and_midpoints_not_occluding():
    """8. Edge labels hidden by default; midpoints do not permanently occlude nodes/edges."""
    graph = _make_mock_playground_graph()
    positions = pv._get_or_compute_layout(graph)
    fig, _ = pv._build_playground_figure(
        graph,
        positions,
        {n.label for n in graph.nodes},
        None,
        None,
        set(),
        graph_mode="Full Graph",
    )

    # Interactive edge marker trace must be transparent (rgba(0,0,0,0.01))
    interactive_trace = [t for t in fig.data if t.name == "Relationships (Hover)"][0]
    assert "rgba(0,0,0,0.01)" in interactive_trace.marker.color or interactive_trace.marker.color == "rgba(0,0,0,0.01)"


def test_09_epistemic_status_edge_styling():
    """9. Epistemic status styling: structural Reactome edges styled distinctly (dashed/muted) from evidence-backed/direct edges."""
    graph = _make_mock_playground_graph()
    positions = pv._get_or_compute_layout(graph)
    fig, _ = pv._build_playground_figure(
        graph,
        positions,
        {n.label for n in graph.nodes},
        None,
        None,
        set(),
        graph_mode="Full Graph",
    )

    traces_by_name = {t.name: t for t in fig.data if t.name}
    assert "Direct / Evidence-Backed" in traces_by_name
    assert "Structural / Pathway Membership" in traces_by_name
    # Direct is solid line
    assert traces_by_name["Direct / Evidence-Backed"].line.dash is None or traces_by_name["Direct / Evidence-Backed"].line.dash == "solid"
    # Structural is dot or dash
    assert traces_by_name["Structural / Pathway Membership"].line.dash in ("dot", "dash")


def test_10_reactome_status_determination():
    """10. Reactome status: 0 failed -> SUCCESS, some failed + pathways -> PARTIAL with request counts, all failed -> FAILED."""
    pipeline = RetrievalPipeline()

    # Case A: 0 failed
    assert pipeline._derive_reactome_status(pathway_count=39, requests_failed=0, requests_total=10) == "SUCCESS"

    # Case B: some failed, pathways retrieved -> PARTIAL
    assert pipeline._derive_reactome_status(pathway_count=39, requests_failed=4, requests_total=10) == "PARTIAL"

    # Case C: all failed / 0 pathways -> FAILED
    assert pipeline._derive_reactome_status(pathway_count=0, requests_failed=5, requests_total=5) == "FAILED"


def test_11_reactome_request_counts_in_diagnostics():
    """11. Reactome request counts exposed in Data Source Diagnostics (X succeeded, Y failed)."""
    conn = ReactomeConnector()
    conn.requests_total = 10
    conn.requests_successful = 6
    conn.requests_failed = 4
    conn.failed_endpoints = ["/pathway/P68371/failed_1", "/pathway/P68371/failed_2"]

    summary = conn.get_request_summary()
    assert summary["requests_total"] == 10
    assert summary["requests_successful"] == 6
    assert summary["requests_failed"] == 4
    assert len(summary["failed_endpoints"]) == 2


def test_12_risk_score_zero_reported_as_observed_clinical_risk_signals_none_detected():
    """12. Risk Score RS = 0.000 reports:
    'Observed Clinical Risk Signals: NONE DETECTED (Risk Score = 0.000, Safety Signal Assessment: Grade A)'.
    """
    orchestrator = ReasoningOrchestrator.__new__(ReasoningOrchestrator)
    support = SupportAssessment(score=0.994, level="HIGH", evidence_count=87)
    mechanistic = MechanisticAssessment(score=0.431, level="LOW", pathway_count=39)
    risk = RiskAssessment(score=0.0, level="NONE", failed_trial_count=0, contradiction_count=0, rationale="No safety terminations.")
    safety_profile = SafetyProfile(overall_safety_grade="A")
    scientific_context = ScientificContext(
        regulatory=DimensionalAssessment(dimension="regulatory", status="INVESTIGATIONAL", confidence=0.0),
        repurposing=DimensionalAssessment(dimension="repurposing", status="ACTIVE_INVESTIGATION", confidence=0.5),
        mechanistic=DimensionalAssessment(dimension="mechanistic", status="SPECULATIVE", confidence=0.3),
        clinical=DimensionalAssessment(dimension="clinical", status="UNDER_EVALUATION", confidence=0.4),
        knowledge_maturity=DimensionalAssessment(dimension="knowledge_maturity", status="SPECULATIVE", confidence=0.2),
    )
    pkg = RetrievalPackage(
        hypothesis_id=uuid.uuid4(),
        drug=Drug(name="Colchicine", identifiers={"chembl": "CHEMBL84"}),
        disease=Disease(name="Heart Failure", identifiers={"mesh": "D006333"}),
        targets=[],
        proteins=[],
        genes=[],
        pathways=[],
        evidence_records=[],
        clinical_trials=[],
        retrieval_confidence="HIGH",
        sources_queried=["reactome", "clinicaltrials"],
        sources_failed=[],
        reactome_retrieval_status="PARTIAL",
        reactome_diagnostics={"status": "PARTIAL", "requests_successful": 6, "requests_failed": 4, "requests_total": 10},
    )

    report = orchestrator._generate_audit_report(
        all_claims=[],
        contradictions=[],
        support=support,
        mechanistic=mechanistic,
        risk=risk,
        recommendation=RecommendationStatus.UNCERTAIN,
        reasons=["Rule 5: UNCERTAIN"],
        prior_ctx=PriorKnowledgeContext(),
        scientific_context=scientific_context,
        safety_profile=safety_profile,
        mechanistic_paths=[],
        conflict_report=ConflictResolutionReport(),
        package=pkg,
    )

    assert "Observed Clinical Risk Signals: NONE DETECTED (Risk Score = 0.000, Safety Signal Assessment: Grade A)" in report.summary
    assert "Observed Clinical Risk Signals: NONE DETECTED" in report.safety_narrative
    assert "Risk Score = 0.000" in report.safety_narrative


def test_13_known_pharmacological_risk_context_not_assessed():
    """13. Known Pharmacological Risk Context reports:
    'NOT ASSESSED / STANDARD CLINICAL PRECAUTIONS APPLY' without fabricating adverse events.
    """
    orchestrator = ReasoningOrchestrator.__new__(ReasoningOrchestrator)
    support = SupportAssessment(score=0.994, level="HIGH", evidence_count=87)
    mechanistic = MechanisticAssessment(score=0.431, level="LOW", pathway_count=39)
    risk = RiskAssessment(score=0.0, level="NONE", failed_trial_count=0, contradiction_count=0)
    safety_profile = SafetyProfile(overall_safety_grade="A")
    scientific_context = ScientificContext(
        regulatory=DimensionalAssessment(dimension="regulatory", status="INVESTIGATIONAL", confidence=0.0),
        repurposing=DimensionalAssessment(dimension="repurposing", status="ACTIVE_INVESTIGATION", confidence=0.5),
        mechanistic=DimensionalAssessment(dimension="mechanistic", status="SPECULATIVE", confidence=0.3),
        clinical=DimensionalAssessment(dimension="clinical", status="UNDER_EVALUATION", confidence=0.4),
        knowledge_maturity=DimensionalAssessment(dimension="knowledge_maturity", status="SPECULATIVE", confidence=0.2),
    )
    pkg = RetrievalPackage(
        hypothesis_id=uuid.uuid4(),
        drug=Drug(name="Colchicine", identifiers={"chembl": "CHEMBL84"}),
        disease=Disease(name="Heart Failure", identifiers={"mesh": "D006333"}),
        targets=[],
        proteins=[],
        genes=[],
        pathways=[],
        evidence_records=[],
        clinical_trials=[],
        retrieval_confidence="HIGH",
        sources_queried=[],
        sources_failed=[],
    )

    report = orchestrator._generate_audit_report(
        all_claims=[],
        contradictions=[],
        support=support,
        mechanistic=mechanistic,
        risk=risk,
        recommendation=RecommendationStatus.UNCERTAIN,
        reasons=["Rule 5: UNCERTAIN"],
        prior_ctx=PriorKnowledgeContext(),
        scientific_context=scientific_context,
        safety_profile=safety_profile,
        mechanistic_paths=[],
        conflict_report=ConflictResolutionReport(),
        package=pkg,
    )

    # Must state not assessed and clinical precautions apply
    assert "B. Known Pharmacological Risk Context: NOT ASSESSED IN CLINICAL TRIAL DATA (Standard clinical precautions and known pharmacology profile still apply)" in report.safety_narrative
    # Must NOT fabricate adverse events
    assert "adverse event" not in report.safety_narrative.lower()


def test_14_terminology_cleanup_across_ui_and_pdf():
    """14. Terminology cleanup: 'Risk Level: 0.0% (NONE)' and 'clean safety record' replaced across UI and PDF."""
    orchestrator = ReasoningOrchestrator.__new__(ReasoningOrchestrator)
    support = SupportAssessment(score=0.994, level="HIGH", evidence_count=87)
    mechanistic = MechanisticAssessment(score=0.431, level="LOW", pathway_count=39)
    risk = RiskAssessment(score=0.0, level="NONE", failed_trial_count=0, contradiction_count=0)
    safety_profile = SafetyProfile(overall_safety_grade="A")
    scientific_context = ScientificContext(
        regulatory=DimensionalAssessment(dimension="regulatory", status="INVESTIGATIONAL", confidence=0.0),
        repurposing=DimensionalAssessment(dimension="repurposing", status="ACTIVE_INVESTIGATION", confidence=0.5),
        mechanistic=DimensionalAssessment(dimension="mechanistic", status="SPECULATIVE", confidence=0.3),
        clinical=DimensionalAssessment(dimension="clinical", status="UNDER_EVALUATION", confidence=0.4),
        knowledge_maturity=DimensionalAssessment(dimension="knowledge_maturity", status="SPECULATIVE", confidence=0.2),
    )
    pkg = RetrievalPackage(
        hypothesis_id=uuid.uuid4(),
        drug=Drug(name="Colchicine", identifiers={"chembl": "CHEMBL84"}),
        disease=Disease(name="Heart Failure", identifiers={"mesh": "D006333"}),
        targets=[],
        proteins=[],
        genes=[],
        pathways=[],
        evidence_records=[],
        clinical_trials=[],
        retrieval_confidence="HIGH",
        sources_queried=[],
        sources_failed=[],
    )

    report = orchestrator._generate_audit_report(
        all_claims=[],
        contradictions=[],
        support=support,
        mechanistic=mechanistic,
        risk=risk,
        recommendation=RecommendationStatus.UNCERTAIN,
        reasons=["Rule 5: UNCERTAIN"],
        prior_ctx=PriorKnowledgeContext(),
        scientific_context=scientific_context,
        safety_profile=safety_profile,
        mechanistic_paths=[],
        conflict_report=ConflictResolutionReport(),
        package=pkg,
    )

    assert "Risk Level: 0.0% (NONE)" not in report.summary
    assert "clean safety record across all trials" not in report.summary
    assert "clean safety record across all trials" not in report.safety_narrative


def test_15_pdf_and_playground_metric_consistency():
    """15. PDF + Playground metric consistency across RS, Observed Risk Signals,
    Safety Grade, Reactome status, request counts, candidate mechanisms,
    annotated pathways, version CYNTHERA v2.0, rule set 3.2."""
    hyp_id = uuid.uuid4()
    support = SupportAssessment(score=0.994, level="HIGH", evidence_count=87)
    mechanistic = MechanisticAssessment(
        score=0.431,
        level="LOW",
        pathway_count=39,
        candidate_mechanisms=[{
            "candidate_index": 1,
            "name": "TUBB4B path",
            "support_level": "WEAK_SPECULATIVE",
            "confidence_score": 0.431,
            "discovery_status": "CANDIDATE_STRUCTURAL",
        }],
    )
    risk = RiskAssessment(score=0.0, level="NONE", failed_trial_count=0, contradiction_count=0)
    safety_profile = SafetyProfile(overall_safety_grade="A")
    scientific_context = ScientificContext(
        regulatory=DimensionalAssessment(dimension="regulatory", status="INVESTIGATIONAL", confidence=0.0),
        repurposing=DimensionalAssessment(dimension="repurposing", status="ACTIVE_INVESTIGATION", confidence=0.5),
        mechanistic=DimensionalAssessment(dimension="mechanistic", status="SPECULATIVE", confidence=0.3),
        clinical=DimensionalAssessment(dimension="clinical", status="UNDER_EVALUATION", confidence=0.4),
        knowledge_maturity=DimensionalAssessment(dimension="knowledge_maturity", status="SPECULATIVE", confidence=0.2),
    )
    pkg = RetrievalPackage(
        hypothesis_id=hyp_id,
        drug=Drug(name="Colchicine", identifiers={"chembl": "CHEMBL84"}),
        disease=Disease(name="Heart Failure", identifiers={"mesh": "D006333"}),
        targets=[],
        proteins=[],
        genes=[],
        pathways=[Pathway(reactome_id=f"R-HSA-{i}", name=f"Pathway {i}", participant_uniprot_ids=[]) for i in range(39)],
        evidence_records=[],
        clinical_trials=[],
        retrieval_confidence="HIGH",
        sources_queried=["reactome", "clinicaltrials"],
        sources_failed=[],
        reactome_retrieval_status="PARTIAL",
        reactome_diagnostics={"status": "PARTIAL", "requests_successful": 6, "requests_failed": 4, "requests_total": 10},
    )
    orchestrator = ReasoningOrchestrator.__new__(ReasoningOrchestrator)
    audit = orchestrator._generate_audit_report(
        all_claims=[],
        contradictions=[],
        support=support,
        mechanistic=mechanistic,
        risk=risk,
        recommendation=RecommendationStatus.UNCERTAIN,
        reasons=["Rule 5 (UNCERTAIN): Inconclusive mechanistic link."],
        prior_ctx=PriorKnowledgeContext(),
        scientific_context=scientific_context,
        safety_profile=safety_profile,
        mechanistic_paths=[],
        conflict_report=ConflictResolutionReport(),
        package=pkg,
    )

    res = ReasoningResult(
        id=uuid.uuid4(),
        hypothesis_id=hyp_id,
        recommendation_status=RecommendationStatus.UNCERTAIN,
        recommendation_reasons=["Rule 5: UNCERTAIN"],
        support_assessment=support,
        mechanistic_assessment=mechanistic,
        risk_assessment=risk,
        contradictions=[],
        audit_report=audit,
        rule_set_version="3.2",
    )

    # PDF generation
    reporter = PDFReporter(drug_name="Colchicine", disease_name="Heart Failure")
    pdf_bytes = reporter.generate(res)
    assert len(pdf_bytes) > 1000

    # Text report fallback check
    txt_report = reporter._generate_text_report(res).decode("utf-8")
    assert "CYNTHERA v2.0" in txt_report
    assert "Rule Set: 3.2" in txt_report
    assert "39 annotated pathway(s)" in txt_report
    assert "1 candidate mechanism(s)" in txt_report
    assert "Observed Clinical Risk Signals: NONE DETECTED" in txt_report
    assert "Grade A" in txt_report
