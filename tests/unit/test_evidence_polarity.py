"""Unit tests for Evidence Polarity and Direction-of-Effect Filtering.

Validates the 12 requirements specified in the Polarity Experiment:
1. Therapeutic improvement -> THERAPEUTIC_SUPPORT
2. Disease worsening -> THERAPEUTIC_OPPOSITION
3. Drug causes adverse event -> ADVERSE_EVENT
4. Drug increases disease risk -> ETIOLOGIC_ASSOCIATION / ADVERSE_EVENT
5. Drug associated with disease with no direction -> UNKNOWN / NEUTRAL_ASSOCIATION
6. Mechanistic relationship without therapeutic direction -> MECHANISTIC_ONLY
7. Adverse-event claim does not contribute to therapeutic Support Score
8. Therapeutic support claim still contributes normally
9. Adverse event does not automatically become OPPOSE
10. TC-082-style aspirin -> hemorrhagic stroke evidence is not counted as therapeutic support
11. Existing legitimate therapeutic claims remain SUPPORT
12. Existing tests continue passing
"""
import uuid
import pytest
from types import SimpleNamespace

from backend.core.domain.claim import Claim
from backend.core.domain.disease import Disease
from backend.core.domain.drug import Drug
from backend.core.domain.retrieval_package import RetrievalPackage
from backend.core.domain.evidence import Evidence
from backend.core.enums.evidence_polarity import EvidencePolarity
from backend.core.enums.evidence_type import EvidenceType
from backend.core.enums.predicate_type import PredicateType
from backend.core.value_objects.erw import ERW
from backend.core.value_objects.provenance import ProvenanceReference
from backend.core.value_objects.identifier import CanonicalIdentifier, ResolvedIdentifierSet
from backend.reasoning.directional.evidence_polarity_classifier import EvidencePolarityClassifier
from backend.reasoning.directional.therapeutic_evidence import is_therapeutically_eligible_evidence
from backend.reasoning.orchestrator.reasoning_orchestrator import ReasoningOrchestrator
from backend.reasoning.agents.prior_knowledge_agent import PriorKnowledgeContext


def _make_claim(
    subject: str,
    predicate: PredicateType,
    object_term: str,
    raw_text: str = "",
    confidence: float = 0.9,
    erw_val: float = 0.8,
) -> Claim:
    return Claim(
        id=uuid.uuid4(),
        subject=subject,
        predicate=predicate,
        object=object_term,
        confidence=confidence,
        erw=ERW(value=erw_val, base_weight=erw_val),
        provenance=ProvenanceReference(source_name="test_lit", source_version="1.0", record_id="REC001"),
        raw_text=raw_text or f"{subject} {predicate.value} {object_term}",
        is_validated=True,
    )


def _make_drug(name: str) -> Drug:
    return Drug(
        name=name,
        identifiers=ResolvedIdentifierSet(
            entity_name=name,
            entity_type="drug",
            identifiers=[CanonicalIdentifier(namespace="chembl", value="CHEMBL_TEST")],
        ),
    )


def _make_disease(name: str) -> Disease:
    return Disease(
        name=name,
        identifiers=ResolvedIdentifierSet(
            entity_name=name,
            entity_type="disease",
            identifiers=[CanonicalIdentifier(namespace="mesh", value="MESH_TEST")],
        ),
    )


def test_1_therapeutic_improvement_is_support():
    claim = _make_claim(
        subject="Lisinopril",
        predicate=PredicateType.PREVENTS,
        object_term="Hypertension",
        raw_text="Lisinopril treatment significantly improves blood pressure control and prevents hypertension.",
    )
    classified = EvidencePolarityClassifier.classify_claim(claim, "Lisinopril", "Hypertension")
    assert classified.polarity == EvidencePolarity.THERAPEUTIC_SUPPORT
    assert classified.therapeutic_support is True
    assert classified.adverse_event_flag is False


def test_2_disease_worsening_is_opposition():
    claim = _make_claim(
        subject="DrugX",
        predicate=PredicateType.WORSENED_OUTCOME,
        object_term="Heart failure",
        raw_text="DrugX therapy worsened heart failure outcomes and increased hospitalization.",
    )
    classified = EvidencePolarityClassifier.classify_claim(claim, "DrugX", "Heart failure")
    assert classified.polarity == EvidencePolarity.THERAPEUTIC_OPPOSITION
    assert classified.therapeutic_support is False
    assert classified.adverse_event_flag is True


def test_3_drug_causes_adverse_event():
    claim = _make_claim(
        subject="Aspirin",
        predicate=PredicateType.CAUSES,
        object_term="Gastrointestinal bleeding",
        raw_text="Aspirin causes gastrointestinal bleeding in elderly patients.",
    )
    classified = EvidencePolarityClassifier.classify_claim(claim, "Aspirin", "Cardiovascular disease")
    assert classified.polarity == EvidencePolarity.ADVERSE_EVENT
    assert classified.therapeutic_support is False
    assert classified.adverse_event_flag is True


def test_4_drug_increases_disease_risk_is_etiologic():
    claim = _make_claim(
        subject="Aspirin",
        predicate=PredicateType.ASSOCIATED_WITH,
        object_term="Hemorrhagic stroke",
        raw_text="Long-term aspirin therapy is an etiologic risk factor associated with increased risk of hemorrhagic stroke.",
    )
    classified = EvidencePolarityClassifier.classify_claim(claim, "Aspirin", "Hemorrhagic stroke")
    assert classified.polarity in (EvidencePolarity.ETIOLOGIC_ASSOCIATION, EvidencePolarity.ADVERSE_EVENT)
    assert classified.therapeutic_support is False
    assert classified.adverse_event_flag is True


def test_5_associated_without_direction_is_neutral_or_unknown():
    claim = _make_claim(
        subject="Metformin",
        predicate=PredicateType.ASSOCIATED_WITH,
        object_term="Alzheimer disease",
        raw_text="Metformin prescription records were associated with Alzheimer disease incidence in database query.",
    )
    classified = EvidencePolarityClassifier.classify_claim(claim, "Metformin", "Alzheimer disease")
    assert classified.polarity in (EvidencePolarity.NEUTRAL_ASSOCIATION, EvidencePolarity.UNKNOWN)
    assert classified.therapeutic_support is False


def test_6_mechanistic_relationship_without_disease_direction():
    claim = _make_claim(
        subject="Aspirin",
        predicate=PredicateType.INHIBITS,
        object_term="PTGS1",
        raw_text="Aspirin irreversibly inhibits cyclooxygenase-1 (PTGS1) in platelets.",
    )
    classified = EvidencePolarityClassifier.classify_claim(
        claim, "Aspirin", "Hemorrhagic stroke", known_targets=["PTGS1", "PTGS2"]
    )
    assert classified.polarity == EvidencePolarity.MECHANISTIC_ONLY
    assert classified.therapeutic_support is False


@pytest.mark.asyncio
async def test_7_adverse_event_claim_does_not_contribute_to_support_score():
    orchestrator = ReasoningOrchestrator()
    drug = _make_drug("Aspirin")
    disease = _make_disease("Hemorrhagic stroke")

    # Claim describing adverse event
    adverse_claim = _make_claim(
        subject="Aspirin",
        predicate=PredicateType.CAUSES,
        object_term="Hemorrhagic stroke",
        raw_text="Aspirin causes hemorrhagic stroke through antiplatelet inhibition.",
    )

    pkg = RetrievalPackage(
        hypothesis_id=uuid.uuid4(),
        drug=drug,
        disease=disease,
        evidence_records=[],
    )
    prior_ctx = PriorKnowledgeContext()

    support = await orchestrator._compute_support_score([adverse_claim], pkg, prior_ctx)
    # The adverse event claim MUST NOT contribute to support
    assert support.score == 0.0
    assert support.evidence_count == 0


@pytest.mark.asyncio
async def test_8_therapeutic_support_claim_contributes_normally():
    orchestrator = ReasoningOrchestrator()
    drug = _make_drug("Lisinopril")
    disease = _make_disease("Hypertension")

    # Therapeutic claim
    support_claim = _make_claim(
        subject="Lisinopril",
        predicate=PredicateType.PREVENTS,
        object_term="Hypertension",
        raw_text="Randomized trial demonstrates Lisinopril treatment significantly improves blood pressure control and prevents hypertension.",
    )

    pkg = RetrievalPackage(
        hypothesis_id=uuid.uuid4(),
        drug=drug,
        disease=disease,
        evidence_records=[],
    )
    prior_ctx = PriorKnowledgeContext()

    support = await orchestrator._compute_support_score([support_claim], pkg, prior_ctx)
    assert support.score > 0.0
    assert support.evidence_count == 1


def test_9_adverse_event_does_not_automatically_become_oppose():
    # Drug causes nausea for a headache indication -> Nausea should NOT create OPPOSE for headache
    claim = _make_claim(
        subject="DrugY",
        predicate=PredicateType.CAUSES,
        object_term="Nausea",
        raw_text="DrugY treatment causes mild nausea as a common adverse event.",
    )
    classified = EvidencePolarityClassifier.classify_claim(claim, "DrugY", "Migraine")
    assert classified.polarity == EvidencePolarity.ADVERSE_EVENT
    assert classified.therapeutic_support is False

    # In opposition assessment, nausea for migraine does not match migraine endpoint
    from backend.reasoning.opposition.therapeutic_opposition_assessor import TherapeuticOppositionAssessor
    assessor = TherapeuticOppositionAssessor()
    opp = assessor.assess([classified], "DrugY", "Migraine")
    # Must NOT generate empirical opposition
    assert opp.score == 0.0
    assert opp.level == "NONE"


def test_10_tc082_aspirin_hemorrhagic_stroke_semantic_gating():
    # Verify both literature record and claim exclusion
    rec = SimpleNamespace(
        title="Aspirin as a risk factor for hemorrhagic stroke in patients with hypertension",
        abstract="Antiplatelet therapy with aspirin increases the risk of hemorrhagic stroke.",
    )
    eligible, reason = is_therapeutically_eligible_evidence(rec, drug_name="Aspirin", disease_name="Hemorrhagic stroke")
    assert eligible is False
    assert "risk" in reason.lower() or "adverse" in reason.lower()

    # Even if abstract mentions treatment of ischemic stroke, hemorrhagic stroke as adverse outcome is blocked
    mixed_rec = SimpleNamespace(
        title="Aspirin treatment for ischemic stroke",
        abstract="While aspirin is effective for ischemic stroke treatment, it is associated with increased risk of hemorrhagic stroke.",
    )
    eligible_mixed, reason_mixed = is_therapeutically_eligible_evidence(
        mixed_rec, drug_name="Aspirin", disease_name="Hemorrhagic stroke"
    )
    assert eligible_mixed is False
    assert "risk" in reason_mixed.lower() or "adverse" in reason_mixed.lower()


def test_11_legitimate_therapeutic_claims_remain_support():
    claim = _make_claim(
        subject="Colchicine",
        predicate=PredicateType.PREVENTS,
        object_term="Pericarditis",
        raw_text="Double-blind randomized clinical trial proves colchicine treatment reduces pericarditis recurrence.",
    )
    classified = EvidencePolarityClassifier.classify_claim(claim, "Colchicine", "Pericarditis")
    assert classified.polarity == EvidencePolarity.THERAPEUTIC_SUPPORT
    assert classified.therapeutic_support is True
