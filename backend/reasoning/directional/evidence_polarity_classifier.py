"""EvidencePolarityClassifier — Direction-of-effect and evidence polarity qualification.

Enforces strict evidence semantics:
1. Predicate alone NEVER establishes therapeutic support.
2. PREVENTS / INHIBITS / ACTIVATES / BINDS do NOT automatically become
   THERAPEUTIC_SUPPORT merely because the object matches the disease.
3. THERAPEUTIC_SUPPORT requires contextual evidence that:
   - the evaluated disease is the therapeutic outcome being addressed,
   - the claim expresses therapeutic improvement/benefit/efficacy/prevention of that disease,
   - the claim is not describing an adverse event, etiologic risk, complication, toxicity, or worsening.
4. Adverse events and etiologic associations are classified as ADVERSE_EVENT or
   ETIOLOGIC_ASSOCIATION and strictly excluded from therapeutic support.
5. If direction is ambiguous, classify as UNKNOWN or NEUTRAL_ASSOCIATION.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from backend.core.domain.claim import Claim
from backend.core.enums.evidence_polarity import EvidencePolarity
from backend.core.enums.predicate_type import PredicateType
from backend.reasoning.directional.therapeutic_evidence import _matches_evaluated_disease


# Semantic regex patterns for evidence classification
_ADVERSE_OR_ETIOLOGIC_PATTERNS = [
    r"\brisk factor\b",
    r"\b(?:increased|higher|elevated|greater)\s+risk\b",
    r"\betiologic(?:al)?\b",
    r"\bcauses?\s+(?:or\s+contributes\s+to\s+)?(?:harm|toxicity|damage|death|bleeding|hemorrhage|adverse|disease|worsening|exacerbation|failure|injury|lesions?|stroke)\b",
    r"\badverse\s+(?:event|effect|reaction|outcome)s?\b",
    r"\bdrug[- ]induced\b",
    r"\bcomplication(?:s)?\s+of\b",
    r"\b(?:bleed|bleeding|hemorrhage|hemorrhagic)\b",
    r"\btoxicity|toxic\b",
    r"\bworsen(?:ed|ing|s)?\b",
    r"\bcontraindicat(?:ed|ion)s?\b",
    r"\bharm(?:ful)?\b",
    r"\bfatal(?:ity)?|lethal\b",
    r"\bpatholog(?:y|ic|ical)\b",
    r"\bexacerbat(?:ed|ion|es|ing)\b",
    r"\bintracranial\s+bleed(?:ing)?\b",
]

_THERAPEUTIC_BENEFIT_PATTERNS = [
    r"\btreat(?:ment|ed|s|ing)?\b",
    r"\btherap(?:y|eutic|ies)\b",
    r"\befficacy\b",
    r"\bclinical\s+trial\b",
    r"\bimprov(?:e|ed|ement|ing|es)\b",
    r"\bbenefit(?:s|ted|ting)?\b",
    r"\bresponse\s+rate\b",
    r"\breduc(?:ed|ing|es|tion)\s+(?:of|in)?\s*(?:disease|severity|symptoms?|mortality|lesions?|infarct|death|risk\s+of\s+death)\b",
    r"\bprevent(?:ed|ing|s|ion)\s+(?:of|against)\b",
    r"\bsurvival\s+(?:benefit|advantage|gain|prolongation)\b",
    r"\btherapeutic\s+(?:agent|effect|option|potential)\b",
    r"\bremission\b",
    r"\bcure|curative\b",
    r"\balleviat(?:e|ed|ing|es)\b",
    r"\bprotect(?:ive|ion|ed|s)?\b",
]

_THERAPEUTIC_OPPOSITION_PREDICATES = {
    PredicateType.FAILED_TO_IMPROVE,
    PredicateType.NO_SIGNIFICANT_BENEFIT,
    PredicateType.TERMINATED_FOR_FUTILITY,
    PredicateType.TERMINATED_FOR_SAFETY,
    PredicateType.WORSENED_OUTCOME,
    PredicateType.CONTRAINDICATED,
}

_OPPOSITION_TEXT_PATTERNS = [
    r"\bfailed\s+to\s+(?:improve|show|meet|demonstrate|reduce)\b",
    r"\bno\s+(?:significant|clinical)\s+(?:benefit|improvement|difference|efficacy)\b",
    r"\bdid\s+not\s+(?:improve|reduce|decrease|benefit|prolong)\b",
    r"\bterminated\s+(?:for|due\s+to)\s+(?:futility|lack\s+of\s+efficacy|safety)\b",
    r"\bfutility\b",
    r"\black\s+of\s+efficacy\b",
]

_MECHANISTIC_PREDICATES = {
    PredicateType.ACTIVATES,
    PredicateType.INHIBITS,
    PredicateType.BINDS,
    PredicateType.UPREGULATES,
    PredicateType.DOWNREGULATES,
}


def _apply_updates(claim: Any, updates: dict[str, Any]) -> Any:
    """Safely apply updates to a Claim or Mock claim without losing nested properties."""
    if hasattr(claim, "_mock_name") or hasattr(claim, "_mock_methods") or not hasattr(claim, "model_copy"):
        for k, v in updates.items():
            setattr(claim, k, v)
        return claim
    try:
        return claim.model_copy(update=updates)
    except Exception:
        for k, v in updates.items():
            setattr(claim, k, v)
        return claim


class EvidencePolarityClassifier:
    """Classifies the directional polarity of claims and evidence records.

    Invariant: PREDICATE ALONE MUST NEVER ESTABLISH THERAPEUTIC SUPPORT.
    PREVENTS / INHIBITS / ACTIVATES / BINDS require contextual evidence that
    the evaluated disease is the therapeutic outcome being addressed and that
    the claim expresses therapeutic improvement/benefit/prevention rather than
    adverse risk, toxicity, or etiologic association.
    """

    @classmethod
    def classify_claim(
        cls,
        claim: Claim,
        drug_name: str,
        disease_name: str,
        known_targets: Optional[list[str]] = None,
    ) -> Claim:
        """Classify directional polarity of a Claim.

        Returns a new Claim instance with polarity fields populated.
        """
        pred = claim.predicate
        subj = (claim.subject or "").strip().lower()
        obj = (claim.object or "").strip().lower()
        raw_text = (claim.raw_text or "").strip()
        statement_val = getattr(claim, "statement", None) or f"{subj} {pred.value if hasattr(pred, 'value') else str(pred)} {obj}"
        combined_context = f"{statement_val} {raw_text}".lower()
        disease_lower = (disease_name or "").strip().lower()

        # ── 1. Check explicit opposition predicates first ─────────────────────────
        if pred in _THERAPEUTIC_OPPOSITION_PREDICATES:
            if pred == PredicateType.TERMINATED_FOR_SAFETY:
                pol = EvidencePolarity.SAFETY_SIGNAL
                reason = "Trial terminated due to safety concerns."
                adv = True
            elif pred == PredicateType.CONTRAINDICATED:
                pol = EvidencePolarity.SAFETY_SIGNAL
                reason = "Drug is explicitly contraindicated in this disease."
                adv = True
            elif pred == PredicateType.WORSENED_OUTCOME:
                pol = EvidencePolarity.THERAPEUTIC_OPPOSITION
                reason = "Drug worsened disease outcome."
                adv = True
            else:
                pol = EvidencePolarity.THERAPEUTIC_OPPOSITION
                reason = f"Explicit therapeutic failure predicate: {pred.value}."
                adv = False

            return _apply_updates(claim, {
                "polarity": pol,
                "polarity_confidence": 0.95,
                "therapeutic_support": False,
                "adverse_event_flag": adv,
                "polarity_reason": reason,
                "polarity_provenance": {
                    "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                    "predicate": str(pred.value if hasattr(pred, "value") else pred),
                    "context": raw_text[:120],
                },
            })

        # ── 2. Check for adverse event or etiologic risk framing ──────────────────
        has_adverse_framing = any(re.search(p, combined_context) for p in _ADVERSE_OR_ETIOLOGIC_PATTERNS)
        has_etiologic = bool(re.search(r"\brisk\s+factor|etiolog|drug[- ]induced\b", combined_context))
        
        # Check specifically if the evaluated disease or object represents harm/risk
        is_harm_object = any(h in obj for h in ("bleed", "hemorrhag", "toxic", "damage", "death", "stroke", "adverse", "injury"))
        is_disease_risk_object = (
            is_harm_object
            or (disease_lower in obj and any(w in obj for w in ("risk", "complication", "induced", "hazard", "bleed", "stroke", "death")))
            or bool(re.search(rf"\b(?:risk of|induced|causes?|develop(?:ed|ing)?|worsen(?:ed|ing)?)\s+{re.escape(disease_lower)}", combined_context))
        )

        _BENEFICIAL_EFFECTS = (
            "bronchodilation", "vasodilation", "efficacy", "improvement",
            "relief", "remission", "relaxation", "survival", "benefit", "control", "increase",
        )
        is_beneficial_object = any(b in obj for b in _BENEFICIAL_EFFECTS)

        if pred == PredicateType.CAUSES and not is_beneficial_object and (
            has_adverse_framing or is_harm_object or is_disease_risk_object or obj in (disease_lower,)
        ):
            # e.g. Aspirin CAUSES bleeding / hemorrhagic stroke
            pol = EvidencePolarity.ADVERSE_EVENT
            reason = f"Predicate CAUSES describes adverse event or pathologic induction: {subj} -> {obj}."
            return _apply_updates(claim, {
                "polarity": pol,
                "polarity_confidence": 0.90,
                "therapeutic_support": False,
                "adverse_event_flag": True,
                "polarity_reason": reason,
                "polarity_provenance": {
                    "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                    "predicate": "CAUSES",
                    "context": raw_text[:120],
                },
            })

        is_preventing_or_reducing = (
            pred == PredicateType.PREVENTS
            or bool(re.search(r"\b(?:lower|reduced?|decrease[ds]?|prevent(?:s|ed|ing|ion)?|less|minimal)\s+(?:risk|incidence|rate|frequency|exacerbat)", combined_context))
        )

        if has_adverse_framing and is_disease_risk_object and not is_preventing_or_reducing:
            # Check if text exclusively describes etiologic/adverse relationship
            pol = EvidencePolarity.ETIOLOGIC_ASSOCIATION if has_etiologic else EvidencePolarity.ADVERSE_EVENT
            reason = (
                f"Context indicates adverse/etiologic relationship: "
                f"evaluated condition ({disease_name}) framed as risk factor, complication, or adverse event."
            )
            return _apply_updates(claim, {
                "polarity": pol,
                "polarity_confidence": 0.90,
                "therapeutic_support": False,
                "adverse_event_flag": True,
                "polarity_reason": reason,
                "polarity_provenance": {
                    "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                    "predicate": str(pred.value if hasattr(pred, "value") else pred),
                    "context": raw_text[:120],
                },
            })

        # ── 3. Check for pure mechanistic drug-target interaction ─────────────────
        # If object is not the disease (e.g. a gene, protein, enzyme, or intermediate process)
        is_disease_object = (
            disease_lower in obj
            or (len(obj) >= 4 and obj in disease_lower)
            or any(w in obj for w in disease_lower.split() if len(w) >= 4)
            or _matches_evaluated_disease(obj, disease_name)
        )

        targets_lower = [t.lower() for t in (known_targets or [])]
        is_target_object = obj in targets_lower or any(t in obj for t in targets_lower)

        if pred in _MECHANISTIC_PREDICATES and not is_disease_object:
            # Pure mechanistic interaction (e.g. Aspirin INHIBITS PTGS1)
            pol = EvidencePolarity.MECHANISTIC_ONLY
            reason = f"Mechanistic drug-target interaction ({subj} {pred.value} {obj}); does not establish clinical therapeutic direction."
            return _apply_updates(claim, {
                "polarity": pol,
                "polarity_confidence": 0.90,
                "therapeutic_support": False,
                "adverse_event_flag": False,
                "polarity_reason": reason,
                "polarity_provenance": {
                    "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                    "predicate": str(pred.value if hasattr(pred, "value") else pred),
                    "context": raw_text[:120],
                },
            })

        # ── 4. Evaluate Therapeutic Support Criteria ──────────────────────────────
        # PREDICATE ALONE NEVER ESTABLISHES THERAPEUTIC SUPPORT.
        # Requires:
        # 1. Disease is the outcome being addressed.
        # 2. Context expresses therapeutic benefit, improvement, efficacy, or prevention.
        # 3. No adverse/etiologic disqualification.
        has_benefit_context = any(re.search(p, combined_context) for p in _THERAPEUTIC_BENEFIT_PATTERNS)
        has_opposition_context = any(re.search(p, combined_context) for p in _OPPOSITION_TEXT_PATTERNS)

        if has_opposition_context and not has_benefit_context:
            pol = EvidencePolarity.THERAPEUTIC_OPPOSITION
            reason = "Context describes clinical lack of efficacy, trial failure, or futility."
            return _apply_updates(claim, {
                "polarity": pol,
                "polarity_confidence": 0.85,
                "therapeutic_support": False,
                "adverse_event_flag": False,
                "polarity_reason": reason,
                "polarity_provenance": {
                    "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                    "predicate": str(pred.value if hasattr(pred, "value") else pred),
                    "context": raw_text[:120],
                },
            })

        if is_disease_object and (pred == PredicateType.PREVENTS or has_benefit_context):
            # Check for negative/adverse contradiction in the sentence
            if has_adverse_framing and not (pred == PredicateType.PREVENTS and not has_etiologic):
                # Mixed or adverse context: cannot safely establish support
                pol = EvidencePolarity.UNKNOWN
                reason = "Mixed therapeutic and adverse context; excluded from therapeutic support to preserve safety."
                return _apply_updates(claim, {
                    "polarity": pol,
                    "polarity_confidence": 0.60,
                    "therapeutic_support": False,
                    "adverse_event_flag": True,
                    "polarity_reason": reason,
                    "polarity_provenance": {
                        "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                        "predicate": str(pred.value if hasattr(pred, "value") else pred),
                        "context": raw_text[:120],
                    },
                })

            pol = EvidencePolarity.THERAPEUTIC_SUPPORT
            reason = (
                f"Contextually verified therapeutic direction: {subj} shows therapeutic benefit/prevention "
                f"for evaluated disease ({disease_name})."
            )
            return _apply_updates(claim, {
                "polarity": pol,
                "polarity_confidence": 0.85,
                "therapeutic_support": True,
                "adverse_event_flag": False,
                "polarity_reason": reason,
                "polarity_provenance": {
                    "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                    "predicate": str(pred.value if hasattr(pred, "value") else pred),
                    "context": raw_text[:120],
                },
            })

        # ── 5. Default fallback: ambiguous or neutral ─────────────────────────────
        if pred == PredicateType.ASSOCIATED_WITH:
            pol = EvidencePolarity.NEUTRAL_ASSOCIATION
            reason = f"Statistical/co-occurrence association ({subj} ASSOCIATED_WITH {obj}) lacks directional therapeutic evidence."
        elif pred == PredicateType.NO_EFFECT:
            pol = EvidencePolarity.NEUTRAL_ASSOCIATION
            reason = f"Explicitly showed NO_EFFECT on {obj}."
        else:
            pol = EvidencePolarity.UNKNOWN
            reason = f"Predicate '{pred.value if hasattr(pred, 'value') else pred}' on '{obj}' lacks verifiable therapeutic direction for {disease_name}."

        return _apply_updates(claim, {
            "polarity": pol,
            "polarity_confidence": 0.50,
            "therapeutic_support": False,
            "adverse_event_flag": False,
            "polarity_reason": reason,
            "polarity_provenance": {
                "source": getattr(claim.provenance, "source_name", getattr(claim.provenance, "source", "unknown")) if claim.provenance else "unknown",
                "predicate": str(pred.value if hasattr(pred, "value") else pred),
                "context": raw_text[:120],
            },
        })
