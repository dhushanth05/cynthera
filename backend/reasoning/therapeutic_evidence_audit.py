"""Auditable classification of evidence used for therapeutic decisions.

This module intentionally distinguishes *retrieved biomedical evidence* from
*authoritative therapeutic evidence for the queried drug--disease pair*.
The former can contribute to the heuristic Support Score; the latter is the
only kind of evidence allowed to bypass the mechanistic-path requirement in
the final decision synthesis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from backend.core.enums.evidence_type import EvidenceType
from backend.core.enums.trial_outcome import TrialOutcomeStatus
from backend.reasoning.directional.therapeutic_evidence import _matches_evaluated_disease


@dataclass(frozen=True)
class TherapeuticEvidenceAudit:
    """One provenance decision for a possible therapeutic-evidence source."""

    source: str
    evidence_type: str
    pair_specific: bool
    therapeutic_relevance: bool
    direction: str
    quality: str
    provenance_id: str | None
    allowed_for_high_quality_therapeutic: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_high_quality_therapeutic_evidence(package: Any) -> list[TherapeuticEvidenceAudit]:
    """Return the provenance audit behind the high-quality-therapeutic gate.

    A registry ``COMPLETED`` status is not an efficacy result, and an Evidence
    record typed as RCT or META_ANALYSIS has no structured direction/outcome in
    the current domain model.  Those records remain useful, traceable inputs to
    SS, but cannot by themselves establish a therapeutic indication.
    """
    records: list[TherapeuticEvidenceAudit] = []

    approval = getattr(package, "approval_signal", None)
    if approval is not None:
        is_pair_approved = bool(
            getattr(approval, "is_approved", False)
            and getattr(approval, "matched_indication_term", "")
            and float(getattr(approval, "match_confidence", 0.0)) > 0.0
        )
        records.append(
            TherapeuticEvidenceAudit(
                source=str(getattr(approval, "source", "chembl")),
                evidence_type="REGULATORY_INDICATION",
                pair_specific=bool(getattr(approval, "matched_indication_term", "")),
                therapeutic_relevance=is_pair_approved,
                direction="SUPPORTS" if is_pair_approved else "UNKNOWN",
                quality="REGULATORY",
                provenance_id=getattr(approval, "matched_indication_term", None) or None,
                allowed_for_high_quality_therapeutic=is_pair_approved,
                reason=(
                    "Disease-matched regulatory indication retrieved from ChEMBL."
                    if is_pair_approved
                    else "No disease-matched approved indication was established by the approval signal."
                ),
            )
        )

    disease_name = getattr(getattr(package, "disease", None), "name", "")
    drug_name = getattr(getattr(package, "drug", None), "name", "")

    from backend.core.enums.trial_attribution import TrialDrugRole
    from backend.reasoning.directional.therapeutic_evidence import is_therapeutically_eligible_evidence

    for trial in getattr(package, "clinical_trials", []) or []:
        trial_text = f"{getattr(trial, 'title', '')} {' '.join(getattr(trial, 'condition_names', []) or [])}".lower()
        trial_disease_match = _matches_evaluated_disease(trial_text, disease_name) if disease_name else True

        # Semantic eligibility of the trial for the evaluated disease
        is_eligible_target = True
        if disease_name:
            try:
                is_eligible_target, _ = is_therapeutically_eligible_evidence(trial, drug_name, disease_name)
            except Exception:
                is_eligible_target = True

        # Trial attribution check
        trial_role = getattr(trial, "candidate_drug_role", None)
        role_str = trial_role.value if hasattr(trial_role, "value") else (str(trial_role) if trial_role is not None else "")
        attr_decision = not getattr(trial, "attribution_rejected", False)

        if not role_str and drug_name:
            try:
                from backend.reasoning.opposition.therapeutic_opposition_assessor import evaluate_trial_attribution
                attr = evaluate_trial_attribution(trial, drug_name)
                if attr:
                    role_str = attr.drug_role.value if hasattr(attr.drug_role, "value") else str(attr.drug_role)
                    attr_decision = attr.final_attribution_decision
            except Exception:
                pass

        is_active_role = attr_decision and role_str not in (
            "BACKGROUND_CONSTANT_THERAPY",
            "BACKGROUND_THERAPY",
            "CONCOMITANT_THERAPY",
            "COMPARATOR_ONLY",
            "ACTIVE_COMPARATOR",
            "PLACEBO_COMPARATOR",
            "OTHER_UNKNOWN",
            "",
        )

        pair_scoped = bool(
            getattr(trial, "drug_chembl_id", None)
            and getattr(trial, "disease_identifier", None)
            and trial_disease_match
            and is_eligible_target
            and is_active_role
        )
        explicit_success = getattr(trial, "status", None) == TrialOutcomeStatus.COMPLETED_SUCCESS
        explicit_failure = bool(
            getattr(trial, "status", None) in (
                TrialOutcomeStatus.COMPLETED_FAILURE,
                TrialOutcomeStatus.TERMINATED_LACK_OF_EFFICACY,
                TrialOutcomeStatus.TERMINATED_SAFETY,
            )
            or getattr(trial, "is_negative_efficacy", False)
        )

        if explicit_failure:
            direction = "OPPOSES"
            quality = "CLINICAL_HUMAN_FAILURE"
            relevance = True
            allowed = False
            reason = f"Clinical trial outcome opposes therapeutic hypothesis: {getattr(trial, 'negative_efficacy_reason', None) or getattr(trial, 'why_stopped', None) or 'Documented trial failure or early termination.'}"
        elif explicit_success:
            trial_direction = getattr(trial, "direction", None)
            dir_str = str(trial_direction.value if hasattr(trial_direction, "value") else (trial_direction or ""))

            is_valid_support = pair_scoped and (
                dir_str in ("SUPPORTS", "POSITIVE", "STATISTICALLY_SIGNIFICANT_BENEFIT")
                or (dir_str not in ("UNKNOWN", "NEUTRAL", "INCONCLUSIVE", "OPPOSES") and is_active_role)
                or (not dir_str and is_active_role)
            )

            direction = "SUPPORTS" if is_valid_support else "UNKNOWN"
            quality = "CLINICAL_HUMAN" if is_valid_support else "REGISTERED_STATUS_ONLY"
            relevance = is_valid_support
            allowed = is_valid_support
            reason = (
                "Pair-scoped trial has an explicit successful therapeutic outcome."
                if is_valid_support
                else "Trial direction is ambiguous/unknown or drug is not evaluated as primary therapeutic agent."
            )
        else:
            direction = "UNKNOWN"
            quality = "REGISTERED_STATUS_ONLY"
            relevance = False
            allowed = False
            reason = "Registry status alone does not establish a successful therapeutic outcome."

        records.append(
            TherapeuticEvidenceAudit(
                source=getattr(getattr(trial, "provenance", None), "source_name", "ClinicalTrials.gov"),
                evidence_type="REGISTERED_CLINICAL_TRIAL",
                pair_specific=pair_scoped,
                therapeutic_relevance=relevance,
                direction=direction,
                quality=quality,
                provenance_id=getattr(trial, "nct_id", None),
                allowed_for_high_quality_therapeutic=allowed,
                reason=reason,
            )
        )

    for evidence in getattr(package, "evidence_records", []) or []:
        evidence_type = getattr(evidence, "evidence_type", None)
        if evidence_type not in {EvidenceType.RCT, EvidenceType.META_ANALYSIS}:
            continue
        records.append(
            TherapeuticEvidenceAudit(
                source=getattr(getattr(evidence, "provenance", None), "source_name", "Unknown"),
                evidence_type=evidence_type.value,
                pair_specific=bool(
                    getattr(evidence, "drug_chembl_id", None)
                    and getattr(evidence, "disease_identifier", None)
                ),
                therapeutic_relevance=False,
                direction="UNKNOWN",
                quality="CLINICAL_RECORD_UNINTERPRETED",
                provenance_id=getattr(evidence, "citation_key", None),
                allowed_for_high_quality_therapeutic=False,
                reason=(
                    "The Evidence model records study type and provenance but no structured "
                    "drug--disease efficacy outcome or therapeutic direction. It may contribute "
                    "to SS but cannot certify high-quality therapeutic support."
                ),
            )
        )

    return records


def has_high_quality_therapeutic_evidence(package: Any) -> tuple[bool, list[TherapeuticEvidenceAudit]]:
    """Return the strict decision gate and its complete audit trail."""
    audit = audit_high_quality_therapeutic_evidence(package)
    return any(item.allowed_for_high_quality_therapeutic for item in audit), audit
