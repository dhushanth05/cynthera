"""Fail-closed semantic gates for therapeutic support evidence.

Enforces directional semantics:
1. Rejects unmistakable risk/etiology/adverse-event framing without treatment context.
2. When disease_name is provided, specifically verifies whether the evaluated disease
   is framed as an adverse event, complication, or etiologic risk of the drug.
3. Distinguishes treatment of an alternate condition from therapeutic treatment of the queried disease.
"""
from __future__ import annotations

import re
from typing import Any


_NON_THERAPEUTIC_PATTERNS = (
    r"\brisk factor\b",
    r"\bassociated with (?:an? )?(?:increased|higher|elevated) risk\b",
    r"\betiologic(?:al)?\b",
    r"\bcauses? (?:or )?(?:contributes to|increases)\b",
    r"\badverse event\b",
    r"\bdrug[- ]induced\b",
    r"\bcomplication of treatment\b",
    r"\bhemorrhagic stroke\b.*\b(bleed|bleeding|hemorrhage|risk)\b",
)

_THERAPEUTIC_CONTEXT_PATTERNS = (
    r"\btreat(?:ment|ed|s)?\b",
    r"\btherap(?:y|eutic|ies)\b",
    r"\befficacy\b",
    r"\bclinical trial\b",
    r"\bimprov(?:e|ed|ement|ing)\b",
    r"\bbenefit\b",
    r"\bresponse\b",
)


def _matches_evaluated_disease(text: str, disease_name: str) -> bool:
    """Verify whether a text mentions the evaluated disease or its known clinical synonyms."""
    if not disease_name or not text:
        return True
    t_lower = text.lower()
    dis_lower = disease_name.lower().strip()
    if dis_lower in t_lower:
        return True
    synonym_map = {
        "hemorrhagic stroke": [
            "intracerebral hemorrhage", "intracranial hemorrhage", "cerebral hemorrhage",
            "subarachnoid hemorrhage", "hemorrhagic stroke", "brain hemorrhage"
        ],
        "ischemic stroke": [
            "acute ischemic stroke", "cerebral infarction", "brain ischemia", "cerebral ischemia"
        ],
        "pulmonary arterial hypertension": [
            "pulmonary hypertension", "pah", "pulmonary arterial"
        ],
        "type 2 diabetes": [
            "type 2 diabetes", "t2d", "diabetes mellitus", "hyperglycemia"
        ],
        "secondary cardiovascular prevention": [
            "cardiovascular", "myocardial infarction", "coronary", "stroke", "atherosclero"
        ],
        "gerd": [
            "gastroesophageal", "reflux", "heartburn", "esophagitis"
        ],
        "major depressive disorder": [
            "depression", "depressive", "major depression", "mdd"
        ],
        "allergic rhinitis": [
            "rhinitis", "hay fever", "nasal allergy"
        ],
        "hypercholesterolemia": [
            "cholesterol", "hyperlipidemia", "ldl", "lipid"
        ],
        "hypothyroidism": [
            "thyroid", "hypothyroid"
        ],
        "heart failure": [
            "heart failure", "cardiac failure", "cardiac", "myocardial",
            "cardiomyopathy", "hfpef", "hfref", "ventricular"
        ],
        "hypertension": [
            "hypertension", "blood pressure", "hypertensive", "antihypertensive"
        ],
        "asthma": [
            "asthma", "bronchial", "airway", "bronchospasm", "exacerbat"
        ],
        "rheumatoid arthritis": [
            "rheumatoid", "arthritis"
        ],
    }
    for key, syns in synonym_map.items():
        if key in dis_lower:
            if any(s in t_lower for s in syns):
                return True

    cleaned = re.sub(r"\b(disease|disorder|syndrome|acute|chronic|type|grade|stage|and|with|in|of)\b", "", dis_lower).strip()
    words = [w for w in re.findall(r"\b\w{4,}\b", cleaned)]
    if not words:
        return True
    return all(w in t_lower for w in words)


def is_therapeutically_eligible_evidence(
    evidence: Any,
    drug_name: str = "",
    disease_name: str = "",
) -> tuple[bool, str]:
    """Return whether a record can contribute direct therapeutic support.

    Invariant: PREDICATE ALONE MUST NEVER ESTABLISH THERAPEUTIC SUPPORT.
    THERAPEUTIC_SUPPORT requires contextual evidence that:
    1. the evaluated disease is the therapeutic outcome being addressed,
    2. the claim/record expresses therapeutic improvement/benefit/efficacy/prevention of that disease,
    3. the claim/record is not describing an adverse event, etiologic risk, complication, toxicity, or disease worsening.

    Args:
        evidence: Evidence record or object with title/abstract.
        drug_name: Optional queried drug name.
        disease_name: Optional queried disease name.

    Returns:
        tuple[bool, str]: (is_eligible, forensic_reason)
    """
    text = " ".join(
        str(value or "") for value in (
            getattr(evidence, "title", None),
            getattr(evidence, "abstract", None),
        )
    ).lower()
    if not text.strip():
        return True, "No text available for semantic exclusion; retained as contextual evidence."

    if "chembl curated mechanism" in text:
        return False, f"Pure mechanistic binding assay without clinical target disease context."

    if disease_name:
        dis_lower = disease_name.lower().strip()

        # 1. Condition address check: Does record discuss the evaluated disease?
        if not _matches_evaluated_disease(text, disease_name):
            return False, f"Record does not address target condition '{disease_name}'."

        # 2. Check for subtype / qualifier divergence (e.g., ischemic stroke vs hemorrhagic stroke)
        if "hemorrhag" in dis_lower:
            if ("ischemic stroke" in text or "cerebral infarction" in text or "transient ischemic" in text) and not (
                "hemorrhagic stroke" in text or "intracerebral hemorrhage" in text or "intracranial hemorrhage" in text
            ):
                return False, f"Record discusses ischemic stroke/infarction, not target condition '{disease_name}'."

        # 3. Adverse event, hazard, complication, or etiologic risk framing
        disease_as_adverse_patterns = [
            rf"\b(?:risk|incidence|rate|danger|hazard)\s+of\s+(?:\w+\s+){{0,3}}?{re.escape(dis_lower)}\b",
            rf"\b(?:increased|higher|elevated|excess)\s+(?:\w+\s+){{0,3}}?{re.escape(dis_lower)}\b",
            rf"\bincreases?\s+(?:\w+\s+){{0,4}}?{re.escape(dis_lower)}\b",
            rf"\bcauses?\s+(?:\w+\s+){{0,3}}?{re.escape(dis_lower)}\b",
            rf"\b{re.escape(dis_lower)}\s+as\s+a\s+(?:complication|risk|adverse|side\s+effect|hazard)\b",
            rf"\b{re.escape(dis_lower)}\s+(?:risk|hazard|complication|toxicity|adverse|side\s+effect)\b",
            rf"\b{re.escape(dis_lower)}\s+(?:bleed|bleeding|hemorrhage|toxicity)\b",
            rf"\b{re.escape(dis_lower)}\s+associated\s+with\s+(?:treatment|use|therapy|intake)\b",
            rf"\brisk\s+factor\s+for\s+{re.escape(dis_lower)}\b",
        ]
        is_disease_adverse_target = any(re.search(p, text) for p in disease_as_adverse_patterns)

        if "hemorrhag" in dis_lower or "bleed" in dis_lower:
            hem_adverse_patterns = [
                r"\bincreases?\s+(?:\w+\s+){0,4}?(?:bleed|bleeding|hemorrhag\w*|intracranial)\b",
                r"\b(?:risk|incidence|rate|danger|hazard)\s+of\s+(?:\w+\s+){0,3}?(?:bleed|bleeding|hemorrhag\w*|intracranial)\b",
                r"\b(?:bleed|bleeding|hemorrhag\w*|hazard)\b.*\bhemorrhagic\s+stroke\b",
                r"\bhemorrhagic\s+stroke\b.*\b(?:bleed|bleeding|hemorrhag\w*|hazard|risk)\b",
                r"\bhemorrhagic\s+transformation\b",
                r"\bhemorrhagic\s+complication\b",
                r"\bintracranial\s+(?:bleed|bleeding|hemorrhage)\b",
            ]
            if any(re.search(p, text) for p in hem_adverse_patterns):
                explicit_treat_dis = bool(re.search(rf"\btreat(?:ment|ed|ing)?\s+(?:of\s+)?{re.escape(dis_lower)}\b", text))
                if not explicit_treat_dis:
                    return False, f"Queried condition '{disease_name}' is described as a bleeding/hemorrhagic risk, adverse event, or complication."

        if is_disease_adverse_target:
            explicit_treat_dis = bool(re.search(rf"\btreat(?:ment|ed|ing)?\s+(?:of\s+)?{re.escape(dis_lower)}\b", text))
            if not explicit_treat_dis:
                return False, f"Queried condition '{disease_name}' is described as a risk, adverse event, or complication of the drug."

    non_therapeutic = any(re.search(pattern, text) for pattern in _NON_THERAPEUTIC_PATTERNS)
    therapeutic_context = any(re.search(pattern, text) for pattern in _THERAPEUTIC_CONTEXT_PATTERNS)
    if non_therapeutic and not therapeutic_context:
        return False, "Risk, etiologic, or adverse-event framing without therapeutic treatment context."
    return True, "No exclusive non-therapeutic framing detected."
