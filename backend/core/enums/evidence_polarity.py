"""EvidencePolarity enum — directional evidence qualification.

Differentiates therapeutic support from adverse events, etiologic associations,
safety signals, neutral associations, and pure mechanistic interactions.
"""
from enum import Enum


class EvidencePolarity(str, Enum):
    """Directional polarity classification of clinical and literature evidence."""

    THERAPEUTIC_SUPPORT = "THERAPEUTIC_SUPPORT"
    """Explicit evidence of therapeutic benefit, improvement, efficacy, or prevention of the evaluated disease."""

    THERAPEUTIC_OPPOSITION = "THERAPEUTIC_OPPOSITION"
    """Explicit evidence of therapeutic failure, futility, lack of efficacy, or disease worsening."""

    ADVERSE_EVENT = "ADVERSE_EVENT"
    """Evidence of drug-induced harm, toxicity, complication, or adverse drug reaction."""

    ETIOLOGIC_ASSOCIATION = "ETIOLOGIC_ASSOCIATION"
    """Epidemiological or observational association where drug is a risk factor, cause, or trigger for the condition."""

    SAFETY_SIGNAL = "SAFETY_SIGNAL"
    """Regulatory warning, contraindication, boxed warning, or elevated clinical safety risk."""

    NEUTRAL_ASSOCIATION = "NEUTRAL_ASSOCIATION"
    """Statistical correlation, co-occurrence, or study mention without established therapeutic direction."""

    MECHANISTIC_ONLY = "MECHANISTIC_ONLY"
    """Drug-target binding, enzymatic modulation, or pathway interaction without direct therapeutic outcome on the evaluated disease."""

    UNKNOWN = "UNKNOWN"
    """Direction cannot be determined confidently from available context."""
