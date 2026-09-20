"""EvidenceSummary — canonical evidence count contract across agents, UI, and PDF.

Provides an authoritative, single source of truth for all evidence-related metrics
in CYNTHERA, resolving discrepancies between raw retrieval records, subgraph-mapped
records, claims, clinical trials, candidate mechanisms, and pathways.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class EvidenceSummary(BaseModel):
    """Authoritative summary of evidence counts across all pipeline layers.

    All layers (Frontend, PDF, Playground, Agent verdicts) MUST use this object
    to describe counts, ensuring consistent semantics and eliminating ambiguity.
    """

    model_config = {"frozen": True}

    # Raw biomedical evidence records (PubMed, Europe PMC, ChEMBL assay, etc.)
    total_retrieved_records: int = Field(
        default=0,
        description="Total raw evidence records in RetrievalPackage (PubMed, Europe PMC, ChEMBL, etc.).",
    )

    # Subgraph-mapped evidence records (associated with nodes/edges in graph)
    subgraph_mapped_records: int = Field(
        default=0,
        description="Evidence records directly mapped to nodes/edges in the active subgraph.",
    )

    # NLP/LLM claims extracted from literature abstracts
    literature_claims_extracted: int = Field(
        default=0,
        description="Structured subject-predicate-object claims extracted from literature.",
    )

    # Human clinical trials (ClinicalTrials.gov)
    clinical_trials_evaluated: int = Field(
        default=0,
        description="Clinical trials specifically evaluated for the drug-disease pair.",
    )
    clinical_trials_failed: int = Field(
        default=0,
        description="Clinically significant failed or terminated clinical trials.",
    )
    clinical_trials_successful: int = Field(
        default=0,
        description="Clinical trials demonstrating pair-specific positive therapeutic efficacy.",
    )

    # Evaluated support signals (records + claims evaluated by SupportAssessmentAgent)
    evaluated_support_signals: int = Field(
        default=0,
        description="Evidence records plus supporting claims evaluated by Support Assessment Agent.",
    )

    # Mechanistic entities
    candidate_mechanisms_count: int = Field(
        default=0,
        description="Multi-hop candidate biological hypotheses connecting drug targets to disease pathology.",
    )
    validated_mechanisms_count: int = Field(
        default=0,
        description="Candidate mechanisms with literature-validated biological bridges.",
    )
    mechanistic_paths_count: int = Field(
        default=0,
        description="Graph traversal paths discovered between drug and disease.",
    )
    pathways_annotated_count: int = Field(
        default=0,
        description="Total Reactome pathways associated with involved targets and genes.",
    )
    candidate_mechanism_pathways_count: int = Field(
        default=0,
        description="Unique pathways traversed by candidate mechanism chains.",
    )

    # Drug targets
    targets_count: int = Field(
        default=0,
        description="Biochemical targets retrieved for the drug.",
    )

    def to_display_dict(self) -> dict[str, Any]:
        """Return human-readable display dictionary for UI and PDF reporting."""
        return {
            "total_retrieved_records": {
                "count": self.total_retrieved_records,
                "label": "Total Evidence Records",
                "description": "Total records retrieved from PubMed, Europe PMC, ChEMBL, and databases",
            },
            "subgraph_mapped_records": {
                "count": self.subgraph_mapped_records,
                "label": "Subgraph-Mapped Records",
                "description": "Evidence records directly anchored to entities in the mechanistic graph",
            },
            "literature_claims": {
                "count": self.literature_claims_extracted,
                "label": "Extracted Literature Claims",
                "description": "Biomedical claims extracted from peer-reviewed literature",
            },
            "evaluated_support_signals": {
                "count": self.evaluated_support_signals,
                "label": "Evaluated Support Signals",
                "description": "Evidence records and extracted claims evaluated by Support Assessment Agent",
            },
            "clinical_trials": {
                "count": self.clinical_trials_evaluated,
                "failed": self.clinical_trials_failed,
                "successful": self.clinical_trials_successful,
                "label": "Clinical Trials Evaluated",
                "description": "Human clinical trials evaluated from ClinicalTrials.gov",
            },
            "mechanisms": {
                "total": self.candidate_mechanisms_count,
                "validated": self.validated_mechanisms_count,
                "paths": self.mechanistic_paths_count,
                "label": "Candidate Mechanisms",
                "description": "Evaluated biological hypotheses connecting drug targets to disease pathology",
            },
            "pathways": {
                "annotated_count": self.pathways_annotated_count,
                "candidate_mechanism_count": self.candidate_mechanism_pathways_count,
                "label": "Reactome Pathways",
                "description": f"Annotated Reactome Pathways: {self.pathways_annotated_count}, Candidate Mechanism Pathways: {self.candidate_mechanism_pathways_count}",
            },
        }
