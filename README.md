# 🧬 CYNTHERA — Evidence-Traceable, Mechanism-Grounded & Directional AI Research Assistant for Drug Repurposing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Unit Tests](https://img.shields.io/badge/tests-366%20passing-brightgreen.svg)](tests/unit/)
[![Architecture](https://img.shields.io/badge/architecture-sealed%20two--tier-purple.svg)](#-architecture)
[![Data](https://img.shields.io/badge/data-100%25%20free%20open%20biomedical-orange.svg)](#-data-sources-100-free--open)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#-license--disclaimer)

**CYNTHERA** investigates a **Drug + Disease** hypothesis and produces a scientifically defensible, multi-dimensional assessment of **Mechanistic Plausibility** and **Therapeutic Directional Consistency**, backed by multi-hop biological reasoning and traceable to primary sources (PMIDs, DOIs, clinical trial IDs, and curated database records).

The system is built on four core principles:

1. **Mechanism-first, evidence-grounded reasoning:** Biological relationships (`Drug → Target → Reaction → Pathway → Disease Gene → Disease`) are constructed from retrieved database records (ChEMBL, Reactome, UniProt, Open Targets, DisGeNET), not invented by a language model. LLMs interpret or extract claims from literature; they never fabricate biological connectivity.
2. **Strict Semantic Separation:** Mechanistic plausibility, molecular polarity, therapeutic direction, and directional mechanism consistency are evaluated as orthogonal dimensions.
3. **Traceability:** Every important conclusion is linked back through claims and evidence to primary sources with direct URLs (PubMed, Europe PMC, ChEMBL, UniProt, Reactome, ClinicalTrials.gov).
4. **Honest Failure & Non-Conflation:** Missing data is never silently converted into biological implausibility. `DATA NOT FOUND`, `SOURCE UNAVAILABLE`, `INSUFFICIENT EVIDENCE`, `CONTRADICTORY EVIDENCE`, `UNSUPPORTED`, and `PLAUSIBLE` are strictly partitioned states.

---

## 🔬 Scientific Core: The Four Reasoning Dimensions

A common failure mode of biomedical AI is collapsing distinct scientific questions into a single scalar confidence score. CYNTHERA strictly separates four orthogonal dimensions:

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                               CYNTHERA REASONING DIMENSIONS                              │
├──────────────────────────────┬──────────────────────────────┬─────────────────────────────┤
│ Dimension                    │ Scientific Question          │ Values / Spectrum           │
├──────────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 1. Mechanistic Plausibility  │ Is there an evidence-backed  │ Score: [0.0, 1.0]           │
│    (MS)                      │ multi-hop path from drug     │ Level: HIGH / MEDIUM /      │
│                              │ target to the disease?       │        LOW / INSUFFICIENT   │
├──────────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 2. Molecular Polarity        │ What is the sign of an       │ POSITIVE (+) / NEGATIVE (-) │
│                              │ individual molecular edge?   │ / UNKNOWN (?)               │
├──────────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 3. Therapeutic Direction     │ Does drug action match the   │ SUPPORTS / OPPOSES /        │
│    Alignment                 │ required disease target      │ INSUFFICIENT                │
│                              │ direction of effect?         │                             │
├──────────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 4. Directional Mechanism     │ Does the multi-hop pathway   │ CONSISTENT / CONTRADICTORY  │
│    Consistency               │ preserve or contradict the   │ / PARTIAL / UNKNOWN         │
│                              │ target therapeutic action?   │                             │
└──────────────────────────────┴──────────────────────────────┴─────────────────────────────┘
```

### The Reactome Invariant
Reactome participation roles (`CATALYST`, `INPUT`, `OUTPUT`, `PARTICIPANT`, `COMPLEX_COMPONENT`, `ENTITY_SET_MEMBER`) represent **structural physical connectivity**, not causal regulation:
- Structural Reactome participation strictly receives `MolecularPolarity.UNKNOWN` and `CausalGrounding.STRUCTURAL`.
- Only explicit regulatory annotations (`POSITIVE_REGULATOR`, `NEGATIVE_REGULATOR`) or curated literature claims carry non-zero sign.
- **Low mechanistic score $\neq$ therapeutic opposition.**
- **Missing reaction evidence $\neq$ mechanistic score of 0.**
- **High mechanistic plausibility $\neq$ therapeutic support** (e.g., *Testosterone* has high mechanistic connectivity to *Prostate Cancer*, but directionally **OPPOSES** treatment because androgen receptor activation promotes tumor proliferation).

---

## ✨ Features & Capabilities

- **Multi-Hop Evidence Graph & Path Finding**: Dynamically constructs a typed biological `EvidenceGraph` (`DRUG`, `TARGET`, `PATHWAY`, `REACTION`, `GENE`, `DISEASE`). Uses depth-first search (DFS) pathfinding to discover 2-hop to 5-hop reaction-enriched cascades:
  - *4-Hop Pathway:* `Drug → Target → Pathway → Disease Gene → Disease`
  - *5-Hop Reaction-Enriched:* `Drug → Target → Reaction → Pathway → Disease Gene → Disease`
- **Zero-Crowding Pathway Ranking**: Filters to confirmed target participants and prioritizes confirmed disease-gene overlap before slicing pathways.
- **Evidence Polarity Classification**: Classifies retrieved scientific evidence and claims into directionally grounded polarities (`SUPPORT`, `OPPOSE`, `NEUTRAL/UNCERTAIN`) with negation handling.
- **Therapeutic Evidence Applicability Gates**: Rigorously validates evidence relevance, preventing comparator-arm outcomes or unselected populations from polluting candidate drug assessments.
- **Contextual Clinical Trial Analysis**: Connects directly to ClinicalTrials.gov and inspects `whyStopped` trial logs with NLP pattern matching to distinguish true clinical safety/efficacy failures from administrative friction (e.g., low accrual, COVID-19 delays, sponsor funding limits).
- **Interactive Investigation Playground**:
  - **Dynamic Subgraph Extraction:** Interactive, force-directed graph exploration of causal paths, evidence nodes, and molecular targets.
  - **Counterfactual "What-If" Scenario Engine:** Simulates the effect of target perturbations (inhibition, activation, knockout) on downstream disease mediators.
- **Claim → Evidence → Source → URL Traceability**: Every conclusion links back to exact PMIDs, DOIs, ChEMBL assay IDs, UniProt accessions, Reactome diagram IDs, and ClinicalTrials.gov NCT numbers with direct URLs.
- **Honest Failure Handling**: Outages and missing data produce partitioned states (`INSUFFICIENT EVIDENCE`, `DATA NOT FOUND`, `SOURCE UNAVAILABLE`) rather than artificial negative scores.
- **Raw-Response SQLite Cache Layer**: Tiered TTL caching for structural, association, literature, and trial queries, with a `--no-cache` CLI/API bypass.
- **Multi-Interface Access**: Streamlit Web Dashboard (`frontend/app.py`), Investigation Playground (`frontend/playground_view.py`), FastAPI REST service, and CLI.

---

## 🏗️ Architecture

CYNTHERA employs a **sealed two-tier architecture**: a deterministic engineering retrieval layer feeds an immutable, structured `RetrievalPackage` into an agentic + rule-based reasoning layer.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          ENGINEERING RETRIEVAL LAYER (Deterministic)                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Master Orchestrator                                                                    │
│   ├── Biological Identifier Normalizer (ChEMBL, UniProt, MeSH, MONDO, HGNC)            │
│   ├── Async Parallel Retrieval Connectors:                                             │
│   │     • ChEMBL (bioactivity, binding affinities, mechanism of action)                │
│   │     • UniProt (canonical accessions, human proteome verification)                  │
│   │     • Reactome (pathway hierarchies, reaction events, molecular roles)             │
│   │     • Open Targets Platform (disease associations, Direction-of-Effect)            │
│   │     • DisGeNET (curated gene-disease scores, actionable target trajectories)       │
│   │     • Europe PMC & PubMed (literature citations, abstracts)                        │
│   │     • ClinicalTrials.gov (recruitment status, whyStopped termination logs)         │
│   │     • Semantic Scholar & OpenAlex (citation-weighted literature metadata)          │
│   └── Tiered SQLite Raw-Response Cache Layer (`data/cynthera.db`)                      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                               Sealed `RetrievalPackage`
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                             REASONING & ASSESSMENT LAYER                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Claim Extraction Agent (Groq / Gemini LLM with automatic rule-based fallback)       │
│ 2. Evidence Graph Builder (Typed multi-relational graph construction)                  │
│ 3. Multi-Hop Reasoner (DFS pathfinder discovering 2-hop to 5-hop reaction paths)       │
│ 4. Mechanism Validator (7-dimensional candidate mechanism validation)                  │
│ 5. Evidence Polarity Classifier & Applicability Gates                                  │
│ 6. Directional Mechanism Evaluator (Path-level regulatory polarity & contradictions)   │
│ 7. Therapeutic Alignment & Opposition Engine (DoE / DATT target consensus)             │
│ 8. Clinical Safety Agent (Trial outcome classification & adverse event profiling)      │
│ 9. Deterministic Consensus & Rule Engine (Evidence-first scoring & status synthesis)  │
│ 10. Investigation Playground Engine (Subgraph extraction & counterfactual simulation)  │
│ 11. Scientific Audit Report & Exporters (Streamlit, REST API, PDF)                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Data Sources (100% Free & Open)

| Data Source | Domain / Coverage | Direct Access |
| :--- | :--- | :--- |
| **ChEMBL** | Drug bioactivities, primary targets, action mechanisms | [ebi.ac.uk/chembl](https://www.ebi.ac.uk/chembl/) |
| **UniProt** | Human protein accessions, canonical sequence mapping | [uniprot.org](https://www.uniprot.org/) |
| **Reactome** | Pathways, reaction events, molecular participant roles | [reactome.org](https://reactome.org/) |
| **Open Targets** | Disease–gene associations, Direction-of-Effect (DoE) | [platform.opentargets.org](https://platform.opentargets.org/) |
| **DisGeNET** | Gene–disease evidence scores, actionable trajectories | [disgenet.org](https://www.disgenet.org/) |
| **PubMed** | MEDLINE citations, biomedical literature abstracts | [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/) |
| **Europe PMC** | Open-access articles, full text, external links | [europepmc.org](https://europepmc.org/) |
| **ClinicalTrials.gov** | Clinical trial protocols, status, `whyStopped` logs | [clinicaltrials.gov](https://clinicaltrials.gov/) |
| **Semantic Scholar** | Citation-weighted scientific literature metadata | [semanticscholar.org](https://www.semanticscholar.org/) |
| **OpenAlex** | Global scholarly bibliographic metadata | [openalex.org](https://openalex.org/) |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/sameekshaa19/cynthera.git
cd cynthera

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:
```bash
copy .env.example .env   # Windows
# or: cp .env.example .env
```

Set your preferred LLM provider in `.env`:
```env
# LLM Extraction Provider (Groq or Gemini)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Higher rate-limit keys for PubMed / Semantic Scholar
NCBI_API_KEY=your_ncbi_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```
> **Note:** CYNTHERA operates seamlessly with automatic rule-based extractors even if LLM keys are omitted.

---

## 💻 Usage

### Command-Line Interface (CLI)

Evaluate a drug-disease hypothesis from the command line:
```bash
python main.py --drug "Sildenafil" --disease "Pulmonary Arterial Hypertension"
```

Bypass local cache for live real-time API retrieval:
```bash
python main.py --drug "Duloxetine" --disease "Diabetic Neuropathy" --no-cache
```

Save the full structured scientific report to JSON:
```bash
python main.py --drug "Testosterone" --disease "Prostate Cancer" --output report.json
```

### Interactive Streamlit Web Interface & Playground

Launch the full web dashboard:
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser.

- **Hypothesis Investigation:** Input drug and disease names to generate multi-hop biological chains, evidence cards, and direct source links.
- **Investigation Playground:** Inspect dynamic causal subgraphs, filter by evidence polarity, and simulate "what-if" counterfactual scenarios (e.g., target inhibition/knockout effects).
- **PDF Report Export:** Download publication-ready scientific audit reports with clickable source links.

### FastAPI REST Service

Launch the async REST API:
```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```
- **Interactive Swagger Documentation:** `http://localhost:8000/docs`
- **Primary Analysis Endpoint:** `POST /api/v1/analyze`
- **Playground Subgraph Endpoint:** `GET /playground/subgraph?investigation_id=...`
- **Counterfactual Simulation:** `POST /playground/simulate`
- **Health Check:** `GET /api/v1/health`

---

## 📊 Sample Output

CYNTHERA formats assessments as a structured research narrative:

```text
==================================================
CYNTHERA — DRUG REPURPOSING ASSESSMENT
==================================================

Drug:     Sildenafil
Disease:  Pulmonary Arterial Hypertension

--------------------------------------------------
OVERALL ASSESSMENT
--------------------------------------------------
Mechanistic Plausibility : HIGH
Score                    : 0.82
Therapeutic Alignment    : SUPPORTS
Evidence Status          : WELL SUPPORTED
Recommendation Status    : PROMISING

--------------------------------------------------
WHY? (Per-Hop Mechanistic Chain)
--------------------------------------------------

1. Drug → Target
   Sildenafil → PDE5A (Inhibitor)
   Evidence: ChEMBL, PubMed, Europe PMC
   [Open ChEMBL] [Open Paper] [Open DOI]

2. Target → Reaction / Pathway
   PDE5A → cGMP degradation → cGMP signaling pathway
   Polarity: NEGATIVE (-) regulation of cGMP breakdown
   Evidence: Reactome, UniProt
   [Open Reactome] [Open Paper]

3. Pathway → Disease Target / Mediator
   cGMP accumulation → PKG activation → Pulmonary Vasodilation
   Evidence: Open Targets, DisGeNET, PubMed
   [Open Open Targets] [Open Paper]

--------------------------------------------------
SUPPORTING EVIDENCE (Literature & Trials)
--------------------------------------------------
Paper 1: Phosphodiesterase-5 inhibitors in pulmonary arterial hypertension...
  Authors: Ghofrani HA, et al. | Year: 2004 | PMID: 15329302
  [Open PubMed] [Open Europe PMC] [Open DOI] [Open Full Text / PDF]

Clinical Trial: SUPER-1 (NCT00078026)
  Phase: Phase 3 | Status: Completed | Result: Efficacy demonstrated
  [Open ClinicalTrials.gov]

--------------------------------------------------
CONTRADICTING / LIMITING EVIDENCE
--------------------------------------------------
None identified meeting opposition criteria.

--------------------------------------------------
SOURCES ACCESSED
--------------------------------------------------
ChEMBL · UniProt · Reactome · Open Targets · DisGeNET ·
PubMed · Europe PMC · ClinicalTrials.gov · Semantic Scholar
```

---

## 🔗 Evidence & Traceability Model

Every conclusion is traceable from top-level recommendation down to primary source URLs:

```
Recommendation / Verdict
    ↓
Candidate Mechanism Assessment
    ↓
Claim ID (Polarity: SUPPORT / OPPOSE / NEUTRAL)
    ↓
Evidence IDs (Applicability Gated)
    ↓
Primary Database Identifier (PMID / DOI / ChEMBL ID / NCT ID)
    ↓
Direct Resource Access URL
```

Direct access buttons distinguish:
- `[Open PubMed]` — Official NCBI PubMed record
- `[Open Europe PMC]` — Europe PMC repository entry
- `[Open DOI]` — Publisher digital object identifier
- `[Open ClinicalTrials.gov]` — NCT clinical trial registry entry
- `[Open Full Text / PDF]` — Available **only** when a legal open-access full text/PDF is confirmed

When the same paper is retrieved across multiple aggregators, it is deduplicated into a single unified record with multiple legitimate access options.

---

## 🛑 Failure-State Handling

Retrieval failures and database absences are **never** converted into scientific claims of biological implausibility:

| State | Meaning |
| :--- | :--- |
| `DATA NOT FOUND` | Query succeeded but no data exists for this biological entity. |
| `SOURCE UNAVAILABLE` | Upstream API failed or timed out (e.g., ChEMBL HTTP 500). |
| `IDENTITY RESOLUTION FAILED` | Drug or disease name could not be resolved to a canonical database ID. |
| `INSUFFICIENT EVIDENCE` | Inadequate evidence to evaluate mechanism — not a claim of implausibility. |
| `CONTRADICTORY EVIDENCE` | Retrieved evidence exhibits genuine biological or clinical conflict. |
| `MECHANISTICALLY UNSUPPORTED`| Complete biological evidence was retrieved but does not connect the target to the disease. |
| `MECHANISTICALLY PLAUSIBLE`  | Retrieved biological evidence confirms an interconnected multi-hop cascade. |

---

## 📐 Scoring & Decision Rules

Scores are **deterministic, evidence-grounded, and mathematically explainable**:

- **Support Score (SS)**: Quantitative strength of retrieved, applicability-gated literature claims.
- **Mechanistic Score (MS)**: Plausibility of the biological cascade (`Drug → Target → Pathway → Disease Gene → Disease`), derived from target binding affinity, pathway participation, and disease-gene association scores.
- **Risk Score (RS)**: Clinical safety signals extracted from adverse event profiles and clinical trial termination records (`whyStopped`).
- **Therapeutic Alignment**: Evaluated across Direction-of-Effect (DoE) and Actionable Target Trajectories (DATTs) to yield `SUPPORTS`, `OPPOSES`, or `INSUFFICIENT`.

---

## 🧪 Benchmark & Evaluation Framework

CYNTHERA includes standardized evaluation datasets grounded in clinical literature and pharmacological databases ([`backend/evaluation/`](backend/evaluation/)):

### Benchmark Composition & Contradiction Controls
The framework validates system performance across curated positive clinical indications, negative controls, and verified biological contradiction pairs:
- *Testosterone $\rightarrow$ Prostate Cancer* (`OPPOSES` — androgen receptor agonist promotes prostate tumor proliferation)
- *Digoxin $\rightarrow$ Wolff-Parkinson-White Syndrome* (`OPPOSES` — AV nodal blockade promotes lethal accessory pathway conduction)
- *Salbutamol $\rightarrow$ Hypertrophic Cardiomyopathy* (`OPPOSES` — $\beta_2$ agonist exacerbates dynamic left ventricular outflow tract obstruction)
- *Verapamil $\rightarrow$ Heart Failure with Reduced Ejection Fraction* (`OPPOSES` — negative inotrope worsens systolic failure)
- *Finasteride $\rightarrow$ ER+ Breast Cancer* (`OPPOSES` — $5\alpha$-reductase inhibition increases substrate aromatization into estrogen)

### Evaluation Suites
- **24-Case Benchmark Dataset** (`TEST` and `DEVELOPMENT` splits)
- **30-Case Holdout Suite** (`backend/evaluation/run_30_case_holdout.py`)
- **100-Case Frozen Evaluation** (`backend/evaluation/run_100_case_evaluation.py`):
  Evaluates 3-class classification (`SUPPORT`, `OPPOSE`, `UNCERTAIN`), Accuracy, Macro F1, and Matthews Correlation Coefficient (MCC) with strict evaluation invariants.

---

## 📁 Project Structure

```
cynthera/
├── backend/
│   ├── api/                          # FastAPI REST application layer & routes
│   │   ├── analyze_routes.py         #   Core /api/v1/analyze endpoints & DTO builders
│   │   └── main.py                   #   FastAPI app instantiation & CORS setup
│   ├── core/
│   │   ├── domain/                   # Domain entities (CandidateMechanism, RetrievalPackage, Claim, Evidence...)
│   │   ├── enums/                    # Typed enums (EvidencePolarity, RecommendationStatus, EvidenceType...)
│   │   └── value_objects/            # Value objects (ERW, ProvenanceReference, CanonicalIdentifier...)
│   ├── engineering/
│   │   ├── identity/                 # Biological identifier resolution (ChEMBL, UniProt, MeSH, MONDO)
│   │   ├── orchestrator/             # MasterOrchestrator deterministic retrieval coordinator
│   │   └── retrieval/                # Async connectors (ChEMBL, Reactome, Open Targets, PubMed, ClinicalTrials...)
│   ├── evaluation/                   # 100-case & 24-case benchmark datasets, evaluation runners, forensic audits
│   ├── infrastructure/               # SQLite cache layer (`cynthera.db`), logging, persistence
│   ├── playground/                   # Investigation Playground backend (subgraph extraction, scenario simulation)
│   ├── reasoning/
│   │   ├── agents/                   # Clinical safety agent, prior knowledge agent
│   │   ├── conflict/                 # Advanced conflict resolver & contradiction detector
│   │   ├── context/                  # Scientific context builder
│   │   ├── directional/              # Evidence polarity classifier & therapeutic evidence applicability gates
│   │   ├── extraction/               # LLM claim extraction agent with rule-based fallback
│   │   ├── mechanistic/              # Evidence graph builder, multi-hop DFS pathfinder, mechanism validator
│   │   ├── opposition/               # Semantic opposition qualification & propagation
│   │   └── orchestrator/             # Reasoning orchestrator & deterministic decision rules
│   └── reporting/                    # PDF report generator & export utilities
├── frontend/
│   ├── app.py                        # Streamlit main interactive dashboard
│   └── playground_view.py            # Streamlit Investigation Playground view
├── tests/
│   └── unit/                         # 366 unit tests covering all modules & invariants
├── data/                             # SQLite persistent cache database
├── main.py                           # CLI entry point
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment configuration template
└── README.md                         # Project documentation
```

---

## 🧪 Running Tests

CYNTHERA maintains 366 unit tests verifying dataflow correctness, directional invariants, and edge cases:

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run evidence polarity classification tests
python -m pytest tests/unit/test_evidence_polarity.py -v

# Run therapeutic opposition & propagation tests
python -m pytest tests/unit/test_therapeutic_opposition.py -v

# Run clinical trial safe parsing & attribution tests
python -m pytest tests/unit/test_trial_attribution.py tests/unit/test_clinicaltrials_safe_fixes.py -v

# Run playground & scenario simulation tests
python -m pytest tests/unit/test_playground_and_risk_audit.py -v
```

---

## 📝 License & Disclaimer

This project is licensed under the **MIT License**.

**Disclaimer:** CYNTHERA is designed exclusively for scientific research, biological hypothesis generation, and educational analysis. It does not provide medical advice, clinical diagnoses, or treatment plans. All computational hypotheses, mechanistic chains, and scores must be evaluated and verified by qualified domain experts before any experimental or clinical application.
