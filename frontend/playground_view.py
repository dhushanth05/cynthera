"""CYNTHERA Playground — Interactive Hypothesis Exploration & Scenario Workspace.

Provides the interactive evidence graph, 'Why is this relationship here?'
inspector, evidence landscape, honest scenario recomputation (MS-only),
and researcher notes.
"""
from __future__ import annotations

import logging
from typing import Any

import streamlit as st
import plotly.graph_objects as go

from backend.storage.repository import StorageRepository
from backend.playground.subgraph_extractor import extract_relevant_subgraph
from backend.playground.scenario_engine import ScenarioEngine
from backend.playground.investigation_store import InvestigationStore
from backend.playground.models import (
    PlaygroundGraphData,
    ScenarioModifications,
    ScenarioModification,
    ResearcherNote,
)
from backend.core.value_objects.source_url_builder import SourceURLBuilder

logger = logging.getLogger(__name__)

# Node type color mapping
NODE_COLORS = {
    "DRUG": "#3b82f6",     # Blue
    "TARGET": "#8b5cf6",   # Purple
    "GENE": "#10b981",     # Emerald green
    "PATHWAY": "#f59e0b",  # Amber
    "DISEASE": "#ef4444",  # Red
}

NODE_BASE_SIZES = {
    "DRUG": 36,
    "DISEASE": 36,
    "TARGET": 26,
    "PATHWAY": 26,
    "GENE": 18,
    "REACTION": 18,
}


def _short_label(name: str, limit: int = 18) -> str:
    """Keep dense graphs readable while preserving full names in hover text."""
    clean = " ".join(str(name).split())
    return clean if len(clean) <= limit else f"{clean[:limit - 1]}…"


def _canonical_node_display_name(node: PlaygroundNode) -> str:
    """Generate a clean, readable canonical symbol for graph rendering."""
    if node.label == "TARGET":
        sym = node.meta.get("gene_symbol") or node.meta.get("symbol")
        if sym:
            return str(sym).upper()
        name = node.name.split("_")[0] if "_" in node.name else node.name
        if len(name) <= 10 and not name.startswith("CHEMBL"):
            return name
        return node.id.replace("TARGET:", "")
    elif node.label == "PATHWAY":
        clean = node.name
        if clean.startswith("Pathway:"):
            clean = clean.split("Pathway:", 1)[1].strip()
        return clean if len(clean) <= 18 else f"{clean[:16]}…"
    elif node.label in ("DRUG", "DISEASE"):
        return node.name if len(node.name) <= 20 else f"{node.name[:18]}…"
    elif node.label == "GENE":
        return node.name
    return _short_label(node.name, limit=14)


def _get_mechanism_node_and_edge_ids(
    cm: dict[str, Any],
    nodes: list[PlaygroundNode],
    edges: list[PlaygroundEdge],
) -> tuple[set[str], set[str]]:
    """Identify nodes and edges traversing a candidate mechanism."""
    matched_nodes = set()
    hops = cm.get("hops", []) if isinstance(cm, dict) else getattr(cm, "hops", [])
    hop_tokens = set()
    for h in hops:
        for key in ("from_node", "to_node", "canonical_from_id", "canonical_to_id"):
            val = str(h.get(key, "") if isinstance(h, dict) else getattr(h, key, "")).strip().upper()
            if val:
                hop_tokens.add(val)
                hop_tokens.add(val.split(":")[-1].strip())

    for n in nodes:
        n_id_u = n.id.upper()
        n_name_u = n.name.upper()
        clean_id = n_id_u.split(":")[-1].strip()
        if (
            n.label in ("DRUG", "DISEASE")
            or n_id_u in hop_tokens
            or n_name_u in hop_tokens
            or clean_id in hop_tokens
            or any(t in n_id_u or t in n_name_u for t in hop_tokens if len(t) > 3)
        ):
            matched_nodes.add(n.id)

    matched_edges = set()
    for e in edges:
        if e.source_id in matched_nodes and e.target_id in matched_nodes:
            matched_edges.add(e.id)
            matched_edges.add(f"EDGE:{e.source_id}->{e.target_id}:{e.predicate}")
    return matched_nodes, matched_edges


def _get_overview_subgraph(
    graph_data: PlaygroundGraphData,
) -> tuple[list[PlaygroundNode], list[PlaygroundEdge]]:
    """Extract clean Overview graph: Drug -> Top Targets -> Candidate Pathways -> Disease."""
    overview_node_ids = set()
    for n in graph_data.nodes:
        if n.label in ("DRUG", "DISEASE"):
            overview_node_ids.add(n.id)

    cms = graph_data.candidate_mechanisms
    if cms:
        for cm in cms[:3]:
            m_nodes, _ = _get_mechanism_node_and_edge_ids(cm, graph_data.nodes, graph_data.edges)
            overview_node_ids.update(m_nodes)
    else:
        targets = [n for n in graph_data.nodes if n.label == "TARGET"]
        targets.sort(key=lambda t: t.evidence_count, reverse=True)
        top_targets = {t.id for t in targets[:2]}
        overview_node_ids.update(top_targets)
        for e in graph_data.edges:
            if e.source_id in top_targets and any(n.id == e.target_id and n.label == "PATHWAY" for n in graph_data.nodes):
                overview_node_ids.add(e.target_id)

    overview_nodes = [n for n in graph_data.nodes if n.id in overview_node_ids]
    overview_edges = [
        e for e in graph_data.edges
        if e.source_id in overview_node_ids and e.target_id in overview_node_ids
    ]
    return overview_nodes, overview_edges


def _playground_summary(graph_data: PlaygroundGraphData, result: Any = None) -> dict[str, int]:
    """Return display counts sourced from the authoritative EvidenceSummary or extracted result."""
    es = getattr(getattr(result, "audit_report", None), "evidence_summary", None) or getattr(result, "evidence_summary", None)
    total_ev = es.total_retrieved_records if (es and hasattr(es, "total_retrieved_records")) else len(graph_data.evidence)
    mechs = es.candidate_mechanisms_count if (es and hasattr(es, "candidate_mechanisms_count")) else len(graph_data.candidate_mechanisms)
    claims = es.literature_claims_extracted if (es and hasattr(es, "literature_claims_extracted")) else len(graph_data.claim_citations)
    trials = es.clinical_trials_evaluated if (es and hasattr(es, "clinical_trials_evaluated")) else len(graph_data.clinical_trials)
    return {
        "mechanisms": mechs,
        "evidence": total_ev,
        "claims": claims,
        "trials": trials,
        "contradictions": len(graph_data.contradictions),
        "mapped_supporting": es.subgraph_mapped_records if (es and hasattr(es, "subgraph_mapped_records")) else graph_data.landscape.supporting_count,
    }


def _presentation_gaps(graph_data: PlaygroundGraphData, result: Any) -> list[str]:
    """Merge existing extractor diagnostics with existing result data-gap messages."""
    gaps = [gap.label for gap in graph_data.evidence_gaps if gap.status != "FOUND"]
    for gap in getattr(result, "data_gaps", []) or []:
        text = str(gap)
        if text and text not in gaps:
            gaps.append(text)
    return gaps


def _get_or_compute_layout(
    nodes: list[PlaygroundNode] | PlaygroundGraphData | Any,
    cache_key: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute a readable left-to-right layered DAG layout with deterministic spacing."""
    if cache_key and cache_key in st.session_state:
        return st.session_state[cache_key]

    actual_nodes = nodes.nodes if hasattr(nodes, "nodes") else nodes

    type_tiers = {
        "DRUG": 0.0,
        "TARGET": 2.2,
        "PATHWAY": 4.4,
        "REACTION": 5.5,
        "GENE": 6.6,
        "DISEASE": 8.8,
    }

    nodes_by_type: dict[str, list[PlaygroundNode]] = {}
    for node in actual_nodes:
        nodes_by_type.setdefault(node.label, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    for label, node_list in nodes_by_type.items():
        base_x = type_tiers.get(label, 3.5)
        node_list.sort(key=lambda n: (-getattr(n, "evidence_count", 0), getattr(n, "name", "")))
        n_count = len(node_list)
        spacing = max(1.4, min(2.5, 9.0 / max(n_count, 1)))
        for idx, node in enumerate(node_list):
            y = (idx - (n_count - 1) / 2.0) * spacing
            pos[node.id] = (base_x, y)

    if cache_key:
        st.session_state[cache_key] = pos
    return pos


def _build_playground_figure(
    graph_data: PlaygroundGraphData,
    positions: dict[str, tuple[float, float]],
    visible_types: set[str],
    selected_element: str | None,
    selected_type: str | None,
    disabled_edges: set[str],
    graph_mode: str = "Overview",
    selected_cm_idx: int | None = None,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build the clean, research-usable Plotly network figure."""
    # 1. Filter active nodes & edges by mode
    if graph_mode == "Overview":
        active_nodes, active_edges = _get_overview_subgraph(graph_data)
        cm_highlight_nodes, cm_highlight_edges = set(), set()
    elif graph_mode == "Mechanisms":
        active_nodes, active_edges = graph_data.nodes, graph_data.edges
        if selected_cm_idx is not None and 0 <= selected_cm_idx < len(graph_data.candidate_mechanisms):
            cm = graph_data.candidate_mechanisms[selected_cm_idx]
            cm_highlight_nodes, cm_highlight_edges = _get_mechanism_node_and_edge_ids(cm, active_nodes, active_edges)
        else:
            cm_highlight_nodes, cm_highlight_edges = set(), set()
    elif graph_mode == "Evidence":
        active_nodes, active_edges = graph_data.nodes, graph_data.edges
        ev_nodes = {n.id for n in active_nodes if n.label in ("DRUG", "DISEASE") or n.evidence_count > 0}
        ev_edges = {
            e.id for e in active_edges
            if e.evidence_count > 0 or e.hop_claims or "DIRECT" in (e.data_quality or "").upper() or "EVIDENCE" in (e.data_quality or "").upper()
        }
        ev_edges.update({
            f"EDGE:{e.source_id}->{e.target_id}:{e.predicate}" for e in active_edges
            if e.evidence_count > 0 or e.hop_claims or "DIRECT" in (e.data_quality or "").upper() or "EVIDENCE" in (e.data_quality or "").upper()
        })
        cm_highlight_nodes = ev_nodes
        cm_highlight_edges = ev_edges
    else:  # "Full Graph"
        active_nodes, active_edges = graph_data.nodes, graph_data.edges
        cm_highlight_nodes, cm_highlight_edges = set(), set()

    visible_node_ids = {n.id for n in active_nodes if n.label in visible_types}

    # Edges grouped by epistemic status
    direct_x, direct_y = [], []
    bridge_x, bridge_y = [], []
    struct_x, struct_y = [], []
    disabled_x, disabled_y = [], []
    dimmed_x, dimmed_y = [], []
    selected_edge_traces = []

    mid_x, mid_y, mid_customdata, mid_hovertext = [], [], [], []
    edge_lookup: dict[str, Any] = {}

    has_cm_focus = len(cm_highlight_nodes) > 0

    for edge in active_edges:
        if edge.source_id not in visible_node_ids or edge.target_id not in visible_node_ids:
            continue
        p0 = positions.get(edge.source_id, (0.0, 0.0))
        p1 = positions.get(edge.target_id, (1.0, 0.0))

        edge_key = f"EDGE:{edge.source_id}->{edge.target_id}:{edge.predicate}"
        edge_lookup[edge_key] = edge

        is_disabled = edge_key in disabled_edges or edge.is_disabled
        is_selected = (selected_type == "edge" and selected_element == edge_key)
        is_in_cm = (edge.id in cm_highlight_edges or edge_key in cm_highlight_edges) if has_cm_focus else False

        # Midpoint for hover & click
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        mid_x.append(mx)
        mid_y.append(my)
        mid_customdata.append(edge_key)
        status_tag = " [DISABLED]" if is_disabled else ""
        mid_hovertext.append(
            f"<b>{edge.predicate}</b>{status_tag}<br>"
            f"Epistemic: {edge.data_quality}<br>"
            f"Strength: {edge.evidence_strength:.2f}<br>"
            f"Source: {edge.source_database or 'Curated DB'}"
        )

        if is_selected:
            selected_edge_traces.append(
                go.Scatter(
                    x=[p0[0], p1[0]],
                    y=[p0[1], p1[1]],
                    mode="lines",
                    line=dict(color="#06b6d4", width=4.5),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        if is_disabled:
            disabled_x.extend([p0[0], p1[0], None])
            disabled_y.extend([p0[1], p1[1], None])
        elif has_cm_focus and not is_in_cm:
            dimmed_x.extend([p0[0], p1[0], None])
            dimmed_y.extend([p0[1], p1[1], None])
        else:
            tier_upper = (edge.data_quality or "").upper()
            src_upper = (edge.source_database or "").upper()
            if "DIRECT" in tier_upper or "EVIDENCE_BACKED" in tier_upper or src_upper in ("CHEMBL", "PUBMED"):
                direct_x.extend([p0[0], p1[0], None])
                direct_y.extend([p0[1], p1[1], None])
            elif "BRIDGE" in tier_upper or "CANDIDATE" in tier_upper or edge.predicate == "ASSOCIATED_WITH":
                bridge_x.extend([p0[0], p1[0], None])
                bridge_y.extend([p0[1], p1[1], None])
            else:  # Structural / Database supported (Reactome)
                struct_x.extend([p0[0], p1[0], None])
                struct_y.extend([p0[1], p1[1], None])

    edge_traces: list[go.Scatter] = []
    if direct_x:
        edge_traces.append(go.Scatter(
            x=direct_x, y=direct_y, mode="lines",
            line=dict(color="#38bdf8", width=2.5),
            hoverinfo="none", name="Direct / Evidence-Backed",
        ))
    if bridge_x:
        edge_traces.append(go.Scatter(
            x=bridge_x, y=bridge_y, mode="lines",
            line=dict(color="#f59e0b", width=2.0, dash="dash"),
            hoverinfo="none", name="Unvalidated Bridge",
        ))
    if struct_x:
        edge_traces.append(go.Scatter(
            x=struct_x, y=struct_y, mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.45)", width=1.5, dash="dot"),
            hoverinfo="none", name="Structural / Pathway Membership",
        ))
    if disabled_x:
        edge_traces.append(go.Scatter(
            x=disabled_x, y=disabled_y, mode="lines",
            line=dict(color="#ef4444", width=1.8, dash="dot"),
            hoverinfo="none", name="Disabled Relationship",
        ))
    if dimmed_x:
        edge_traces.append(go.Scatter(
            x=dimmed_x, y=dimmed_y, mode="lines",
            line=dict(color="rgba(100, 116, 139, 0.12)", width=1.0),
            hoverinfo="none", showlegend=False,
        ))

    # Transparent midpoint interactive marker trace (prevents permanent diamonds!)
    interactive_edge_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0.01)"),
        customdata=mid_customdata,
        hovertext=mid_hovertext,
        hoverinfo="text",
        name="Relationships (Hover)",
    )

    # Nodes
    node_x, node_y = [], []
    node_text, node_customdata = [], []
    node_colors, node_sizes = [], []
    node_border_widths, node_border_colors = [], []
    node_display_texts = []
    node_lookup: dict[str, Any] = {}

    for node in active_nodes:
        if node.id not in visible_node_ids:
            continue
        node_lookup[node.id] = node
        p = positions.get(node.id, (0.0, 0.0))
        node_x.append(p[0])
        node_y.append(p[1])

        is_selected = (selected_type == "node" and selected_element == f"NODE:{node.id}")
        is_cm_node = (node.id in cm_highlight_nodes) if has_cm_focus else True
        is_dimmed = has_cm_focus and not is_cm_node

        base_sz = NODE_BASE_SIZES.get(node.label, 18)
        sz = (base_sz + 6) if (is_selected or (has_cm_focus and is_cm_node and node.label in ("TARGET", "PATHWAY"))) else base_sz

        color = "rgba(100, 116, 139, 0.25)" if is_dimmed else NODE_COLORS.get(node.label, "#94a3b8")
        bw = 3 if is_selected else (2 if (has_cm_focus and is_cm_node) else 1)
        bc = "#06b6d4" if is_selected else ("#ffffff" if (has_cm_focus and is_cm_node) else "#1e293b")

        node_colors.append(color)
        node_sizes.append(sz)
        node_border_widths.append(bw)
        node_border_colors.append(bc)
        node_customdata.append(f"NODE:{node.id}")

        # Hover text with full IDs
        node_text.append(
            f"<b>{node.name}</b><br>"
            f"Type: {node.label}<br>"
            f"ID: {node.id}<br>"
            f"Direct Evidence: {node.evidence_count} records"
        )

        # Graph on-node text: clean canonical symbol
        if is_dimmed:
            node_display_texts.append("")
        else:
            node_display_texts.append(_canonical_node_display_name(node))

    nodes_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=node_border_widths, color=node_border_colors),
        ),
        text=node_display_texts,
        textposition="bottom center",
        textfont=dict(size=10, color="#f1f5f9"),
        customdata=node_customdata,
        hovertext=node_text,
        hoverinfo="text",
        name="Entities",
    )

    all_traces = edge_traces + selected_edge_traces + [interactive_edge_trace, nodes_trace]
    fig = go.Figure(data=all_traces)

    fig.update_layout(
        margin=dict(l=15, r=15, t=15, b=15),
        plot_bgcolor="#0b1120",
        paper_bgcolor="#0b1120",
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=max(550, min(850, 420 + max(len(active_nodes), 1) * 7)),
        clickmode="event+select",
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94a3b8", size=10),
        ),
    )

    # Focus / fit on highlighted candidate mechanism if active
    if has_cm_focus and any(nid in positions for nid in cm_highlight_nodes):
        cm_x = [positions[nid][0] for nid in cm_highlight_nodes if nid in positions]
        cm_y = [positions[nid][1] for nid in cm_highlight_nodes if nid in positions]
        if cm_x and cm_y:
            fig.update_xaxes(range=[min(cm_x) - 1.2, max(cm_x) + 1.2])
            fig.update_yaxes(range=[min(cm_y) - 1.2, max(cm_y) + 1.2])

    return fig, {"edges": edge_lookup, "nodes": node_lookup}


def render_playground_view() -> None:
    """Main render entrypoint for the Playground view."""
    repo = StorageRepository(db_path="data/cynthera.db")
    investigation_store = InvestigationStore(db_path="data/cynthera.db")
    scenario_engine = ScenarioEngine()

    st.markdown("## 🔍 CYNTHERA Playground")
    st.caption("Interactive Evidence Graph Exploration, Hop-level Traceability & Scenario Workspace")

    # ─────────────────────────────────────────────
    # Hypothesis Selection Bar
    # ─────────────────────────────────────────────
    evaluations = repo.list_evaluations(limit=100)
    if not evaluations and not st.session_state.get("results"):
        st.info("No completed evaluations found. Go to **🔬 Evaluate** to run an analysis first.")
        return

    # Build evaluation options
    eval_options: dict[str, str] = {}
    if st.session_state.get("results"):
        curr_res = st.session_state.results["result"]
        curr_pkg = st.session_state.results["package"]
        label = f"Current: {curr_pkg.drug.name} ➔ {curr_pkg.disease.name} ({curr_res.recommendation_status.value})"
        eval_options[label] = str(curr_res.hypothesis_id)

    for ev in evaluations:
        label = f"{ev.get('drug_name')} ➔ {ev.get('disease_name')} [{ev.get('recommendation')}] ({ev.get('hypothesis_id', '')[:8]})"
        eval_options[label] = ev.get("hypothesis_id", "")

    # Selected hypothesis ID
    default_idx = 0
    target_hyp_id = st.session_state.get("playground_hyp_id")
    if target_hyp_id:
        for idx, hid in enumerate(eval_options.values()):
            if hid == target_hyp_id:
                default_idx = idx
                break

    col_sel, col_reset = st.columns([5, 1])
    with col_sel:
        selected_label = st.selectbox(
            "Select Evaluation to Explore:",
            options=list(eval_options.keys()),
            index=default_idx,
            label_visibility="collapsed",
        )
    with col_reset:
        if st.button("🔄 Reset Graph View", use_container_width=True):
            st.session_state.pop("playground_selected_element", None)
            st.session_state.pop("playground_selected_type", None)
            target_id = eval_options.get(selected_label)
            if target_id:
                st.session_state.pop(f"layout_{target_id}", None)
            st.rerun()

    active_hyp_id = eval_options.get(selected_label)
    if not active_hyp_id:
        st.warning("Please select a valid evaluation.")
        return

    # Load Package and ReasoningResult
    pkg = repo.get_retrieval_package(active_hyp_id)
    res = repo.get_reasoning_result(active_hyp_id)

    # Fallback to session_state if active is currently running in memory
    if (pkg is None or res is None) and st.session_state.get("results"):
        if str(st.session_state.results["result"].hypothesis_id) == active_hyp_id:
            pkg = st.session_state.results["package"]
            res = st.session_state.results["result"]

    if pkg is None or res is None:
        st.error(f"Could not load data for hypothesis `{active_hyp_id}`.")
        return

    # Extract Graph Data
    graph_data = extract_relevant_subgraph(pkg, res)
    summary = _playground_summary(graph_data, res)
    presentation_gaps = _presentation_gaps(graph_data, res)

    st.markdown("### Evidence landscape")
    es = getattr(getattr(res, "audit_report", None), "evidence_summary", None) or getattr(res, "evidence_summary", None)
    val_note = f" ({es.validated_mechanisms_count} valid)" if es and hasattr(es, "validated_mechanisms_count") else ""
    fail_note = f" ({es.clinical_trials_failed} failed)" if es and hasattr(es, "clinical_trials_failed") else ""
    mapped_note = f"Total retrieved: {summary['evidence']} | Subgraph mapped: {es.subgraph_mapped_records}" if es and hasattr(es, "subgraph_mapped_records") else ""

    sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
    sm1.metric("Mechanisms", f"{summary['mechanisms']}{val_note}", help="Evaluated biological hypotheses connecting drug targets to disease pathology")
    sm2.metric("Evidence Records", summary["evidence"], help=mapped_note if mapped_note else "Total records retrieved from PubMed, Europe PMC, ChEMBL, and databases")
    sm3.metric("Literature Claims", summary["claims"], help="Biomedical claims extracted from peer-reviewed literature")
    sm4.metric("Clinical Trials", f"{summary['trials']}{fail_note}", help="Human clinical trials evaluated from ClinicalTrials.gov")
    sm5.metric("Contradictions", summary["contradictions"], help="Contradictory findings registered across literature claims")
    sm6.metric("Evidence Gaps", len(presentation_gaps), help="Existing backend/extractor diagnostics; not a new score.")

    if summary["mapped_supporting"] != summary["evidence"]:
        st.caption(
            f"Mapped supporting evidence: {summary['mapped_supporting']} of {summary['evidence']} records. "
            "This is a graph-mapping count, not the Support Score."
        )

    if presentation_gaps:
        with st.expander(f"Evidence gaps present ({len(presentation_gaps)})", expanded=False):
            for gap in presentation_gaps:
                st.markdown(f"- {gap}")
    else:
        st.caption("No evidence gaps detected by the existing backend diagnostics.")

    # State initialization for this hypothesis
    disabled_edges_key = f"disabled_edges_{active_hyp_id}"
    if disabled_edges_key not in st.session_state:
        st.session_state[disabled_edges_key] = set()
    disabled_edges: set[str] = st.session_state[disabled_edges_key]

    selected_element = st.session_state.get("playground_selected_element")
    selected_type = st.session_state.get("playground_selected_type")

    # ─────────────────────────────────────────────
    # Main Playground 2-Column Layout
    # ─────────────────────────────────────────────
    col_graph, col_sidebar = st.columns([7, 5])

    # Compute layout
    positions = _get_or_compute_layout(graph_data)

    with col_graph:
        # View mode selector and candidate mechanism selector
        col_mode, col_cm = st.columns([1, 1])
        with col_mode:
            graph_mode = st.radio(
                "Graph View Mode:",
                options=["Overview", "Mechanisms", "Evidence", "Full Graph"],
                index=0,
                horizontal=True,
                key=f"graph_mode_{active_hyp_id}",
            )

        selected_cm_idx = None
        cms = graph_data.candidate_mechanisms
        if graph_mode == "Mechanisms" and cms:
            with col_cm:
                cm_options = []
                for i, cm in enumerate(cms):
                    cm_name = cm.get("name") if isinstance(cm, dict) else getattr(cm, "name", f"Mechanism {i+1}")
                    cm_lvl = cm.get("support_level") if isinstance(cm, dict) else getattr(cm, "support_level", "")
                    cm_options.append(f"[{i+1}] {cm_name} ({cm_lvl})")

                selected_cm_str = st.selectbox(
                    "Candidate Mechanism Selector:",
                    options=cm_options,
                    index=0,
                    key=f"selected_cm_{active_hyp_id}",
                )
                if selected_cm_str:
                    selected_cm_idx = cm_options.index(selected_cm_str)

        # Filters toolbar
        all_labels = sorted(list({n.label for n in graph_data.nodes}))
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            visible_types = set(
                st.multiselect(
                    "Filter Node Types:",
                    options=all_labels,
                    default=all_labels,
                    label_visibility="collapsed",
                )
            )
        with col_f2:
            st.caption(f"**{len(graph_data.nodes)}** nodes, **{len(graph_data.edges)}** edges")

        fig, lookups = _build_playground_figure(
            graph_data,
            positions,
            visible_types,
            selected_element,
            selected_type,
            disabled_edges,
            graph_mode=graph_mode,
            selected_cm_idx=selected_cm_idx,
        )

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["points"],
            key=f"plotly_chart_{active_hyp_id}",
        )

        # Process click events
        if event and "selection" in event and event["selection"]["points"]:
            pt = event["selection"]["points"][0]
            cdata = pt.get("customdata")
            if cdata:
                if cdata.startswith("NODE:"):
                    st.session_state["playground_selected_type"] = "node"
                    st.session_state["playground_selected_element"] = cdata
                elif cdata.startswith("EDGE:"):
                    st.session_state["playground_selected_type"] = "edge"
                    st.session_state["playground_selected_element"] = cdata

        # Legend footer
        st.markdown(
            """
            <div style="display: flex; gap: 1rem; font-size: 0.8rem; color: #94a3b8; justify-content: center; margin-top: 0.25rem; flex-wrap: wrap;">
                <span><span style="color: #3b82f6;">●</span> Drug</span>
                <span><span style="color: #8b5cf6;">●</span> Target</span>
                <span><span style="color: #10b981;">●</span> Gene</span>
                <span><span style="color: #f59e0b;">●</span> Pathway</span>
                <span><span style="color: #ef4444;">●</span> Disease</span>
                <span><span style="color: #38bdf8; font-weight: bold;">―</span> Direct / Evidence</span>
                <span><span style="color: #f59e0b; font-weight: bold;">- -</span> Unvalidated Bridge</span>
                <span><span style="color: #94a3b8; font-weight: bold;">···</span> Structural</span>
                <span><span style="color: #06b6d4;">◆</span> Selected</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────
    # Right Sidebar: Inspector & Workspace Tabs
    # ─────────────────────────────────────────────
    with col_sidebar:
        tab_inspect, tab_landscape, tab_scenario, tab_notes = st.tabs([
            "❓ Inspector",
            "⚖️ Assessment",
            "🧪 Scenario",
            "📝 Notes",
        ])

        # ── Tab 1: "Why is this relationship here?" Inspector ────────────────
        with tab_inspect:
            active_cm = None
            if graph_mode == "Mechanisms" and selected_cm_idx is not None and 0 <= selected_cm_idx < len(graph_data.candidate_mechanisms):
                active_cm = graph_data.candidate_mechanisms[selected_cm_idx]

            if active_cm:
                cm_name = active_cm.get("name") if isinstance(active_cm, dict) else getattr(active_cm, "name", "")
                cm_level = active_cm.get("support_level") if isinstance(active_cm, dict) else getattr(active_cm, "support_level", "")
                cm_conf = active_cm.get("confidence_score") if isinstance(active_cm, dict) else getattr(active_cm, "confidence_score", 0.0)
                cm_chain = active_cm.get("summary_chain") if isinstance(active_cm, dict) else getattr(active_cm, "summary_chain", [])
                cm_hops = active_cm.get("hops") if isinstance(active_cm, dict) else getattr(active_cm, "hops", [])
                cm_cits = active_cm.get("literature_citations") if isinstance(active_cm, dict) else getattr(active_cm, "literature_citations", [])

                badge_color = "#10b981" if "STRONGLY" in cm_level else ("#f59e0b" if "MODERATELY" in cm_level else "#ef4444")
                st.markdown(f"### 🧬 Candidate Mechanism [{selected_cm_idx + 1}]")
                st.markdown(f"**{cm_name}**")
                c1, c2 = st.columns(2)
                c1.markdown(f"**Support:** <span style='color:{badge_color}; font-weight:bold;'>{cm_level.replace('_', ' ')}</span>", unsafe_allow_html=True)
                c2.metric("Confidence", f"{cm_conf:.1%}")

                if cm_chain:
                    st.markdown("#### Summary Chain")
                    st.info(" ➔ ".join(cm_chain))

                if cm_hops:
                    st.markdown("#### Step-by-Step Biological Hops")
                    for h_idx, h in enumerate(cm_hops):
                        fn = h.get("from_node", "") if isinstance(h, dict) else getattr(h, "from_node", "")
                        tn = h.get("to_node", "") if isinstance(h, dict) else getattr(h, "to_node", "")
                        pred = h.get("predicate", "") if isinstance(h, dict) else getattr(h, "predicate", "")
                        stat = h.get("status", "") if isinstance(h, dict) else getattr(h, "status", "")
                        src = h.get("source_database", "") if isinstance(h, dict) else getattr(h, "source_database", "")
                        h_links = h.get("links", []) if isinstance(h, dict) else getattr(h, "links", [])

                        with st.expander(f"Hop {h_idx + 1}: {fn} ➔ {pred} ➔ {tn}", expanded=(h_idx == 0)):
                            st.write(f"• **Status:** `{stat}`")
                            st.write(f"• **Source DB:** `{src}`")
                            for lk in h_links:
                                url = lk.get("url") if isinstance(lk, dict) else getattr(lk, "url", None)
                                lbl = (lk.get("display_label") if isinstance(lk, dict) else getattr(lk, "display_label", None)) or "View Source"
                                if url:
                                    st.markdown(f"• [{lbl}]({url})")

                if cm_cits:
                    st.markdown("#### Literature Citations")
                    for cit in cm_cits[:5]:
                        title = cit.get("title", "") if isinstance(cit, dict) else getattr(cit, "title", "")
                        url = cit.get("url", "") if isinstance(cit, dict) else getattr(cit, "url", "")
                        if url:
                            st.markdown(f"• [{title}]({url})")
                        elif title:
                            st.markdown(f"• {title}")

                st.markdown("---")

            if selected_type == "edge" and selected_element:
                edge = lookups["edges"].get(selected_element)
                if edge:
                    node_names = {node.id: node.name for node in graph_data.nodes}
                    source_name = node_names.get(edge.source_id, edge.source_id)
                    target_name = node_names.get(edge.target_id, edge.target_id)
                    st.markdown("### ❓ Why is this relationship here?")
                    st.markdown(
                        f"**{source_name}**  \n↓ **{edge.predicate}** ↓  \n**{target_name}**"
                    )
                    st.markdown("---")

                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("Paths", edge.path_count, help="Candidate mechanistic paths traversing this edge")
                    c_m2.metric("Evidence", edge.evidence_count, help="Direct evidence records supporting this connection")
                    c_m3.metric("Strength", f"{edge.evidence_strength:.2f}")

                    st.markdown("#### Provenance & Sources")
                    st.write(f"• **Source DB:** `{edge.source_database or 'Curated Biomedical DB'}`")
                    st.write(f"• **Evidence type:** `{edge.data_quality}`")
                    if (edge.source_database or "").lower() == "reactome" or edge.data_quality.upper() == "STRUCTURAL":
                        st.info(
                            "Structural pathway membership is shown for traceability; it does not by itself "
                            "establish a therapeutic or causal relationship."
                        )
                    else:
                        st.caption("This relationship is presented from the stored evidence graph and provenance.")
                    st.write(f"• **Quality:** `{edge.data_quality}`")
                    if edge.provenance:
                        st.write(f"• **Detail:** {edge.provenance}")

                    # Hop claims from candidate mechanisms
                    if edge.hop_claims:
                        st.markdown("#### Hop-Level Scientific Claims")
                        for hc in edge.hop_claims[:5]:
                            txt = hc.get("text", str(hc))
                            direction = hc.get("direction", "supporting")
                            icon = "✓" if direction == "supporting" else "⚠"
                            color = "#10b981" if direction == "supporting" else "#ef4444"
                            st.markdown(f"<span style='color: {color};'>{icon}</span> {txt}", unsafe_allow_html=True)

                    # External links
                    if edge.links:
                        st.markdown("#### Source Links")
                        for lk in edge.links:
                            url = lk.get("url")
                            lbl = lk.get("display_label") or lk.get("source_name") or "Open Source"
                            if url:
                                st.markdown(f"• [{lbl}]({url})")

                    st.markdown("---")
                    # Challenge / Disable button
                    is_dis = selected_element in disabled_edges
                    btn_text = "✅ Re-enable Relationship" if is_dis else "⚠️ Challenge & Disable Relationship"
                    if st.button(btn_text, key="toggle_edge_btn", type="secondary" if is_dis else "primary", use_container_width=True):
                        if is_dis:
                            disabled_edges.remove(selected_element)
                        else:
                            disabled_edges.add(selected_element)
                        st.rerun()
                else:
                    st.info("Select an edge on the graph to inspect why it was retrieved.")

            elif selected_type == "node" and selected_element:
                nid = selected_element.replace("NODE:", "")
                node = lookups["nodes"].get(nid)
                if node:
                    st.markdown(f"### Node: `{node.name}`")
                    st.write(f"• **Type:** `{node.label}`")
                    st.write(f"• **Identifier:** `{node.id}`")
                    st.write(f"• **Direct Evidence Count:** {node.evidence_count}")

                    # External links for this node
                    if node.links:
                        st.markdown("#### External Links")
                        for lk in node.links:
                            url = lk.get("url")
                            lbl = lk.get("display_label") or lk.get("source_name") or "View Record"
                            if url:
                                st.markdown(f"• [{lbl}]({url})")

                    # Connected relationships
                    connected = [e for e in graph_data.edges if e.source_id == nid or e.target_id == nid]
                    st.markdown(f"#### Connected Relationships ({len(connected)})")
                    for e in connected[:8]:
                        arrow = "➔" if e.source_id == nid else "⬅"
                        peer = e.target_id if e.source_id == nid else e.source_id
                        st.caption(f"{arrow} **{e.predicate}** with `{peer}` (Strength: {e.evidence_strength:.2f})")
                else:
                    st.info("Select a node on the graph to inspect details.")
            elif not active_cm:
                st.info("👈 Click any node (circle) or relationship on the graph to inspect provenance, paths, and details.")

        # ── Tab 2: Evidence Landscape & Gaps ────────────────────────────────
        with tab_landscape:
            ls = graph_data.landscape
            st.markdown("### ⚖️ Evidence Assessment")

            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.metric("Support Score (SS)", f"{ls.support_score:.3f}", ls.support_level)
                st.metric("Mechanistic Score (MS)", f"{ls.mechanistic_score:.3f}", ls.mechanistic_level)
            with c_s2:
                st.metric("Risk Score (RS)", f"{ls.risk_score:.3f}", ls.risk_level)
                st.metric("Opposition Score", f"{ls.opposition_score:.3f}", ls.opposition_level)

            st.markdown(f"**Recommendation:** `{ls.recommendation_status}`")
            for reason in ls.recommendation_reasons[:3]:
                st.caption(f"• {reason}")

            st.markdown("---")
            st.markdown("### 🧩 Evidence Gaps & Coverage")
            for gap in graph_data.evidence_gaps:
                if gap.status == "FOUND":
                    icon = "✅"
                    color = "#10b981"
                elif gap.status == "LIMITED":
                    icon = "⚡"
                    color = "#f59e0b"
                elif gap.status == "CONFLICTED":
                    icon = "⚠️"
                    color = "#ef4444"
                else:
                    icon = "❓"
                    color = "#64748b"
                st.markdown(f"<span style='color: {color};'>{icon} <b>{gap.label}</b></span>", unsafe_allow_html=True)

            if graph_data.contradictions:
                st.markdown("---")
                st.markdown("### ⚠️ Conflicting Findings")
                for c in graph_data.contradictions:
                    st.warning(f"**{c.conflict_type.upper()} Conflict:** {c.explanation} (Score: {c.contradiction_score:.2f})")

        # ── Tab 3: Scenario What-If Mode (Honest Recomputation) ──────────────
        with tab_scenario:
            st.markdown("### 🧪 Scenario What-If Mode")
            st.caption("Test how the hypothesis holds up when specific biological relationships are challenged or removed.")

            disabled_count = len(disabled_edges)
            st.write(f"**Challenged / Disabled Relationships:** {disabled_count}")

            if disabled_count == 0:
                st.info("No relationships are disabled. Click any relationship on the graph and select 'Challenge & Disable Relationship' to start a scenario.")
            else:
                for d_edge in list(disabled_edges):
                    c_txt, c_btn = st.columns([4, 1])
                    with c_txt:
                        st.caption(f"• `{d_edge.replace('EDGE:', '')}`")
                    with c_btn:
                        if st.button("✕", key=f"del_{d_edge}", help="Remove from disabled"):
                            disabled_edges.remove(d_edge)
                            st.rerun()

                # Build modifications
                mods = ScenarioModifications(
                    modifications=[
                        ScenarioModification(action="disable_edge", edge_id=d_edge)
                        for d_edge in disabled_edges
                    ]
                )

                # Compute scenario result
                scen_res = scenario_engine.compute_scenario(pkg, res, mods)

                st.markdown("---")
                st.markdown("#### Scenario Impact Comparison")

                # Comparison metrics
                c_orig, c_scen, c_delta = st.columns(3)
                c_orig.metric("Original MS", f"{scen_res.original_mechanistic_score:.3f}")
                c_scen.metric("Scenario MS", f"{scen_res.scenario_mechanistic_score:.3f}")
                ms_delta = scen_res.scenario_mechanistic_score - scen_res.original_mechanistic_score
                c_delta.metric("MS Delta", f"{ms_delta:+.3f}")

                # Recommendation shift
                st.markdown(
                    f"**Recommendation Shift:** `{scen_res.original_recommendation}` ➔ "
                    f"<span style='color: {'#10b981' if scen_res.scenario_recommendation == 'PROMISING' else '#f59e0b'}; font-weight: 700;'>"
                    f"{scen_res.scenario_recommendation}</span>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"• **Mechanistic Paths Lost:** {scen_res.affected_paths} (Remaining: {scen_res.scenario_path_count})")
                for chg in scen_res.changes_summary[:4]:
                    st.caption(f"• {chg}")

                st.markdown("---")
                # Honest disclaimer alert
                st.warning(
                    f"**Honest Evaluation Notice:** {scen_res.disclaimer}\n\n"
                    f"• **Recomputed:** `{scen_res.scores_recomputed}`\n"
                    f"• **Kept Original:** `{scen_res.scores_kept_original}`"
                )

                if st.button("Reset All Challenged Relationships", type="secondary"):
                    disabled_edges.clear()
                    st.rerun()

        # ── Tab 4: Researcher Notes (SQLite Persisted) ────────────────────────
        with tab_notes:
            st.markdown("### 📝 Researcher Notes")
            st.caption("Annotate and document your investigation findings. Persisted across sessions.")

            new_note_text = st.text_area("Add note:", height=90, placeholder="Document observations, rationale for challenged edges, or clinical context...")
            target_type_opt = st.selectbox("Attach note to:", ["hypothesis", "edge", "node"], index=0)

            target_id_val = None
            if target_type_opt == "edge" and selected_type == "edge":
                target_id_val = selected_element
            elif target_type_opt == "node" and selected_type == "node":
                target_id_val = selected_element

            if st.button("💾 Save Note", use_container_width=True, type="primary"):
                if new_note_text.strip():
                    note_obj = ResearcherNote(
                        hypothesis_id=active_hyp_id,
                        target_type=target_type_opt,
                        target_id=target_id_val,
                        text=new_note_text.strip(),
                    )
                    investigation_store.save_note(note_obj)
                    st.success("Note saved!")
                    st.rerun()
                else:
                    st.warning("Note cannot be empty.")

            st.markdown("---")
            existing_notes = investigation_store.get_notes(active_hyp_id)
            if existing_notes:
                st.markdown(f"**Saved Notes ({len(existing_notes)}):**")
                for nt in existing_notes:
                    c_ntext, c_ndel = st.columns([5, 1])
                    with c_ntext:
                        st.markdown(f"**[{nt.target_type.upper()}]** {nt.text}")
                        st.caption(f"Added: {nt.created_at.strftime('%Y-%m-%d %H:%M')}")
                    with c_ndel:
                        if st.button("🗑️", key=f"del_note_{nt.id}"):
                            investigation_store.delete_note(nt.id)
                            st.rerun()
            else:
                st.caption("No researcher notes added yet for this evaluation.")
