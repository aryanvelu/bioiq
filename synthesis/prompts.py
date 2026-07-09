"""
synthesis/prompts.py

All prompt-building functions for the BioIQ synthesis layer.

Functions here receive a context dict (from retrieve_context) and return a
fully-formed prompt string ready to pass to the Claude API.
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Research-brief prompt
# ---------------------------------------------------------------------------

_CATEGORY_LABELS = {
    "disease_modifying": "Disease-Modifying (targets causal gene/protein)",
    "symptomatic":       "Symptomatic (approved for this disease, manages symptoms)",
    "off_label":         "Off-Label (approved for other conditions, used in this disease)",
    "enzyme_replacement":"Enzyme Replacement Therapy (approved biologic)",
    "biologic":          "Biologic / Monoclonal Antibody",
    "gene_therapy":      "Gene Therapy (viral vector delivery, in clinical trials)",
    "rna_aso":           "RNA / Antisense Oligonucleotide (approved or in trials)",
}


def build_brief_prompt(context: dict) -> str:
    """
    Assemble the research-brief prompt for Claude.

    Args:
        context: Dict returned by retrieve_context.

    Returns:
        Fully-formed prompt string for the seven-section research brief.
    """
    disease = context["disease"]
    omim = context["omim_text"]
    genes = context["genes"]
    drugs_by_cat = context["drugs_by_category"]
    pathways = context["pathways"]
    ts = context["trials_summary"]
    summary = context["linkage_summary"]

    # Genes block
    gene_lines = []
    for g in genes:
        cv = g["clinvar_summary"]
        cv_str = ""
        if cv.get("variant_count", 0) > 0:
            path_str = ", ".join(
                f"{k}={v}" for k, v in list(cv["pathogenicity_counts"].items())[:3]
            )
            conseq_str = ", ".join(
                f"{k}={v}" for k, v in list(cv["molecular_consequence_counts"].items())[:3]
            )
            cv_str = f" | ClinVar: {cv['variant_count']} variants [{path_str}] ({conseq_str})"
        gene_lines.append(
            f"  • {g['symbol']} (locus={g['locus'] or 'unknown'}, "
            f"inheritance={g['inheritance'] or 'unknown'}, "
            f"assoc={g['association_type']}, source={g['source']}){cv_str}"
        )
    genes_block = "\n".join(gene_lines) if gene_lines else "  (none found)"

    # Drugs block
    drug_lines = []
    for cat, drugs in sorted(drugs_by_cat.items()):
        label = _CATEGORY_LABELS.get(cat, cat)
        drug_lines.append(f"\n  [{label}]")
        for drug in drugs:
            mechs = ", ".join(drug["mechanism"]) if drug["mechanism"] else "mechanism unspecified"
            drug_lines.append(f"    • {drug['name']} — {mechs} (source={drug['source']})")
    if not drug_lines:
        drug_lines = ["  (no drugs in graph)"]
    drugs_block = "\n".join(drug_lines)

    # Pathways block
    pathway_by_toplevel: dict[str, list] = defaultdict(list)
    for pw in pathways:
        top = pw.get("top_level_pathway") or "Other"
        pathway_by_toplevel[top].append(pw.get("name", ""))
    pathway_lines = []
    for top, names in sorted(pathway_by_toplevel.items()):
        pathway_lines.append(f"  • {top}: {', '.join(names)}")
    pathways_block = "\n".join(pathway_lines) if pathway_lines else "  (none found)"

    # Trials block
    phase_str = " | ".join(f"{k}: {v}" for k, v in sorted(ts["phase_counts"].items()))
    status_str = " | ".join(f"{k}: {v}" for k, v in sorted(ts["status_counts"].items()))
    trial_lines = []
    for t in ts["active_trials"][:10]:
        trial_lines.append(
            f"  • [{t['phase']}] {t['sponsor'] or 'unknown sponsor'} — {t['title']}"
        )
    trials_block = "\n".join(trial_lines) if trial_lines else "  (no actively recruiting trials)"

    return f"""You are a biomedical research intelligence analyst. Your task is to synthesize the structured database information below into a rigorous, cited research brief for {disease['name']}.

CRITICAL RULES:
1. Every factual claim must end with a source tag in square brackets, e.g. [OMIM], [ClinVar], [Reactome], [ChEMBL], [ClinicalTrials], [Orphanet].
2. Produce exactly seven sections using the headings below.
3. Do not invent data. If a section has no data (e.g., no disease-modifying drugs exist), state that explicitly — it is a clinically important finding.
4. For Section 6 (Drug Repurposing), reason like a human analyst: look for drugs approved in other diseases whose mechanism-of-action overlaps with disrupted pathways here. A key pattern to flag when relevant: shared protein-quality-control / chaperone machinery (e.g., CFTR correctors rescue misfolded proteins — the same chaperonin/proteasomal system processes other misfolded disease proteins, suggesting pharmacochaperone strategies may be transferable across diseases).
5. Keep each section focused and evidence-based. Use precise biological language.

--- DATABASE CONTEXT FOR {disease['name'].upper()} ---

IDENTIFIERS:
  OMIM: {disease['omim_id']} | Orphanet: ORPHA:{disease['orphacode']}
  Title: {disease['omim_title'] or 'N/A'}

OMIM DESCRIPTION:
{omim['description'] or '(not available)'}

OMIM CLINICAL FEATURES (excerpt):
{omim['clinical_features'] or '(not available)'}

OMIM MOLECULAR GENETICS (excerpt):
{omim['molecular_genetics'] or '(not available)'}

GENES ({summary['gene_count']} total):
{genes_block}

DRUGS ({summary['drug_count']} total across {len(drugs_by_cat)} categories):
{drugs_block}

REACTOME PATHWAYS ({summary['pathway_count']} linked):
{pathways_block}

CLINICAL TRIALS ({ts['total']} total):
  Phases: {phase_str}
  Status: {status_str}
  Key sponsors: {', '.join(ts['key_sponsors']) if ts['key_sponsors'] else 'none identified'}
  Actively recruiting trials (sample):
{trials_block}

--- END OF DATABASE CONTEXT ---

Now write the research brief using exactly these seven section headings:

## 1. Disease Overview
## 2. Genetic Architecture
## 3. Pathway Biology
## 4. Current Treatment Landscape
## 5. Clinical Trial Landscape
## 6. Drug Repurposing Opportunities
## 7. Open Research Questions

For Section 4, explicitly label each drug or drug class by its category (disease-modifying, symptomatic, off-label, enzyme replacement, gene therapy, RNA/ASO). If no disease-modifying drugs exist, state this as a key clinical signal and explain what it means for patients and researchers.

For Section 5 (Clinical Trial Landscape), organize the content into exactly three subsections, in this order, using these ### subheadings:
### Late-Stage Pipeline — Phase III / registrational and other late-stage programs.
### Investigational Themes — the mechanistic strategies being explored across early- and mid-phase trials, grouped by approach rather than listed trial-by-trial.
### Critical Gaps — mechanisms, patient subpopulations, or endpoints that currently have no active trials. Keep every factual claim source-tagged.
""".strip()


# ---------------------------------------------------------------------------
# Hypothesis-generation prompt
# ---------------------------------------------------------------------------

def build_hypothesis_prompt(context: dict) -> str:
    """
    Assemble the forward-looking hypothesis prompt for Claude.

    Each therapeutic hypothesis_statement must follow the exact two-paragraph
    prose format specified below — paragraph 1 folds the unmet need, the
    intervention, and the mechanistic rationale into one continuous argument;
    paragraph 2 states the analogous clinical precedent.

    Args:
        context: Dict returned by retrieve_context.

    Returns:
        Fully-formed prompt string that instructs Claude to call the
        submit_hypotheses tool with structured forward-looking research
        hypotheses.
    """
    disease = context["disease"]
    omim = context["omim_text"]
    genes = context["genes"]
    drugs_by_cat = context["drugs_by_category"]
    pathways = context["pathways"]
    ts = context["trials_summary"]

    gene_lines = [
        f"  • {g['symbol']} — {g.get('association_type', 'unknown')} "
        f"(locus={g.get('locus') or '?'}, inheritance={g.get('inheritance') or '?'}, "
        f"source={g.get('source', '')})"
        for g in genes
    ]
    genes_block = "\n".join(gene_lines) or "  (none identified)"

    pathway_lines = [
        f"  • {pw.get('name', '')} "
        f"[{pw.get('top_level_pathway', '')}] ({pw.get('reactome_id', '')})"
        for pw in pathways
    ]
    pathways_block = "\n".join(pathway_lines) or "  (none)"

    drug_lines = []
    for cat, drugs in sorted(drugs_by_cat.items()):
        for d in drugs:
            drug_lines.append(f"  • {d['name']} [{cat}]")
    drugs_block = "\n".join(drug_lines) or "  (none in graph)"

    phase_str = " | ".join(f"{k}: {v}" for k, v in sorted(ts["phase_counts"].items()))

    return f"""You are a biomedical research strategist generating forward-looking, novel research hypotheses for {disease['name']}.

Your sole task: identify the most promising UNTESTED opportunities — approaches NOT already in clinical trials for this disease and NOT already in the existing drug landscape below.

=== BIOLOGICAL KNOWLEDGE BASE: {disease['name'].upper()} ===

DISEASE IDs: OMIM {disease['omim_id']} | Orphanet ORPHA:{disease['orphacode']}

MOLECULAR GENETICS (OMIM excerpt):
{(omim.get('molecular_genetics') or '(not available)')[:1800]}

CAUSAL AND MODIFIER GENES (cite these in your rationale):
{genes_block}

DISRUPTED BIOLOGICAL PATHWAYS (cite pathway name AND Reactome ID):
{pathways_block}

EXISTING DRUG LANDSCAPE — your hypotheses must go BEYOND all of these:
{drugs_block}

CLINICAL TRIAL LANDSCAPE:
  Total registered trials: {ts['total']}
  Phase breakdown: {phase_str}
  Actively recruiting: {ts['status_counts'].get('RECRUITING', 0)}

=== END OF KNOWLEDGE BASE ===

=== FORMATTING RULES — READ CAREFULLY ===

RULE 1 — hypothesis name:
  A short 2–4 word display label. Examples: "LOXL2/LOXL3 inhibitor", "Allele-specific ASO",
  "IGF-1R kinase inhibitor", "Integrin αVβ3 antagonist".

RULE 2 — hypothesis_statement: TWO natural paragraphs of flowing prose separated by a single
  blank line — no bullets, no sub-headings, no "Mechanistic rationale" label anywhere, and no
  parenthetical IDs in the prose (place all Reactome IDs in the citations field instead).
    Paragraph 1 (UNMET NEED -> INTERVENTION -> MECHANISM, as one continuous argument):
      Open with the specific unmet clinical need in this disease, then lead directly into the
      therapeutic intervention — name the compound or class first, state what it does, and name
      the expected therapeutic effect — and then continue, in the SAME paragraph and at the SAME
      level of mechanistic detail, into why it works: name the specific disrupted pathway,
      identify the gene or protein whose dysfunction drives it, and name the downstream
      consequence that creates the therapeutic opening. The mechanistic rationale must read as a
      seamless continuation of the same argument, never as a separate labelled section.
    Paragraph 2 (ANALOGY): Begin "Analogous success in [Disease name]:" then state what was
      achieved clinically and how far the programme progressed.

  EXAMPLE of correct format (do not copy — generate {disease['name']}-specific content):
  "Marfan syndrome patients have no medical therapy that halts progressive aortic root
  dilation, leaving prophylactic surgery as the only option; LOXL2/LOXL3 inhibitors such as the
  PAT-1251 class stabilise aortic elastic fibres by restoring ECM crosslink homeostasis in
  FBN1-deficient tissue and are expected to slow that dilation, because the elastic fibre
  formation programme is driven into compensatory overdrive by hyperactivated LOXL2, which
  intensifies crosslink remodelling in the aortic wall and releases sequestered TGF-β ligands
  that sustain downstream SMAD2/3 phosphorylation through TGFBR1 and TGFBR2.

  Analogous success in idiopathic pulmonary fibrosis: LOXL2 inhibition with simtuzumab
  demonstrated clear target engagement and normalised collagen crosslink profiles in a Phase II
  IPF trial, validating LOXL2 as a druggable ECM-remodelling enzyme in chronic fibrotic disease."

RULE 3 — scoring:
  feasibility_score (1–10): does a testable compound exist with human safety data?
    8–10 = FDA-approved or late-phase; 5–7 = Phase I/II data; 1–4 = de-novo required.
  novelty_score (1–10): how unexplored is this for THIS disease specifically?
  commercial_potential_score (1–10): orphan pricing precedent, IP position, payer access.
    8–10 = clear orphan economics + novel IP; 1–4 = generic in crowded market.
  Final ranking is recomputed in Python as feasibility×0.40 + novelty×0.35 + commercial×0.25.
  Feasibility carries the highest weight because a hypothesis that cannot reach patients
  has no real-world value regardless of its scientific novelty.

RULE 3b — citations field: for each therapeutic hypothesis, populate the citations array
  with every Reactome pathway ID (R-HSA-XXXXXXX format) you referenced when writing
  hypothesis_statement. This is where all pathway IDs live.
  Do not put R-HSA IDs anywhere else.

RULE 4 — other sections:
  repurposing_candidates: currently FDA-approved drugs only; name the specific pathway
    or gene-product overlap.
  experimental_priorities: name the exact model system (cell line, mouse strain, cohort),
    intervention, and primary assay with readout.
  analogous_disease (standalone field): real disease where a Phase II or later result
    confirms the analogous mechanism — not just preclinical evidence.

Use the submit_hypotheses tool to return your complete structured output now."""
