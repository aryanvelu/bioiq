"""
synthesis/engine.py

Core reasoning layer for BioIQ.

Takes a disease name, retrieves everything the knowledge graph knows about it,
assembles a context-rich prompt, calls the Anthropic Claude API, and returns a
structured, cited research brief.

Pipeline: retrieve_context → build_prompt → generate_brief
"""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from synthesis.prompts import build_hypothesis_prompt

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 14000

# claude-sonnet-4-6 pricing (USD per token)
COST_PER_INPUT_TOKEN = 3.00 / 1_000_000
COST_PER_OUTPUT_TOKEN = 15.00 / 1_000_000

PROCESSED_DIR = Path("data/processed")
BRIEFS_DIR = PROCESSED_DIR / "briefs"
HYPOTHESES_DIR = PROCESSED_DIR / "hypotheses"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output post-processing — clean_output
# ---------------------------------------------------------------------------
#
# Two text fixes are applied to every brief string and every hypotheses dict
# before it is returned to the API layer, on BOTH the cache-hit path and the
# fresh-generation path:
#
#   Fix 1 — strip Reactome pathway IDs (R-HSA-<digits>) that appear inline in
#           prose, while preserving them where they belong: a dedicated
#           citations line, or the hypotheses `citations` array.
#   Fix 2 — drop any sentence that mentions "Reactome pathway" / "the Reactome
#           pathway", since the pathway is cited separately anyway.
#
# NOTE: the spec asked for "three text fixes" but described only these two; a
# third can be slotted into _clean_text below if/when it is specified.

_RHSA_TOKEN = r"R-HSA-\d+"

# A whole line that is ONLY R-HSA IDs (optionally "Citations:"-prefixed) is a
# dedicated citations line and is preserved verbatim — Fix 1 never touches it.
_CITATIONS_LINE_RE = re.compile(
    r"^\s*(?:citations?\s*[:\-]\s*)?"
    r"(?:\[?\s*" + _RHSA_TOKEN + r"\s*\]?[\s,;]*)+$",
    re.IGNORECASE,
)

# R-HSA IDs wrapped in ()/[] inside prose (including comma-separated runs).
_RHSA_PAREN_RE = re.compile(
    r"\s*[\(\[]\s*" + _RHSA_TOKEN
    + r"(?:\s*[,;]\s*" + _RHSA_TOKEN + r")*\s*[\)\]]"
)

# Bare R-HSA IDs sitting in prose, with any leading separator.
_RHSA_BARE_RE = re.compile(r"\s*[,;]?\s*" + _RHSA_TOKEN)

# A full sentence mentioning "Reactome pathway" — removed whole (Fix 2).
_REACTOME_SENTENCE_RE = re.compile(
    r"[ \t]*[^.!?\n]*\bReactome pathway\b[^.!?\n]*(?:[.!?]|$)",
    re.IGNORECASE,
)

# Keys whose values are pure citation data and must never be text-cleaned.
_PRESERVE_KEYS = {"citations"}


def _tidy_spacing(text: str) -> str:
    """Collapse doubled spaces and remove spaces stranded before punctuation."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def _strip_inline_rhsa(text: str) -> str:
    """Fix 1 — remove inline R-HSA IDs, keep dedicated citation lines intact."""
    out = []
    for line in text.split("\n"):
        if _CITATIONS_LINE_RE.match(line):
            out.append(line)                      # dedicated citations line — keep
            continue
        cleaned = _RHSA_PAREN_RE.sub("", line)
        cleaned = _RHSA_BARE_RE.sub("", cleaned)
        out.append(_tidy_spacing(cleaned))
    return "\n".join(out)


def _drop_reactome_sentences(text: str) -> str:
    """Fix 2 — remove any sentence mentioning 'Reactome pathway'."""
    return _tidy_spacing(_REACTOME_SENTENCE_RE.sub("", text))


def _clean_text(text: str) -> str:
    """Apply both prose fixes to a single string; non-strings pass through."""
    if not isinstance(text, str) or not text:
        return text
    return _strip_inline_rhsa(_drop_reactome_sentences(text))


def clean_output(obj):
    """
    Apply the brief/hypothesis text fixes to any output before it is returned.

    Accepts either a brief (a markdown string) or a hypotheses payload (a dict,
    possibly nested with lists). Strings are cleaned; dicts and lists are walked
    recursively. Values under a `citations` key are preserved untouched so the
    Reactome IDs collected there survive Fix 1.

    Args:
        obj: A str, dict, or list from generate_brief / generate_hypotheses.

    Returns:
        The same shape as the input, with Fix 1 and Fix 2 applied to all prose.
    """
    if isinstance(obj, str):
        return _clean_text(obj)
    if isinstance(obj, list):
        return [clean_output(x) for x in obj]
    if isinstance(obj, dict):
        return {
            k: (v if k in _PRESERVE_KEYS else clean_output(v))
            for k, v in obj.items()
        }
    return obj


# ---------------------------------------------------------------------------
# 1. retrieve_context
# ---------------------------------------------------------------------------

def _load_clinvar_summary(gene_symbol: str) -> dict:
    """
    Load and summarize ClinVar variants for a gene.

    Returns a dict with variant_count, pathogenicity_counts (pathogenic /
    likely_pathogenic / vus / benign), and molecular_consequence_counts
    (missense / nonsense / frameshift / etc.).
    """
    path = PROCESSED_DIR / "clinvar" / f"{gene_symbol}.json"
    if not path.exists():
        return {"variant_count": 0, "pathogenicity_counts": {}, "molecular_consequence_counts": {}}

    try:
        with open(path) as f:
            variants = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"variant_count": 0, "pathogenicity_counts": {}, "molecular_consequence_counts": {}}

    if not isinstance(variants, list):
        return {"variant_count": 0, "pathogenicity_counts": {}, "molecular_consequence_counts": {}}

    path_counts: Counter = Counter()
    conseq_counts: Counter = Counter()
    for v in variants:
        sig = (v.get("clinical_significance") or "unknown").lower()
        path_counts[sig] += 1
        conseq = (v.get("molecular_consequence") or "unknown").lower()
        conseq_counts[conseq] += 1

    return {
        "variant_count": len(variants),
        "pathogenicity_counts": dict(path_counts.most_common()),
        "molecular_consequence_counts": dict(conseq_counts.most_common()),
    }


def retrieve_context(disease_name: str) -> dict | None:
    """
    Pull everything the knowledge graph knows about a disease and return it as
    a structured context dictionary ready to be passed to build_prompt.

    Args:
        disease_name: Canonical disease name, e.g. "Huntington Disease".

    Returns:
        A dict with the following keys, or None if no matching disease is found:
          disease         — core disease metadata (name, OMIM id, Orphacode, title)
          omim_text       — dict with description, clinical_features, molecular_genetics
                            (each trimmed to a useful length for the prompt)
          genes           — list of gene dicts: symbol, locus, inheritance, source,
                            association_type, clinvar_summary
          drugs_by_category — dict mapping category name → list of
                              {name, mechanism, source} dicts
          pathways        — list of {reactome_id, name, top_level_pathway} dicts
          trials_summary  — dict with phase_counts, status_counts, total,
                            active_trials (RECRUITING list with title/phase/sponsor),
                            key_sponsors (top 5 by trial count)
          linkage_summary — gene/drug/pathway/trial counts with drug_categories
    """
    # Import here to avoid circular import at module level
    from graph.builder import query_disease

    result = query_disease(disease_name)
    if result is None:
        logger.error("No graph data found for %r", disease_name)
        return None

    # -- Disease metadata --------------------------------------------------
    d = result["disease"]
    disease_meta = {
        "name": d.get("name"),
        "omim_id": d.get("omim_id"),
        "orphacode": d.get("orphacode"),
        "omim_title": d.get("omim_title"),
    }

    # -- OMIM text (trimmed to protect context budget) ---------------------
    omim_text = {
        "description": (d.get("omim_text_description") or "")[:600],
        "clinical_features": (d.get("omim_text_clinical_features") or "")[:2000],
        "molecular_genetics": (d.get("omim_text_molecular_genetics") or "")[:2500],
    }

    # -- Genes + ClinVar summaries -----------------------------------------
    genes = []
    for g in result["genes"]:
        sym = g.get("symbol", "")
        genes.append({
            "symbol": sym,
            "locus": g.get("locus"),
            "inheritance": g.get("inheritance"),
            "source": g.get("source"),
            "association_type": g.get("association_type"),
            "clinvar_summary": _load_clinvar_summary(sym),
        })

    # -- Drugs grouped by category (strip long indication lists) -----------
    drugs_by_category: dict[str, list] = defaultdict(list)
    for drug in result["drugs"]:
        category = drug.get("drug_category") or "unknown"
        # Keep mechanism but shorten indication to save tokens
        mechs = drug.get("mechanism") or []
        drugs_by_category[category].append({
            "name": drug.get("name") or "",
            "mechanism": mechs[:3],   # cap at 3 mechanisms
            "source": drug.get("source"),
        })

    # -- Pathways ----------------------------------------------------------
    pathways = result["pathways"]

    # -- Trials summary ----------------------------------------------------
    trials = result["trials"]
    phase_counts: Counter = Counter(t.get("phase") or "N/A" for t in trials)
    status_counts: Counter = Counter(t.get("status") or "unknown" for t in trials)

    # Active trials (RECRUITING) — include title, phase, sponsor
    active_trials = [
        {
            "nct_id": t["nct_id"],
            "title": (t.get("title") or "")[:100],
            "phase": t.get("phase"),
            "sponsor": t.get("sponsor"),
            "enrollment": t.get("enrollment"),
        }
        for t in trials
        if t.get("status") == "RECRUITING"
    ]

    sponsor_counts: Counter = Counter(
        t.get("sponsor") for t in trials if t.get("sponsor")
    )
    key_sponsors = [s for s, _ in sponsor_counts.most_common(5)]

    trials_summary = {
        "total": len(trials),
        "phase_counts": dict(phase_counts.most_common()),
        "status_counts": dict(status_counts.most_common()),
        "active_trials": active_trials[:15],  # cap to stay within prompt budget
        "key_sponsors": key_sponsors,
    }

    return {
        "disease": disease_meta,
        "omim_text": omim_text,
        "genes": genes,
        "drugs_by_category": dict(drugs_by_category),
        "pathways": pathways,
        "trials_summary": trials_summary,
        "linkage_summary": result["linkage_summary"],
    }


# ---------------------------------------------------------------------------
# 2. build_prompt
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


def build_prompt(context: dict) -> str:
    """
    Assemble a structured prompt for Claude from the context dictionary.

    The prompt instructs Claude to produce a seven-section research brief with
    every factual claim ending in a [SOURCE] citation tag.

    Args:
        context: Dict returned by retrieve_context.

    Returns:
        A fully-formed prompt string to pass to the Claude API.
    """
    disease = context["disease"]
    omim = context["omim_text"]
    genes = context["genes"]
    drugs_by_cat = context["drugs_by_category"]
    pathways = context["pathways"]
    ts = context["trials_summary"]
    summary = context["linkage_summary"]

    # -- Format genes section ---------------------------------------------
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

    # -- Format drugs section ---------------------------------------------
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

    # -- Format pathways section ------------------------------------------
    pathway_by_toplevel: dict[str, list] = defaultdict(list)
    for pw in pathways:
        top = pw.get("top_level_pathway") or "Other"
        pathway_by_toplevel[top].append(pw.get("name", ""))
    pathway_lines = []
    for top, names in sorted(pathway_by_toplevel.items()):
        pathway_lines.append(f"  • {top}: {', '.join(names)}")
    pathways_block = "\n".join(pathway_lines) if pathway_lines else "  (none found)"

    # -- Format trials section --------------------------------------------
    phase_str = " | ".join(f"{k}: {v}" for k, v in sorted(ts["phase_counts"].items()))
    status_str = " | ".join(f"{k}: {v}" for k, v in sorted(ts["status_counts"].items()))
    trial_lines = []
    for t in ts["active_trials"][:10]:
        trial_lines.append(
            f"  • [{t['phase']}] {t['sponsor'] or 'unknown sponsor'} — {t['title']}"
        )
    trials_block = "\n".join(trial_lines) if trial_lines else "  (no actively recruiting trials)"

    # -- Assemble the full prompt -----------------------------------------
    prompt = f"""You are a biomedical research intelligence analyst. Your task is to synthesize the structured database information below into a rigorous, cited research brief for {disease['name']}.

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
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# 3. generate_brief
# ---------------------------------------------------------------------------

def generate_brief(disease_name: str, force_refresh: bool = False) -> str:
    """
    Generate a structured, cited research brief for a disease.

    Cache key is the disease name only. If a brief was previously saved to
    data/processed/briefs/{disease_name}.json it is returned verbatim with no
    Claude API call, making repeat queries free — the same caching behaviour as
    generate_hypotheses. On a cache miss, runs the full pipeline
    (retrieve_context → build_prompt → Anthropic Claude API), saves the result
    to that path, and returns it. Logs token counts and estimated cost after
    each successful API call. Returns a clear error message string (rather than
    raising) if any step fails.

    Args:
        disease_name: Canonical disease name, e.g. "Cystic Fibrosis".
        force_refresh: If True, skip the cache and regenerate from the Claude
            API even when a cached brief exists — use this to refresh a brief
            after the underlying database data has changed.

    Returns:
        The complete research brief as a string, or an error message string
        prefixed with "ERROR:" if the pipeline failed.
    """
    # Cache read — same path used to save below. A cached brief needs no API
    # key and no API call, so repeat queries are free.
    safe_name = disease_name.replace(" ", "_").replace("/", "_")
    cache_path = BRIEFS_DIR / f"{safe_name}.json"
    if cache_path.exists() and not force_refresh:
        logger.info("Returning cached brief for %r from %s", disease_name, cache_path)
        try:
            with open(cache_path) as f:
                return clean_output(json.load(f)["brief"])
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # Corrupt or partial cache file — fall through and regenerate.
            logger.warning("Cached brief at %s unreadable (%s); regenerating.", cache_path, e)

    if not ANTHROPIC_API_KEY:
        return "ERROR: ANTHROPIC_API_KEY not set in .env — cannot call Claude API."

    # Step 1: retrieve
    logger.info("Retrieving context for %r", disease_name)
    context = retrieve_context(disease_name)
    if context is None:
        return f"ERROR: No knowledge graph data found for '{disease_name}'. Run build_full_graph() first."

    # Step 2: build prompt
    logger.info("Building prompt for %r", disease_name)
    prompt = build_prompt(context)

    # Step 3: call Claude API
    logger.info("Calling Claude API (%s) for %r", MODEL, disease_name)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a biomedical research intelligence analyst specializing in rare "
                "and chronic disease biology. You synthesize structured database "
                "information into rigorous, precisely cited research briefs. Every "
                "factual claim you make must cite its source database in [square brackets]."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        logger.error("Anthropic API call failed for %r: %s", disease_name, e)
        return f"ERROR: Anthropic API call failed — {e}"

    # Step 4: log usage and cost
    usage = message.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cost_usd = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)
    logger.info(
        "Claude API call complete for %r: input=%d tokens, output=%d tokens, "
        "estimated cost=$%.4f",
        disease_name,
        input_tokens,
        output_tokens,
        cost_usd,
    )

    brief = message.content[0].text

    # Step 5: persist brief to disk (cache_path computed at the top of the function)
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "disease_name": disease_name,
        "model": MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "brief": brief,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Brief saved to %s", cache_path)

    return clean_output(brief)


# ---------------------------------------------------------------------------
# 4. Hypothesis generation — tool schema
# ---------------------------------------------------------------------------

# Tool definition passed to Claude to force structured JSON output via tool use.
# Claude must call this tool; its input IS the hypotheses dict we return.
_HYPOTHESES_TOOL: dict = {
    "name": "submit_hypotheses",
    "description": "Submit the complete structured research hypothesis output for the disease.",
    "input_schema": {
        "type": "object",
        "required": [
            "therapeutic_hypotheses",
            "repurposing_candidates",
            "experimental_priorities",
            "regulatory_pathway",
        ],
        "properties": {
            "therapeutic_hypotheses": {
                "type": "array",
                "description": "Top 5 novel therapeutic hypotheses, ranked 1 (best) to 5.",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "required": [
                        "rank",
                        "name",
                        "hypothesis_statement",
                        "feasibility_score",
                        "novelty_score",
                        "commercial_potential_score",
                        "analogous_disease",
                        "citations",
                    ],
                    "properties": {
                        "rank": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": (
                                "Preliminary rank 1–5 based on your own assessment; "
                                "final ranking will be recomputed from a weighted composite "
                                "score (feasibility 40%, novelty 35%, commercial potential 25%)."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Short 2–4 word display title used as the hypothesis heading. "
                                "Examples: 'LOXL2/LOXL3 inhibitor', 'Allele-specific ASO', "
                                "'IGF-1R kinase inhibitor', 'Integrin αVβ3 antagonist'. "
                                "Lead with the target or compound class. No verbs, no full sentences."
                            ),
                        },
                        "hypothesis_statement": {
                            "type": "string",
                            "description": (
                                "TWO natural paragraphs of flowing prose separated by a blank "
                                "line — no bullets, no sub-headings, no 'Mechanistic rationale' "
                                "label, no parenthetical IDs (all Reactome IDs go in the "
                                "citations field).\n"
                                "Paragraph 1: open with the unmet clinical need, lead into the "
                                "therapeutic intervention (compound or class named first, what "
                                "it does, expected effect), then continue seamlessly — same "
                                "paragraph, same level of mechanistic detail — into the "
                                "rationale: the specific disrupted pathway, the gene or protein "
                                "whose dysfunction drives it, and the downstream consequence "
                                "that opens the therapeutic window. The mechanism must read as a "
                                "continuation of the same argument, not a separate labelled "
                                "section.\n"
                                "Paragraph 2: begin 'Analogous success in [Disease]:' then state "
                                "what was achieved clinically and how far the programme progressed."
                            ),
                        },
                        "feasibility_score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": (
                                "Clinical testing feasibility — weighted 40% in final ranking "
                                "because a scientifically brilliant hypothesis that cannot "
                                "realistically reach patients (no compound exists, no safety "
                                "data, insurmountable CMC barriers) has zero real-world value. "
                                "1 = de-novo drug design required, no existing safety data; "
                                "5 = investigational compound with some Phase I data; "
                                "10 = FDA-approved drug, defined PK/PD, clear IND pathway."
                            ),
                        },
                        "novelty_score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": (
                                "How unexplored this specific approach is for THIS disease "
                                "(weighted 35%). "
                                "1 = already well-studied in this disease; "
                                "10 = never considered for this disease in published literature."
                            ),
                        },
                        "commercial_potential_score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": (
                                "Commercial attractiveness (weighted 25%). Consider: patient "
                                "population size, pricing precedent in analogous rare diseases, "
                                "IP position, competitive landscape, and payer willingness. "
                                "1 = generic compound in large competitive market; "
                                "10 = novel composition-of-matter IP, clear orphan pricing "
                                "precedent ($200K+/year), no approved competition."
                            ),
                        },
                        "analogous_disease": {
                            "type": "string",
                            "description": (
                                "Name of a real disease where a mechanistically similar "
                                "approach has already succeeded clinically (reached approval "
                                "or shown clear Phase II efficacy)."
                            ),
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "All Reactome pathway IDs (R-HSA-XXXXXXX format) referenced "
                                "in this hypothesis. Move every pathway ID here — do NOT "
                                "embed IDs parenthetically inside hypothesis_statement prose."
                            ),
                        },
                    },
                },
            },
            "repurposing_candidates": {
                "type": "array",
                "description": "Top 3 approved drugs most worth repurposing, ranked 1 to 3.",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "required": [
                        "rank",
                        "drug_name",
                        "approved_indication",
                        "mechanistic_overlap",
                        "patient_subpopulation",
                    ],
                    "properties": {
                        "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                        "drug_name": {
                            "type": "string",
                            "description": "Generic name of the approved drug.",
                        },
                        "approved_indication": {
                            "type": "string",
                            "description": "What the drug is currently approved to treat.",
                        },
                        "mechanistic_overlap": {
                            "type": "string",
                            "description": (
                                "Specific pathway or gene-product overlap between the "
                                "drug's known mechanism and this disease's biology."
                            ),
                        },
                        "patient_subpopulation": {
                            "type": "string",
                            "description": (
                                "Which subset of patients should be enrolled first and why "
                                "(e.g., by genotype, disease stage, or biomarker status)."
                            ),
                        },
                    },
                },
            },
            "experimental_priorities": {
                "type": "array",
                "description": "Top 3 experiments ranked by expected information value (1 = highest).",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "required": [
                        "rank",
                        "experiment_description",
                        "rationale",
                        "expected_information_value",
                    ],
                    "properties": {
                        "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                        "experiment_description": {
                            "type": "string",
                            "description": (
                                "Specific experiment: name the model system "
                                "(cell line, animal model, patient cohort), the "
                                "intervention, and the primary readout/assay."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this experiment, and what knowledge gap it closes.",
                        },
                        "expected_information_value": {
                            "type": "string",
                            "description": (
                                "What decision or downstream hypothesis this enables "
                                "if it succeeds — and what it rules out if it fails."
                            ),
                        },
                    },
                },
            },
            "regulatory_pathway": {
                "type": "object",
                "required": ["fda_designations", "rationale", "strategic_implications"],
                "properties": {
                    "fda_designations": {
                        "type": "array",
                        "description": (
                            "List of FDA designations this disease/indication qualifies for "
                            "(e.g., Orphan Drug Designation, Fast Track, Breakthrough Therapy, "
                            "Accelerated Approval, Rare Pediatric Disease)."
                        ),
                        "items": {"type": "string"},
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Why these designations apply — cite disease prevalence, "
                            "unmet need, and current trial landscape."
                        ),
                    },
                    "strategic_implications": {
                        "type": "string",
                        "description": (
                            "What these designations mean practically for a company: "
                            "development timeline, market exclusivity period, FDA fee "
                            "waivers, rolling review eligibility, and voucher value."
                        ),
                    },
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 5. Hypothesis prompt builder — defined in synthesis/prompts.py, imported above
# ---------------------------------------------------------------------------
# build_hypothesis_prompt(context) is imported from synthesis.prompts


# ---------------------------------------------------------------------------
# 6. generate_hypotheses — main public function
# ---------------------------------------------------------------------------

def generate_hypotheses(disease_name: str, refresh: bool = False) -> dict:
    """
    Generate structured, forward-looking research hypotheses for a disease.

    Uses a separate Claude API call with forced tool-use to return validated
    structured JSON covering therapeutic hypotheses, repurposing candidates,
    experimental priorities, and the FDA regulatory pathway.

    Cache key is the disease name only — prompt changes never invalidate the
    cache. Pass refresh=True (or --refresh from CLI) to force regeneration.

    Args:
        disease_name: Canonical disease name, e.g. "Marfan Syndrome".
        refresh: If True, bypass the cache and regenerate from the Claude API.

    Returns:
        Dict with keys: disease_name, model, input_tokens, output_tokens,
        cost_usd, therapeutic_hypotheses, repurposing_candidates,
        experimental_priorities, regulatory_pathway.
        On failure, returns a dict with a single "error" key.
    """
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not set in .env — cannot call Claude API."}

    # Cache key = disease name only; never invalidated by prompt changes.
    safe_name = disease_name.replace(" ", "_").replace("/", "_")
    cache_path = HYPOTHESES_DIR / f"{safe_name}.json"
    if cache_path.exists() and not refresh:
        logger.info("Returning cached hypotheses for %r from %s", disease_name, cache_path)
        with open(cache_path) as f:
            return clean_output(json.load(f))

    # Retrieve graph context
    logger.info("Retrieving context for hypothesis generation for %r", disease_name)
    context = retrieve_context(disease_name)
    if context is None:
        return {"error": f"No knowledge graph data found for '{disease_name}'."}

    # Build prompt (function lives in synthesis/prompts.py)
    prompt = build_hypothesis_prompt(context)

    # Call Claude with tool_choice forced to our schema — guarantees valid JSON
    logger.info("Calling Claude API (hypothesis mode) for %r", disease_name)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a biomedical research strategist specialising in rare and chronic "
                "disease drug development. You identify novel, mechanistically grounded "
                "research hypotheses that go beyond existing knowledge. You must use the "
                "submit_hypotheses tool to return your output."
            ),
            messages=[{"role": "user", "content": prompt}],
            tools=[_HYPOTHESES_TOOL],
            tool_choice={"type": "tool", "name": "submit_hypotheses"},
        )
    except anthropic.APIError as e:
        logger.error("Anthropic API call failed for %r: %s", disease_name, e)
        return {"error": f"Anthropic API call failed — {e}"}

    # Truncation guard — stop_reason "max_tokens" means the output was cut off
    # mid-generation, making the tool input incomplete and unusable.
    if message.stop_reason == "max_tokens":
        logger.error(
            "Hypothesis generation truncated for %r (%d output tokens hit ceiling).",
            disease_name, message.usage.output_tokens,
        )
        return {
            "error": (
                f"Output truncated at token limit ({message.usage.output_tokens} tokens). "
                "The response was cut off before all sections were generated. "
                "Increase MAX_TOKENS in engine.py and retry."
            )
        }

    # Extract the tool-use block — tool_choice forces exactly one call
    hypotheses_data: dict | None = None
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_hypotheses":
            hypotheses_data = block.input
            break

    if hypotheses_data is None:
        return {"error": "Model did not return the submit_hypotheses tool call."}

    # Section completeness validation — catch any partial output before it poisons the cache
    _REQUIRED_SECTIONS = {
        "therapeutic_hypotheses": 5,
        "repurposing_candidates":  3,
        "experimental_priorities": 3,
    }
    for section, expected_count in _REQUIRED_SECTIONS.items():
        items = hypotheses_data.get(section)
        if not isinstance(items, list):
            return {"error": f"Validation failed: section '{section}' is missing or not a list."}
        if len(items) < expected_count:
            return {
                "error": (
                    f"Validation failed: '{section}' has {len(items)} items "
                    f"(expected {expected_count}). Output may be truncated."
                )
            }
    if not isinstance(hypotheses_data.get("regulatory_pathway"), dict):
        return {"error": "Validation failed: 'regulatory_pathway' section is missing or malformed."}

    # Log cost
    usage = message.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cost_usd = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)
    logger.info(
        "Hypothesis API call complete for %r: input=%d tokens, output=%d tokens, "
        "estimated cost=$%.4f",
        disease_name, input_tokens, output_tokens, cost_usd,
    )

    # Compute composite scores in Python (deterministic — not left to the model).
    # Weights: feasibility 40%, novelty 35%, commercial potential 25%.
    # Feasibility carries the highest weight because a hypothesis that cannot
    # reach patients has zero real-world value regardless of scientific novelty.
    for hyp in hypotheses_data["therapeutic_hypotheses"]:
        f = hyp.get("feasibility_score", 5)
        n = hyp.get("novelty_score", 5)
        c = hyp.get("commercial_potential_score", 5)
        raw = (0.40 * f) + (0.35 * n) + (0.25 * c)
        hyp["composite_score"] = round(raw, 2)
        hyp["score_breakdown"] = (
            f"feasibility({f})×0.40 + novelty({n})×0.35 + "
            f"commercial({c})×0.25 = {raw:.2f}"
        )

    # Re-rank by composite score (highest first)
    hypotheses_data["therapeutic_hypotheses"].sort(
        key=lambda h: h["composite_score"], reverse=True
    )
    for i, hyp in enumerate(hypotheses_data["therapeutic_hypotheses"], start=1):
        hyp["rank"] = i

    # Assemble result with metadata
    result = {
        "disease_name": disease_name,
        "model": MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        **hypotheses_data,
    }

    # Cache to disk — only reached if all validation passed
    HYPOTHESES_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Hypotheses saved to %s", cache_path)

    return clean_output(result)


# ---------------------------------------------------------------------------
# CLI entry point — test all five diseases
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_diseases = [
        "Marfan Syndrome",
        "Cystic Fibrosis",
        "Huntington Disease",
        "Gaucher Disease",
        "Duchenne Muscular Dystrophy",
    ]

    for disease_name in test_diseases:
        print(f"\n{'='*80}")
        print(f"  RESEARCH BRIEF: {disease_name.upper()}")
        print(f"{'='*80}\n")
        brief = generate_brief(disease_name)
        print(brief)
        print()
