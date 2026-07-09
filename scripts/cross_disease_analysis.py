"""
scripts/cross_disease_analysis.py

Cross-disease pattern mining over the cached hypothesis JSONs.

Reads every file in data/processed/hypotheses/ (one per disease, produced by
synthesis.engine.generate_hypotheses) and runs three deterministic analyses,
then a single Claude "reasoning" pass that judges which of the raw overlaps
represent GENUINE shared biology versus coincidental pathway-label matches.

Three analyses
--------------
1. Drug overlap        — drugs that appear as repurposing candidates in more
                         than one disease, with each disease's mechanistic
                         rationale and the Reactome pathway IDs / top-level
                         categories the diseases share.
2. Pathway overlap     — Reactome pathway IDs that appear in hypothesis
                         citations across more than one disease, ranked by the
                         number of diseases citing them.
3. Mechanism themes    — diseases grouped by the top-level Reactome category of
                         their cited pathways, ranked by how many diseases fall
                         under each category (a proxy for therapeutic-mechanism
                         classes that recur across the corpus).

Reasoning pass
--------------
The raw overlaps above are label matches — two diseases can share a broad
category name ("Metabolism of proteins") for completely unrelated reasons. A
single Claude call evaluates, for each overlapping drug, whether it engages the
SAME molecular target across diseases or merely shares a broad category, and
tags each overlap genuine/coincidental with a high/medium/low confidence based
on target specificity. The same pass judges each shared top-level category.

Everything — deterministic analyses plus the reasoning verdicts — is written to
data/processed/cross_disease_patterns.json. That file is the cache: on a normal
run it is loaded and summarised without any API call. Pass --refresh to
recompute and re-call Claude.

Usage:
    python scripts/cross_disease_analysis.py            # use cache if present
    python scripts/cross_disease_analysis.py --refresh  # recompute + re-call API
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# Make the project root importable when run as a script (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

# Reuse the model id, pricing, key and paths already defined for the synthesis
# layer so this script never drifts from the rest of the project.
from synthesis.engine import (
    ANTHROPIC_API_KEY,
    COST_PER_INPUT_TOKEN,
    COST_PER_OUTPUT_TOKEN,
    HYPOTHESES_DIR,
    MODEL,
    PROCESSED_DIR,
)

REACTOME_DIR = PROCESSED_DIR / "reactome"
OUTPUT_PATH = PROCESSED_DIR / "cross_disease_patterns.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cross_disease_analysis")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_hypotheses() -> dict[str, dict]:
    """
    Load every hypothesis JSON in data/processed/hypotheses/.

    Returns:
        A dict mapping disease_name -> parsed hypothesis payload. Files that
        fail to parse are logged and skipped rather than crashing the run.
    """
    hyps: dict[str, dict] = {}
    if not HYPOTHESES_DIR.exists():
        logger.error("Hypotheses directory not found: %s", HYPOTHESES_DIR)
        return hyps

    for path in sorted(HYPOTHESES_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping unreadable hypothesis file %s: %s", path, e)
            continue
        name = data.get("disease_name") or path.stem.replace("_", " ")
        hyps[name] = data
        logger.info("Loaded hypotheses for %r", name)

    return hyps


def load_pathway_index(pathway_ids: set[str]) -> dict[str, dict]:
    """
    Look up the name and top-level category for a set of Reactome pathway IDs.

    Reads data/processed/reactome/{id}.json for each id. IDs with no backing
    file are recorded with name/top_level set to None so downstream code can
    still reference them without a KeyError.

    Args:
        pathway_ids: Reactome IDs (R-HSA-XXXXXXX) to resolve.

    Returns:
        Dict mapping pathway_id -> {"name": str|None, "top_level": str|None}.
    """
    index: dict[str, dict] = {}
    for pid in sorted(pathway_ids):
        path = REACTOME_DIR / f"{pid}.json"
        if not path.exists():
            logger.warning("No Reactome file for cited pathway %s", pid)
            index[pid] = {"name": None, "top_level": None}
            continue
        try:
            with open(path) as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read Reactome file %s: %s", path, e)
            index[pid] = {"name": None, "top_level": None}
            continue
        top = r.get("top_level_pathway") or {}
        index[pid] = {
            "name": r.get("pathway_name"),
            "top_level": top.get("name"),
        }
    return index


def _cited_pathway_ids(disease_payload: dict) -> set[str]:
    """
    Collect every Reactome pathway ID cited by a disease's therapeutic
    hypotheses (the `citations` field on each hypothesis).

    Args:
        disease_payload: One parsed hypothesis JSON.

    Returns:
        Set of Reactome IDs cited anywhere in that disease's hypotheses.
    """
    ids: set[str] = set()
    for hyp in disease_payload.get("therapeutic_hypotheses", []):
        for cit in hyp.get("citations", []):
            if isinstance(cit, str) and cit.startswith("R-HSA-"):
                ids.add(cit)
    return ids


def _normalize_drug(name: str) -> str:
    """
    Normalise a drug name into a match key so the same drug spelled slightly
    differently across diseases collapses to one entry.

    Lowercases, trims, and strips any trailing parenthetical (so
    "Rapamycin (sirolimus)" -> "rapamycin"). The original display name is kept
    separately by the caller.

    Args:
        name: Raw drug_name from a repurposing candidate.

    Returns:
        A normalised lookup key.
    """
    key = name.strip().lower()
    key = re.sub(r"\s*\(.*?\)\s*$", "", key)  # drop a trailing "(...)" clause
    return key.strip()


# ---------------------------------------------------------------------------
# Analysis 1 — drug overlap
# ---------------------------------------------------------------------------

def analyze_drug_overlaps(
    hyps: dict[str, dict], pathway_index: dict[str, dict]
) -> list[dict]:
    """
    Find drugs that appear as repurposing candidates in more than one disease.

    For each overlapping drug, report which diseases it appears in, the
    mechanistic rationale given in each disease, and the Reactome pathway IDs /
    top-level categories those diseases have in common (the intersection of
    their cited pathway sets). Repurposing candidates carry no citations of
    their own, so "shared pathways" is inherited from the diseases the drug
    bridges — it is a proxy for where the shared biology could live, and is
    exactly what the reasoning pass later stress-tests.

    Args:
        hyps: disease_name -> hypothesis payload.
        pathway_index: Reactome ID -> {name, top_level}.

    Returns:
        List of overlap dicts, sorted by number of diseases (descending).
    """
    # drug key -> {"display": str, "diseases": {disease: rationale/indication}}
    by_drug: dict[str, dict] = defaultdict(
        lambda: {"display": None, "per_disease": {}}
    )

    for disease, payload in hyps.items():
        for cand in payload.get("repurposing_candidates", []):
            raw = cand.get("drug_name") or ""
            if not raw.strip():
                continue
            key = _normalize_drug(raw)
            entry = by_drug[key]
            entry["display"] = entry["display"] or raw.strip()
            entry["per_disease"][disease] = {
                "approved_indication": cand.get("approved_indication", ""),
                "mechanistic_rationale": cand.get("mechanistic_overlap", ""),
            }

    # Precompute each disease's cited pathway IDs + categories once.
    disease_pathways = {d: _cited_pathway_ids(p) for d, p in hyps.items()}
    disease_categories = {
        d: {pathway_index.get(pid, {}).get("top_level") for pid in ids}
        - {None}
        for d, ids in disease_pathways.items()
    }

    overlaps: list[dict] = []
    for key, entry in by_drug.items():
        diseases = sorted(entry["per_disease"])
        if len(diseases) < 2:
            continue

        shared_ids = set.intersection(*(disease_pathways[d] for d in diseases))
        shared_cats = set.intersection(*(disease_categories[d] for d in diseases))

        overlaps.append(
            {
                "drug_name": entry["display"],
                "diseases": diseases,
                "disease_count": len(diseases),
                "per_disease_rationale": entry["per_disease"],
                "shared_reactome_pathway_ids": sorted(shared_ids),
                "shared_pathway_categories": sorted(shared_cats),
            }
        )

    overlaps.sort(key=lambda o: (-o["disease_count"], o["drug_name"].lower()))
    return overlaps


# ---------------------------------------------------------------------------
# Analysis 2 — pathway overlap
# ---------------------------------------------------------------------------

def analyze_pathway_overlaps(
    hyps: dict[str, dict], pathway_index: dict[str, dict]
) -> list[dict]:
    """
    Find Reactome pathway IDs cited by more than one disease, ranked by the
    number of diseases citing them.

    This is a strict exact-ID match. In the current corpus the five diseases
    cite completely disjoint pathway sets, so this list is expected to be empty
    — a meaningful negative result on its own (the diseases converge only at the
    coarser top-level-category level captured by analyze_mechanism_themes).

    Args:
        hyps: disease_name -> hypothesis payload.
        pathway_index: Reactome ID -> {name, top_level}.

    Returns:
        List of {pathway_id, pathway_name, top_level_category, diseases,
        disease_count}, sorted by disease_count descending. Empty if no ID is
        shared.
    """
    id_to_diseases: dict[str, set[str]] = defaultdict(set)
    for disease, payload in hyps.items():
        for pid in _cited_pathway_ids(payload):
            id_to_diseases[pid].add(disease)

    shared = []
    for pid, diseases in id_to_diseases.items():
        if len(diseases) < 2:
            continue
        meta = pathway_index.get(pid, {})
        shared.append(
            {
                "pathway_id": pid,
                "pathway_name": meta.get("name"),
                "top_level_category": meta.get("top_level"),
                "diseases": sorted(diseases),
                "disease_count": len(diseases),
            }
        )

    shared.sort(key=lambda p: (-p["disease_count"], p["pathway_id"]))
    return shared


# ---------------------------------------------------------------------------
# Analysis 3 — mechanism theme clustering
# ---------------------------------------------------------------------------

def analyze_mechanism_themes(
    hyps: dict[str, dict], pathway_index: dict[str, dict]
) -> list[dict]:
    """
    Group diseases by the top-level Reactome category of their cited pathways.

    Each top-level category ("Extracellular matrix organization", "Metabolism
    of proteins", ...) becomes a candidate mechanism theme; the diseases that
    cite at least one pathway under it are its members. Ranking by member count
    surfaces the mechanism classes that recur across the most diseases.

    Args:
        hyps: disease_name -> hypothesis payload.
        pathway_index: Reactome ID -> {name, top_level}.

    Returns:
        List of {top_level_category, diseases, disease_count,
        pathways_by_disease}, sorted by disease_count then pathway count,
        descending.
    """
    # category -> disease -> [(pathway_id, pathway_name), ...]
    cat_map: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for disease, payload in hyps.items():
        for pid in _cited_pathway_ids(payload):
            meta = pathway_index.get(pid, {})
            top = meta.get("top_level")
            if not top:
                continue
            cat_map[top][disease].append(
                {"pathway_id": pid, "pathway_name": meta.get("name")}
            )

    themes = []
    for category, per_disease in cat_map.items():
        total_pathways = sum(len(v) for v in per_disease.values())
        themes.append(
            {
                "top_level_category": category,
                "diseases": sorted(per_disease),
                "disease_count": len(per_disease),
                "pathway_count": total_pathways,
                "pathways_by_disease": {
                    d: per_disease[d] for d in sorted(per_disease)
                },
            }
        )

    themes.sort(
        key=lambda t: (-t["disease_count"], -t["pathway_count"], t["top_level_category"])
    )
    return themes


# ---------------------------------------------------------------------------
# Reasoning pass — genuine vs coincidental
# ---------------------------------------------------------------------------

# Forced tool-use schema: guarantees Claude returns structured, validated JSON.
_INTERPRETATION_TOOL: dict = {
    "name": "submit_interpretation",
    "description": (
        "Submit the biological interpretation of the cross-disease overlaps: "
        "which drug overlaps and shared pathway categories reflect genuine "
        "shared molecular biology versus coincidental broad-label matches."
    ),
    "input_schema": {
        "type": "object",
        "required": ["drug_overlap_assessments", "mechanism_theme_assessments"],
        "properties": {
            "drug_overlap_assessments": {
                "type": "array",
                "description": "One assessment per overlapping drug provided.",
                "items": {
                    "type": "object",
                    "required": [
                        "drug_name",
                        "diseases",
                        "hits_same_molecular_target",
                        "shared_molecular_target",
                        "verdict",
                        "confidence",
                        "reasoning",
                    ],
                    "properties": {
                        "drug_name": {"type": "string"},
                        "diseases": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "hits_same_molecular_target": {
                            "type": "boolean",
                            "description": (
                                "True ONLY if the drug engages the same "
                                "molecular target (same protein/gene product) "
                                "in every disease it appears in. False if the "
                                "diseases invoke it through different targets "
                                "or only a shared broad pathway category."
                            ),
                        },
                        "shared_molecular_target": {
                            "type": "string",
                            "description": (
                                "Name the specific shared target/protein if "
                                "hits_same_molecular_target is true (e.g. "
                                "'KEAP1/NRF2'); otherwise 'none'."
                            ),
                        },
                        "verdict": {
                            "type": "string",
                            "enum": ["genuine", "coincidental"],
                            "description": (
                                "'genuine' = the overlap reflects the same "
                                "underlying molecular biology; 'coincidental' = "
                                "the drugs/diseases only share a broad category "
                                "label with different real mechanisms."
                            ),
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": (
                                "Confidence based on target specificity: 'high' "
                                "= same well-defined molecular target across "
                                "diseases; 'medium' = same pathway/protein "
                                "family but not identical target; 'low' = only a "
                                "broad category in common."
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "2-4 sentences justifying the verdict.",
                        },
                    },
                },
            },
            "mechanism_theme_assessments": {
                "type": "array",
                "description": (
                    "One assessment per shared top-level pathway category "
                    "(only those spanning 2+ diseases)."
                ),
                "items": {
                    "type": "object",
                    "required": [
                        "top_level_category",
                        "diseases",
                        "therapeutic_mechanism_class",
                        "verdict",
                        "confidence",
                        "reasoning",
                    ],
                    "properties": {
                        "top_level_category": {"type": "string"},
                        "diseases": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "therapeutic_mechanism_class": {
                            "type": "string",
                            "description": (
                                "The druggable mechanism class this shared "
                                "category implies, if any (e.g. 'proteostasis / "
                                "pharmacological chaperone', 'anti-fibrotic "
                                "TGF-beta modulation'). Use 'none' if the shared "
                                "category is too generic to imply one."
                            ),
                        },
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "genuine_shared_biology",
                                "generic_label",
                                "coincidental",
                            ],
                            "description": (
                                "'genuine_shared_biology' = the diseases "
                                "converge on the same real sub-mechanism; "
                                "'generic_label' = a real but very broad "
                                "category covering unrelated sub-pathways; "
                                "'coincidental' = same bucket name, "
                                "mechanistically unrelated."
                            ),
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "2-4 sentences justifying the verdict.",
                        },
                    },
                },
            },
        },
    },
}


def _build_interpretation_prompt(
    drug_overlaps: list[dict], mechanism_themes: list[dict]
) -> str:
    """
    Assemble the prompt for the reasoning pass.

    Feeds Claude the deterministic drug overlaps (with each disease's
    mechanistic rationale) and the shared top-level categories (with the
    specific pathways each disease cites under them), and asks it to separate
    genuine shared molecular biology from coincidental broad-label matches.

    Args:
        drug_overlaps: Output of analyze_drug_overlaps.
        mechanism_themes: Output of analyze_mechanism_themes (all categories;
            the prompt itself focuses attention on the multi-disease ones).

    Returns:
        The prompt string.
    """
    lines: list[str] = []

    lines.append("=== DRUG OVERLAPS (same drug proposed for repurposing in 2+ diseases) ===\n")
    if drug_overlaps:
        for o in drug_overlaps:
            lines.append(f"DRUG: {o['drug_name']}  —  diseases: {', '.join(o['diseases'])}")
            lines.append(f"  shared Reactome pathway IDs: {o['shared_reactome_pathway_ids'] or '(none — cited pathway sets are disjoint)'}")
            lines.append(f"  shared top-level categories: {o['shared_pathway_categories'] or '(none)'}")
            for disease, info in o["per_disease_rationale"].items():
                lines.append(f"  [{disease}] approved for: {info['approved_indication']}")
                lines.append(f"  [{disease}] proposed mechanistic overlap: {info['mechanistic_rationale']}")
            lines.append("")
    else:
        lines.append("(no drug appears as a repurposing candidate in more than one disease)\n")

    lines.append("=== SHARED TOP-LEVEL PATHWAY CATEGORIES (2+ diseases) ===\n")
    multi = [t for t in mechanism_themes if t["disease_count"] >= 2]
    if multi:
        for t in multi:
            lines.append(f"CATEGORY: {t['top_level_category']}  —  diseases: {', '.join(t['diseases'])}")
            for disease, pws in t["pathways_by_disease"].items():
                names = "; ".join(f"{p['pathway_name']} ({p['pathway_id']})" for p in pws)
                lines.append(f"  [{disease}] cites: {names}")
            lines.append("")
    else:
        lines.append("(no top-level category spans more than one disease)\n")

    body = "\n".join(lines)

    return f"""You are a translational biology analyst evaluating whether cross-disease overlaps in a rare-disease knowledge base reflect GENUINE shared molecular biology or merely COINCIDENTAL matches on broad category labels.

Below are two kinds of overlap mined mechanically from per-disease research hypotheses.

{body}

Your task:
1. For EACH overlapping drug, decide whether it engages the SAME molecular target (same protein/gene product) in every disease it appears in, or whether the diseases only share a broad pathway category. Assign a verdict (genuine/coincidental) and a confidence (high/medium/low) driven by target specificity — high only when the same well-defined molecular target is engaged across diseases.
2. For EACH shared top-level pathway category, decide whether the diseases converge on the same real sub-mechanism (genuine_shared_biology), share a real but very broad umbrella covering unrelated sub-pathways (generic_label), or merely collide on a bucket name (coincidental). Name the implied therapeutic mechanism class where one genuinely exists.

Be skeptical: a shared category name like "Metabolism of proteins" or "Transport of small molecules" is usually too broad to be a real shared mechanism unless the specific sub-pathways line up. Reward specificity. Return your answer via the submit_interpretation tool."""


def interpret_overlaps(
    drug_overlaps: list[dict], mechanism_themes: list[dict]
) -> dict:
    """
    Run the single Claude reasoning pass over the deterministic overlaps.

    Uses forced tool-use so the response is always valid structured JSON. On any
    failure (no API key, API error, missing tool call) returns a dict with an
    "error" key and empty assessment lists so the caller can still write the
    deterministic analyses to disk.

    Args:
        drug_overlaps: Output of analyze_drug_overlaps.
        mechanism_themes: Output of analyze_mechanism_themes.

    Returns:
        Dict with keys: drug_overlap_assessments, mechanism_theme_assessments,
        model, input_tokens, output_tokens, cost_usd. Includes "error" if the
        call could not be completed.
    """
    empty = {
        "drug_overlap_assessments": [],
        "mechanism_theme_assessments": [],
        "model": MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }

    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set — skipping reasoning pass.")
        return {**empty, "error": "ANTHROPIC_API_KEY not set in .env."}

    prompt = _build_interpretation_prompt(drug_overlaps, mechanism_themes)

    logger.info("Calling Claude API (%s) for cross-disease interpretation", MODEL)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=(
                "You are a rigorous translational biology analyst. You "
                "distinguish genuine shared molecular mechanisms from "
                "coincidental shared category labels, and you never overstate "
                "an overlap that rests only on a broad pathway name. You must "
                "use the submit_interpretation tool to return your answer."
            ),
            messages=[{"role": "user", "content": prompt}],
            tools=[_INTERPRETATION_TOOL],
            tool_choice={"type": "tool", "name": "submit_interpretation"},
        )
    except anthropic.APIError as e:
        logger.error("Anthropic API call failed: %s", e)
        return {**empty, "error": f"Anthropic API call failed — {e}"}

    data: dict | None = None
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_interpretation":
            data = block.input
            break

    if data is None:
        return {**empty, "error": "Model did not return the submit_interpretation tool call."}

    usage = message.usage
    cost = (usage.input_tokens * COST_PER_INPUT_TOKEN) + (
        usage.output_tokens * COST_PER_OUTPUT_TOKEN
    )
    logger.info(
        "Interpretation complete: input=%d tokens, output=%d tokens, cost=$%.4f",
        usage.input_tokens, usage.output_tokens, cost,
    )

    return {
        "drug_overlap_assessments": data.get("drug_overlap_assessments", []),
        "mechanism_theme_assessments": data.get("mechanism_theme_assessments", []),
        "model": MODEL,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(cost, 6),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_analysis(refresh: bool = False) -> dict:
    """
    Run the full cross-disease analysis and cache it to cross_disease_patterns.json.

    On a normal run, if the cache file already exists it is loaded and returned
    unchanged (no API call). Pass refresh=True to recompute the deterministic
    analyses and re-run the Claude reasoning pass.

    Args:
        refresh: If True, bypass the cache and regenerate.

    Returns:
        The full patterns dict (also written to OUTPUT_PATH).
    """
    if OUTPUT_PATH.exists() and not refresh:
        logger.info("Returning cached patterns from %s (use --refresh to rebuild)", OUTPUT_PATH)
        with open(OUTPUT_PATH) as f:
            return json.load(f)

    hyps = load_hypotheses()
    if not hyps:
        logger.error("No hypothesis files found — nothing to analyse.")
        return {"error": "No hypothesis files found in data/processed/hypotheses/."}

    # Resolve every cited pathway ID to name + top-level category once.
    all_ids: set[str] = set()
    for payload in hyps.values():
        all_ids |= _cited_pathway_ids(payload)
    pathway_index = load_pathway_index(all_ids)

    logger.info("Analysing %d diseases across %d cited pathways", len(hyps), len(all_ids))

    drug_overlaps = analyze_drug_overlaps(hyps, pathway_index)
    pathway_overlaps = analyze_pathway_overlaps(hyps, pathway_index)
    mechanism_themes = analyze_mechanism_themes(hyps, pathway_index)

    interpretation = interpret_overlaps(drug_overlaps, mechanism_themes)

    # Build the "genuine only" distilled view the task asks for: keep only the
    # overlaps the reasoning pass tagged as real shared biology, carrying their
    # confidence rating forward.
    drug_verdicts = {
        a.get("drug_name", "").strip().lower(): a
        for a in interpretation["drug_overlap_assessments"]
    }
    genuine_drug_overlaps = []
    for o in drug_overlaps:
        v = drug_verdicts.get(o["drug_name"].strip().lower())
        if v and v.get("verdict") == "genuine":
            genuine_drug_overlaps.append(
                {
                    "drug_name": o["drug_name"],
                    "diseases": o["diseases"],
                    "shared_molecular_target": v.get("shared_molecular_target"),
                    "confidence": v.get("confidence"),
                    "reasoning": v.get("reasoning"),
                    "shared_reactome_pathway_ids": o["shared_reactome_pathway_ids"],
                }
            )

    theme_verdicts = {
        a.get("top_level_category"): a
        for a in interpretation["mechanism_theme_assessments"]
    }
    genuine_mechanism_themes = []
    for t in mechanism_themes:
        if t["disease_count"] < 2:
            continue
        v = theme_verdicts.get(t["top_level_category"])
        if v and v.get("verdict") == "genuine_shared_biology":
            genuine_mechanism_themes.append(
                {
                    "top_level_category": t["top_level_category"],
                    "diseases": t["diseases"],
                    "therapeutic_mechanism_class": v.get("therapeutic_mechanism_class"),
                    "confidence": v.get("confidence"),
                    "reasoning": v.get("reasoning"),
                }
            )

    result = {
        "generated_by": "scripts/cross_disease_analysis.py",
        "diseases_analyzed": sorted(hyps),
        "reasoning_model": interpretation.get("model"),
        "reasoning_cost_usd": interpretation.get("cost_usd", 0.0),
        "reasoning_error": interpretation.get("error"),
        # Raw mechanical overlaps
        "drug_overlaps": drug_overlaps,
        "pathway_overlaps": pathway_overlaps,
        "mechanism_themes": mechanism_themes,
        # Claude's per-item verdicts
        "drug_overlap_assessments": interpretation["drug_overlap_assessments"],
        "mechanism_theme_assessments": interpretation["mechanism_theme_assessments"],
        # Distilled: only the overlaps judged to be genuine shared biology
        "genuine_drug_overlaps": genuine_drug_overlaps,
        "genuine_mechanism_themes": genuine_mechanism_themes,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Cross-disease patterns saved to %s", OUTPUT_PATH)

    return result


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    """
    Print a clean human-readable summary: top 5 drug overlaps, top 5 shared
    pathways, top 3 mechanism themes — annotated with the reasoning verdicts.

    Args:
        result: The patterns dict returned by run_analysis.
    """
    if "error" in result and not result.get("drug_overlaps"):
        print(f"\nERROR: {result['error']}\n")
        return

    drug_verdicts = {
        a.get("drug_name", "").strip().lower(): a
        for a in result.get("drug_overlap_assessments", [])
    }
    theme_verdicts = {
        a.get("top_level_category"): a
        for a in result.get("mechanism_theme_assessments", [])
    }

    bar = "=" * 78
    print(f"\n{bar}")
    print("  CROSS-DISEASE PATTERN SUMMARY")
    print(f"  diseases analysed: {', '.join(result.get('diseases_analyzed', []))}")
    if result.get("reasoning_error"):
        print(f"  NOTE: reasoning pass unavailable ({result['reasoning_error']})")
    print(bar)

    # --- Top 5 drug overlaps ---------------------------------------------
    print("\nTOP 5 DRUG OVERLAPS (same drug repurposed across diseases)")
    print("-" * 78)
    drug_overlaps = result.get("drug_overlaps", [])
    if not drug_overlaps:
        print("  (no drug appears as a repurposing candidate in more than one disease)")
    for o in drug_overlaps[:5]:
        v = drug_verdicts.get(o["drug_name"].strip().lower(), {})
        verdict = v.get("verdict", "unjudged")
        conf = v.get("confidence", "?")
        target = v.get("shared_molecular_target", "?")
        print(f"  • {o['drug_name']}  [{', '.join(o['diseases'])}]")
        print(f"      verdict: {verdict}  |  confidence: {conf}  |  shared target: {target}")
        if v.get("reasoning"):
            print(f"      {v['reasoning']}")

    # --- Top 5 shared pathways -------------------------------------------
    print("\nTOP 5 SHARED REACTOME PATHWAYS (exact ID, cited by 2+ diseases)")
    print("-" * 78)
    pathway_overlaps = result.get("pathway_overlaps", [])
    if not pathway_overlaps:
        print("  (none — the diseases cite completely disjoint pathway IDs;")
        print("   convergence appears only at the top-level category level below)")
    for p in pathway_overlaps[:5]:
        print(f"  • {p['pathway_id']} {p['pathway_name']} — {p['disease_count']} diseases: {', '.join(p['diseases'])}")

    # --- Top 3 mechanism themes ------------------------------------------
    print("\nTOP 3 MECHANISM THEMES (top-level pathway category across diseases)")
    print("-" * 78)
    themes = [t for t in result.get("mechanism_themes", []) if t["disease_count"] >= 2]
    if not themes:
        print("  (no top-level category spans more than one disease)")
    for t in themes[:3]:
        v = theme_verdicts.get(t["top_level_category"], {})
        verdict = v.get("verdict", "unjudged")
        conf = v.get("confidence", "?")
        cls = v.get("therapeutic_mechanism_class", "?")
        print(f"  • {t['top_level_category']}  ({t['disease_count']} diseases: {', '.join(t['diseases'])})")
        print(f"      verdict: {verdict}  |  confidence: {conf}  |  mechanism class: {cls}")
        if v.get("reasoning"):
            print(f"      {v['reasoning']}")

    # --- Distilled genuine-only headline ---------------------------------
    print(f"\n{bar}")
    print("  GENUINE MECHANISTIC OVERLAPS (after reasoning filter)")
    print(bar)
    gdo = result.get("genuine_drug_overlaps", [])
    gmt = result.get("genuine_mechanism_themes", [])
    if not gdo and not gmt:
        print("  none passed the genuine-shared-biology filter")
    for o in gdo:
        print(f"  • DRUG  {o['drug_name']} [{', '.join(o['diseases'])}] "
              f"→ target {o['shared_molecular_target']} (confidence {o['confidence']})")
    for t in gmt:
        print(f"  • THEME {t['top_level_category']} [{', '.join(t['diseases'])}] "
              f"→ {t['therapeutic_mechanism_class']} (confidence {t['confidence']})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args, run (or load) the analysis, and print the summary."""
    parser = argparse.ArgumentParser(
        description="Mine cross-disease patterns from cached hypothesis JSONs."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute analyses and re-call the Claude reasoning pass "
        "instead of loading the cached cross_disease_patterns.json.",
    )
    args = parser.parse_args()

    result = run_analysis(refresh=args.refresh)
    print_summary(result)


if __name__ == "__main__":
    main()
