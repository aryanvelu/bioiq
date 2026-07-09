"""
graph/entity_resolution.py

Cross-source entity resolution for BioIQ.

Links disease records across OMIM, ClinVar, Orphanet, ChEMBL, Reactome,
and ClinicalTrials into a single unified disease node per disease, ready
for the knowledge graph builder.

Auto-registration
-----------------
There is no hardcoded disease list. The disease registry is built dynamically
from whatever has already been ingested to disk: build_registry() scans
data/processed/omim_*.json and creates one entry per OMIM node — canonical name
(cleaned from preferred_title), MIM number, and gene symbols — then attaches an
Orphanet ORPHAcode by matching each Orphanet file's preferred_name to a disease.
Consequently, ANY disease that has been ingested flows through entity resolution
and into the graph with no code change: ingest → resolve → build is seamless.

Known issues handled explicitly:
  - GBA→GBA1 gene symbol drift (HGNC renamed GBA to GBA1 in 2023), via the
    retained KNOWN_GENE_ALIASES normalization dict.
  - OMIM preferred_title is uppercase and sometimes inverted
    ("MUSCULAR DYSTROPHY, DUCHENNE TYPE"); _friendly_name() reorders/cleans it
    back to the colloquial form ("Duchenne Muscular Dystrophy").
  - Name-keyed files (ClinicalTrials raw searches, ChEMBL indication files) may
    be stored under a slightly different wording than the OMIM-derived canonical
    name, so those linkages use tolerant token-set matching rather than exact
    filename equality.
"""

import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
UNIFIED_DIR = Path("data/processed/unified")
RAW_DIR = Path("data/raw")

UNIFIED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Gene symbol normalization — the ONLY curated table that remains.
# Maps historical/aliased symbols to their current HGNC symbol.
# ---------------------------------------------------------------------------
KNOWN_GENE_ALIASES: dict[str, str] = {
    "GBA": "GBA1",   # HGNC renamed GBA → GBA1 in 2023 (pseudogene conflict)
}

# upper(symbol) → canonical symbol
_GENE_SYMBOL_NORM: dict[str, str] = {
    old.upper(): new for old, new in KNOWN_GENE_ALIASES.items()
}


def _norm_gene(symbol: str) -> str:
    """Return the canonical gene symbol, resolving known drift (GBA→GBA1)."""
    return _GENE_SYMBOL_NORM.get(symbol.upper(), symbol.upper())


# ---------------------------------------------------------------------------
# Disease-name helpers
# ---------------------------------------------------------------------------

# Words that mark a droppable subtype/modifier qualifier in an OMIM title
# (as opposed to a disease-defining descriptor like "chronic obstructive").
_GENERIC_QUALIFIERS: set[str] = {
    "type", "susceptibility", "to", "familial", "essential", "autosomal",
    "dominant", "recessive", "sporadic", "idiopathic", "hereditary", "systemic",
    "classic", "classical", "atypical", "late", "early", "onset", "late-onset",
    "early-onset", "infantile", "infantile-onset", "juvenile", "adult",
    "neonatal", "congenital", "x-linked", "linked", "demyelinating", "axonal",
    "intermediate", "severe", "mild", "group", "complementation", "and", "with",
}


def _is_generic_token(tok: str) -> bool:
    """True if a title token is a subtype marker/number rather than a real name word."""
    t = tok.lower()
    if t in _GENERIC_QUALIFIERS:
        return True
    if re.fullmatch(r"[0-9]+[a-z]?", t):   # 1, 2, 1a, 2b — numeric subtype codes
        return True
    if re.fullmatch(r"[ivxlc]+", t):       # roman numerals
        return True
    return len(t) == 1                     # stray single letters (A, B, ...)


def _friendly_name(preferred_title: str) -> str:
    """
    Turn an OMIM preferred_title into a clean, readable canonical disease name.

    OMIM titles are uppercase, carry a trailing "; SYMBOL", are often inverted
    ("MUSCULAR DYSTROPHY, DUCHENNE TYPE") and carry subtype qualifiers. This:
      - drops the "; SYMBOL" suffix,
      - splits on commas into a main term plus qualifiers, then for each qualifier:
          * an eponymous "<Name> TYPE" (Duchenne/Becker) is reordered to the front,
          * a purely generic modifier ("SUSCEPTIBILITY TO", "FAMILIAL", "TYPE I",
            "CLASSIC TYPE", "X-LINKED") is dropped,
          * a disease-defining descriptor ("CHRONIC OBSTRUCTIVE") is prepended,
      - drops parenthetical clarifications ("(CROHN DISEASE)"),
      - strips a trailing subtype number/roman numeral ("ALPORT SYNDROME 1"),
      - Title-cases the result.

    The original five disease names (Marfan/Cystic Fibrosis/Huntington/Gaucher/
    Duchenne Muscular Dystrophy) are reproduced exactly.

    Args:
        preferred_title: OMIM node's preferred_title.

    Returns:
        A cleaned, human-readable canonical name (may be "" if input is empty).
    """
    if not preferred_title:
        return ""
    head = preferred_title.split(";")[0].strip()
    segs = [s.strip() for s in head.split(",") if s.strip()]
    if not segs:
        return ""
    main, quals = segs[0], segs[1:]

    prefix = ""
    for q in quals:
        m = re.match(r"^([A-Za-z][\w-]*)\s+TYPE\b", q, re.I)
        if m and not _is_generic_token(m.group(1)):
            prefix = m.group(1) + " "                 # eponymous subtype → reorder
            break
        if all(_is_generic_token(w) for w in q.split()):
            continue                                  # generic modifier → drop
        # Disease-defining descriptor (e.g. "CHRONIC OBSTRUCTIVE") → prepend.
        prefix = re.sub(r"\s+(?:\d+|[IVXivx]+)$", "", q).strip() + " "
        break

    name = f"{prefix}{main}".strip()
    name = re.sub(r"\s*\([^)]*\)", "", name)          # drop parentheticals
    name = re.sub(r"\s+(?:\d+|[IVX]+)$", "", name)     # drop trailing subtype number/roman
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def _name_tokens(name: str) -> set[str]:
    """Lower-cased alphanumeric token set of a name, for tolerant matching."""
    return set(re.findall(r"[a-z0-9]+", (name or "").lower()))


def _search_key(name: str) -> str:
    """Derive a ClinicalTrials search key from a disease name (lowercase, underscored)."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


# ---------------------------------------------------------------------------
# Dynamic registry — built from files on disk, not hardcoded
# ---------------------------------------------------------------------------

def build_registry(processed_dir: Path = PROCESSED_DIR) -> dict:
    """
    Build the disease registry dynamically from ingested files on disk.

    One entry per data/processed/omim_*.json file. Each entry carries the
    canonical name (from preferred_title), MIM number, canonical gene symbols
    (from the OMIM genes list), a ClinicalTrials search key, and — if an
    Orphanet file's name matches — an ORPHAcode. ClinVar genes are the same set
    as the OMIM genes.

    Args:
        processed_dir: Root of data/processed/.

    Returns:
        dict canonical_name → registry entry dict.
    """
    processed_dir = Path(processed_dir)
    registry: dict[str, dict] = {}

    # --- OMIM: one disease per omim_*.json --------------------------------
    for omim_file in sorted(processed_dir.glob("omim_*.json")):
        try:
            with open(omim_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping OMIM file %s: %s", omim_file, e)
            continue

        title = data.get("preferred_title") or ""
        mim = data.get("mim_number")
        name = _friendly_name(title) or (f"OMIM {mim}" if mim else None)
        if not name:
            continue

        genes: list[str] = []
        for g in data.get("genes", []):
            sym = g.get("approved_gene_symbol")
            if sym:
                ng = _norm_gene(sym)
                if ng not in genes:
                    genes.append(ng)

        if name in registry:
            # Same cleaned name from two OMIM files — merge gene lists.
            for ng in genes:
                if ng not in registry[name]["genes"]:
                    registry[name]["genes"].append(ng)
            registry[name]["clinvar_genes"] = list(registry[name]["genes"])
            continue

        registry[name] = {
            "canonical_name": name,
            "omim_id": mim,
            "orphacode": None,
            "genes": genes,
            "clinvar_genes": list(genes),
            "clinicaltrials_search_key": _search_key(name),
            "omim_title": title,
            "orphanet_name": None,
            "aliases": {name.lower(), title.lower()},
        }

    # --- Orphanet: attach an ORPHAcode by tolerant name match -------------
    orphanet_dir = processed_dir / "orphanet"
    if orphanet_dir.exists():
        disease_tokens = {n: _name_tokens(n) for n in registry}
        for orpha_file in sorted(orphanet_dir.glob("*.json")):
            try:
                with open(orpha_file) as f:
                    od = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            oname = od.get("preferred_name") or ""
            ocode = od.get("orphacode")
            otok = _name_tokens(oname)
            if not otok or ocode is None:
                continue

            # Candidate diseases whose token set contains the Orphanet name's
            # tokens; prefer an exact set match, else the most specific (fewest
            # tokens) superset.
            best, best_size, best_exact = None, None, False
            for n, dt in disease_tokens.items():
                if not (otok <= dt):
                    continue
                exact = otok == dt
                if best is None or (exact and not best_exact) or (
                    exact == best_exact and len(dt) < best_size
                ):
                    best, best_size, best_exact = n, len(dt), exact

            if best and registry[best]["orphacode"] is None:
                registry[best]["orphacode"] = ocode
                registry[best]["orphanet_name"] = oname
                registry[best]["aliases"].add(oname.lower())

    return registry


def _canonical_disease(raw_name: str, registry: dict) -> str | None:
    """Map a raw disease name/alias to its canonical name using the registry."""
    q = (raw_name or "").strip().lower()
    if not q:
        return None
    for canonical, reg in registry.items():
        if q == canonical.lower() or q in reg.get("aliases", set()):
            return canonical
    # Fall back to token-set containment.
    qt = _name_tokens(q)
    if qt:
        for canonical in registry:
            ct = _name_tokens(canonical)
            if qt == ct or qt <= ct:
                return canonical
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_disease_map(processed_dir: Path = PROCESSED_DIR) -> dict:
    """
    Read all processed JSON files across every source and build a master
    dictionary keyed by canonical disease name.

    The set of diseases comes from build_registry() (i.e. from the OMIM files on
    disk), so newly ingested diseases are picked up automatically with no code
    change. Each entry contains:
        canonical_name      str
        omim_id             int | None
        orphacode           int | None
        genes               list[str]        canonical gene symbols
        omim_data           dict | None      full OMIM processed record
        orphanet_data       dict | None      full Orphanet processed record
        clinvar_variants    dict             {gene: [variant_dicts]}
        chembl_drug_ids     list[str]
        reactome_pathway_ids list[str]
        clinicaltrials_nct_ids list[str]
        linkage_gaps        list[str]        human-readable gap descriptions

    Args:
        processed_dir: Root of data/processed/ tree.

    Returns:
        Master disease map dict.
    """
    processed_dir = Path(processed_dir)
    registry = build_registry(processed_dir)

    # Seed from the dynamic registry
    disease_map: dict[str, dict] = {}
    for canonical, reg in registry.items():
        disease_map[canonical] = {
            "canonical_name": canonical,
            "omim_id": reg.get("omim_id"),
            "orphacode": reg.get("orphacode"),
            "genes": list(reg.get("genes", [])),
            "omim_data": None,
            "orphanet_data": None,
            "clinvar_variants": {},
            "chembl_drug_ids": [],
            "reactome_pathway_ids": [],
            "clinicaltrials_nct_ids": [],
            "linkage_gaps": [],
        }

    # --- OMIM (attach full record by MIM number) --------------------------
    mim_to_canonical = {
        reg["omim_id"]: canonical
        for canonical, reg in registry.items()
        if reg.get("omim_id")
    }
    for omim_file in sorted(processed_dir.glob("omim_*.json")):
        try:
            with open(omim_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping OMIM file %s: %s", omim_file, e)
            continue
        canonical = mim_to_canonical.get(data.get("mim_number"))
        if canonical:
            disease_map[canonical]["omim_data"] = data
            logger.info("OMIM %s → %s", data.get("mim_number"), canonical)

    # --- Orphanet (attach full record by ORPHAcode from the registry) -----
    orphanet_dir = processed_dir / "orphanet"
    code_to_canonical = {
        reg["orphacode"]: canonical
        for canonical, reg in registry.items()
        if reg.get("orphacode") is not None
    }
    if orphanet_dir.exists():
        for orpha_file in sorted(orphanet_dir.glob("*.json")):
            try:
                with open(orpha_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping Orphanet file %s: %s", orpha_file, e)
                continue
            canonical = code_to_canonical.get(data.get("orphacode"))
            if canonical:
                disease_map[canonical]["orphanet_data"] = data
                if not data.get("associated_genes"):
                    disease_map[canonical]["linkage_gaps"].append(
                        "Orphanet record has no gene associations"
                    )
                logger.info("Orphanet ORPHA:%s → %s", data.get("orphacode"), canonical)
    else:
        for canonical in disease_map:
            disease_map[canonical]["linkage_gaps"].append("Orphanet processed directory not found")

    for canonical, record in disease_map.items():
        if record["orphacode"] is None:
            record["linkage_gaps"].append("No Orphanet match found")

    # --- ClinVar (link a gene's variants to the disease owning that gene) --
    clinvar_dir = processed_dir / "clinvar"
    if clinvar_dir.exists():
        for cv_file in sorted(clinvar_dir.glob("*.json")):
            gene_raw = cv_file.stem
            gene_norm = _norm_gene(gene_raw)
            try:
                with open(cv_file) as f:
                    variants = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping ClinVar file %s: %s", cv_file, e)
                continue
            if not isinstance(variants, list):
                continue

            for canonical, reg in registry.items():
                reg_genes = [_norm_gene(g) for g in reg.get("genes", [])]
                if gene_norm in reg_genes:
                    disease_map[canonical]["clinvar_variants"][gene_raw] = variants
                    logger.info("ClinVar %s (%s) → %s", gene_raw, gene_norm, canonical)
                    break

    for canonical, record in disease_map.items():
        if not record["clinvar_variants"]:
            record["linkage_gaps"].append("No ClinVar variants linked")

    # --- ChEMBL (link a drug to every disease whose gene it targets) ------
    chembl_dir = processed_dir / "chembl"
    if chembl_dir.exists():
        for chembl_file in sorted(chembl_dir.glob("*.json")):
            try:
                with open(chembl_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping ChEMBL file %s: %s", chembl_file, e)
                continue

            # indication_{disease}.json files are LISTS handled by the builder;
            # only per-drug {chembl_id}.json dicts carry a gene target to link.
            if chembl_file.name.startswith("indication_") or not isinstance(data, dict):
                continue

            chembl_id = data.get("chembl_id")
            if not chembl_id:
                continue
            target_genes = [_norm_gene(g) for g in (data.get("target_genes") or [])]

            for canonical, reg in registry.items():
                reg_genes = [_norm_gene(g) for g in reg.get("genes", [])]
                if any(tg in reg_genes for tg in target_genes):
                    if chembl_id not in disease_map[canonical]["chembl_drug_ids"]:
                        disease_map[canonical]["chembl_drug_ids"].append(chembl_id)
                    logger.info("ChEMBL %s (%s) → %s", chembl_id, data.get("name"), canonical)

    for canonical, record in disease_map.items():
        if not record["chembl_drug_ids"]:
            record["linkage_gaps"].append(
                "No ChEMBL approved drugs linked (gene target not in ChEMBL mechanism data)"
            )

    # --- Reactome (per disease, gather pathways its genes participate in) --
    raw_reactome_dir = RAW_DIR / "reactome"
    reactome_processed_dir = processed_dir / "reactome"
    if raw_reactome_dir.exists() and reactome_processed_dir.exists():
        processed_pathway_ids = {f.stem for f in reactome_processed_dir.glob("*.json")}
        for canonical, reg in registry.items():
            for gene in reg.get("genes", []):
                search_file = raw_reactome_dir / f"search_{gene}.json"
                # Symbol drift: if the current symbol has no search file, try any
                # historical alias that maps to it (e.g. GBA for GBA1).
                if not search_file.exists():
                    for old, new in KNOWN_GENE_ALIASES.items():
                        if new == gene:
                            candidate = raw_reactome_dir / f"search_{old}.json"
                            if candidate.exists():
                                search_file = candidate
                                break
                if not search_file.exists():
                    disease_map[canonical]["linkage_gaps"].append(
                        f"No Reactome search file for gene {gene}"
                    )
                    continue
                try:
                    with open(search_file) as f:
                        pathways = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Skipping %s: %s", search_file, e)
                    continue
                if not isinstance(pathways, list):
                    continue
                for p in pathways:
                    pid = p.get("stId")
                    if pid and pid in processed_pathway_ids:
                        if pid not in disease_map[canonical]["reactome_pathway_ids"]:
                            disease_map[canonical]["reactome_pathway_ids"].append(pid)

            if not disease_map[canonical]["reactome_pathway_ids"]:
                disease_map[canonical]["linkage_gaps"].append("No Reactome pathways linked")
            else:
                logger.info(
                    "Reactome: linked %d pathways → %s",
                    len(disease_map[canonical]["reactome_pathway_ids"]), canonical,
                )

    # --- ClinicalTrials (tolerant match of canonical name → raw search file) --
    raw_ct_dir = RAW_DIR / "clinicaltrials"
    if raw_ct_dir.exists():
        ct_files = sorted(raw_ct_dir.glob("search_*.json"))
        ct_tokens = {f: _name_tokens(f.stem[len("search_"):]) for f in ct_files}
        for canonical in disease_map:
            dtok = _name_tokens(canonical)
            # Pick the search file whose tokens are contained in the disease
            # name (equal preferred), with the greatest overlap.
            best, best_score = None, -1
            for f, ftok in ct_tokens.items():
                if ftok and ftok <= dtok:
                    score = len(ftok)
                    if score > best_score:
                        best, best_score = f, score
            if best is None:
                disease_map[canonical]["linkage_gaps"].append(
                    "No ClinicalTrials search file found for this disease"
                )
                continue
            try:
                with open(best) as f:
                    studies = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping %s: %s", best, e)
                continue
            if not isinstance(studies, list):
                continue

            nct_ids = []
            for study in studies:
                ps = study.get("protocolSection", {})
                nct_id = ps.get("identificationModule", {}).get("nctId")
                if nct_id and nct_id not in nct_ids:
                    nct_ids.append(nct_id)

            disease_map[canonical]["clinicaltrials_nct_ids"] = nct_ids
            logger.info("ClinicalTrials: %d trials → %s (%s)", len(nct_ids), canonical, best.name)
            if not nct_ids:
                disease_map[canonical]["linkage_gaps"].append(
                    "No ClinicalTrials trials found for this disease"
                )
    else:
        for canonical in disease_map:
            disease_map[canonical]["linkage_gaps"].append("ClinicalTrials raw directory not found")

    return disease_map


def resolve_entity(disease_name: str, disease_map: dict | None = None) -> dict | None:
    """
    Resolve any disease name or alias to its complete unified record.

    Args:
        disease_name: Raw input name (case-insensitive; canonical name, OMIM
            title, Orphanet name, or a token-subset thereof).
        disease_map:  Pre-built map from build_disease_map(); built on demand.

    Returns:
        Unified disease record dict, or None if the name is unrecognized.
    """
    if disease_map is None:
        disease_map = build_disease_map()

    canonical = _canonical_disease(disease_name, build_registry())
    if canonical is None or canonical not in disease_map:
        logger.warning("Cannot resolve disease name: %r", disease_name)
        return None
    return disease_map[canonical]


def export_unified_nodes(
    disease_map: dict | None = None,
    output_dir: Path = UNIFIED_DIR,
) -> dict:
    """
    Save one JSON file per disease to output_dir.

    The exported record strips the full omim_data / orphanet_data blobs and
    replaces them with a compact summary so the unified node stays legible.

    Args:
        disease_map: Pre-built map from build_disease_map(); built if None.
        output_dir:  Destination directory (created if absent).

    Returns:
        dict mapping canonical name → output Path.
    """
    if disease_map is None:
        disease_map = build_disease_map()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for canonical, record in disease_map.items():
        # Build a compact unified node — omit raw blobs, keep IDs + summaries
        omim_data = record.get("omim_data") or {}
        orphanet_data = record.get("orphanet_data") or {}

        node = {
            "id": f"DISEASE:{re.sub(r'[^\w]', '_', canonical).upper()}",
            "source": "BioIQ:EntityResolution",
            "canonical_name": canonical,
            "omim_id": record.get("omim_id"),
            "omim_preferred_title": omim_data.get("preferred_title"),
            "orphacode": record.get("orphacode"),
            "orphanet_preferred_name": orphanet_data.get("preferred_name"),
            "genes": record.get("genes", []),
            "omim_genes": [
                g.get("approved_gene_symbol")
                for g in omim_data.get("genes", [])
                if g.get("approved_gene_symbol")
            ],
            "orphanet_genes": [
                g.get("symbol")
                for g in orphanet_data.get("associated_genes", [])
                if g.get("symbol")
            ],
            "clinvar_genes": sorted(record.get("clinvar_variants", {}).keys()),
            "clinvar_variant_count": sum(
                len(v) for v in record.get("clinvar_variants", {}).values()
            ),
            "chembl_drug_ids": record.get("chembl_drug_ids", []),
            "reactome_pathway_ids": record.get("reactome_pathway_ids", []),
            "clinicaltrials_nct_ids": record.get("clinicaltrials_nct_ids", []),
            "trial_count": len(record.get("clinicaltrials_nct_ids", [])),
            "linkage_gaps": record.get("linkage_gaps", []),
        }

        safe_name = re.sub(r"[^\w-]", "_", canonical)
        out_path = output_dir / f"{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(node, f, indent=2)
        saved[canonical] = out_path
        logger.info("Exported unified node → %s", out_path)

    return saved
