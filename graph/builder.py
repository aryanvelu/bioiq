"""
graph/builder.py

Assembles BioIQ's SQLite knowledge graph from unified disease nodes.

Reads data/processed/unified/*.json (produced by graph/entity_resolution.py)
and enriches each node by loading the corresponding processed source files
(OMIM text, ChEMBL drug details, Reactome pathway details, ClinicalTrials
trial details).  The result is a fully-joined SQLite database at
data/bioiq.db that the synthesis layer can query with a single function call.

Schema (MVP; migrated to Neo4j post-MVP):
  disease            — one row per disease
  genes              — one row per unique gene symbol
  drugs              — one row per unique ChEMBL drug
  pathways           — one row per unique Reactome pathway
  trials             — one row per unique clinical trial
  disease_genes      — disease ↔ gene (with source + association_type)
  disease_drugs      — disease ↔ drug
  disease_pathways   — disease ↔ pathway
  disease_trials     — disease ↔ trial
"""

import json
import logging
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path("data/bioiq.db")
UNIFIED_DIR = Path("data/processed/unified")
PROCESSED_DIR = Path("data/processed")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Schema creation
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS disease (
    id                          TEXT PRIMARY KEY,
    name                        TEXT NOT NULL,
    omim_id                     INTEGER,
    orphacode                   INTEGER,
    omim_title                  TEXT,
    omim_text_description       TEXT,
    omim_text_clinical_features TEXT,
    omim_text_molecular_genetics TEXT
);

CREATE TABLE IF NOT EXISTS genes (
    id          TEXT PRIMARY KEY,   -- gene symbol
    symbol      TEXT NOT NULL,
    locus       TEXT,
    inheritance TEXT
);

CREATE TABLE IF NOT EXISTS drugs (
    id            TEXT PRIMARY KEY,
    chembl_id     TEXT,                -- NULL for ClinicalTrials-derived entries
    name          TEXT,
    indication    TEXT,                -- JSON array as text
    mechanism     TEXT,                -- JSON array as text
    drug_category TEXT,                -- disease_modifying|symptomatic|off_label|enzyme_replacement|biologic|gene_therapy|rna_aso
    source        TEXT                 -- chembl | clinicaltrials
);

CREATE TABLE IF NOT EXISTS pathways (
    id               TEXT PRIMARY KEY,  -- reactome_id
    reactome_id      TEXT NOT NULL,
    name             TEXT,
    top_level_pathway TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    id          TEXT PRIMARY KEY,   -- nct_id
    nct_id      TEXT NOT NULL,
    title       TEXT,
    status      TEXT,
    phase       TEXT,
    sponsor     TEXT,
    enrollment  INTEGER
);

-- Relationship tables ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS disease_genes (
    disease_id       TEXT NOT NULL REFERENCES disease(id),
    gene_id          TEXT NOT NULL REFERENCES genes(id),
    source           TEXT,   -- omim | orphanet | clinvar | chembl
    association_type TEXT,   -- causal | modifying | linked
    PRIMARY KEY (disease_id, gene_id)
);

CREATE TABLE IF NOT EXISTS disease_drugs (
    disease_id TEXT NOT NULL REFERENCES disease(id),
    drug_id    TEXT NOT NULL REFERENCES drugs(id),
    PRIMARY KEY (disease_id, drug_id)
);

CREATE TABLE IF NOT EXISTS disease_pathways (
    disease_id  TEXT NOT NULL REFERENCES disease(id),
    pathway_id  TEXT NOT NULL REFERENCES pathways(id),
    PRIMARY KEY (disease_id, pathway_id)
);

CREATE TABLE IF NOT EXISTS disease_trials (
    disease_id TEXT NOT NULL REFERENCES disease(id),
    trial_id   TEXT NOT NULL REFERENCES trials(id),
    PRIMARY KEY (disease_id, trial_id)
);
"""


def create_schema(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Create the SQLite database and all tables (idempotent).

    Args:
        db_path: Path for the SQLite file.  Created if it does not exist.

    Returns:
        Open sqlite3.Connection to db_path.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    logger.info("Schema ready at %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# Helpers for loading enrichment data from processed source files
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load %s: %s", path, e)
        return None


def _find_indication_file(canonical_name: str) -> Path | None:
    """
    Find the ChEMBL indication_*.json file for a disease by tolerant name match.

    Used when the exact indication_{safe_name}.json is absent because the file
    was saved under a slightly different wording than the canonical name.
    Matches when the file's name tokens are contained in the disease's tokens,
    preferring the greatest overlap.

    Args:
        canonical_name: The disease's canonical name.

    Returns:
        Path to the best-matching indication file, or None.
    """
    tokens = set(re.findall(r"[a-z0-9]+", canonical_name.lower()))
    if not tokens:
        return None
    best, best_score = None, 0
    for f in (PROCESSED_DIR / "chembl").glob("indication_*.json"):
        ftok = set(re.findall(r"[a-z0-9]+", f.stem[len("indication_"):].lower()))
        if ftok and ftok <= tokens and len(ftok) > best_score:
            best, best_score = f, len(ftok)
    return best


def _omim_node(omim_id: int | None) -> dict:
    """Return the processed OMIM node for omim_id, or {} if absent."""
    if not omim_id:
        return {}
    return _load_json(PROCESSED_DIR / f"omim_{omim_id}.json") or {}


def _orphanet_node(orphacode: int | None) -> dict:
    """Return the processed Orphanet node for orphacode, or {} if absent."""
    if not orphacode:
        return {}
    return _load_json(PROCESSED_DIR / "orphanet" / f"{orphacode}.json") or {}


def _chembl_node(chembl_id: str) -> dict:
    return _load_json(PROCESSED_DIR / "chembl" / f"{chembl_id}.json") or {}


def _reactome_node(pathway_id: str) -> dict:
    return _load_json(PROCESSED_DIR / "reactome" / f"{pathway_id}.json") or {}


def _trial_node(nct_id: str) -> dict:
    return _load_json(PROCESSED_DIR / "clinicaltrials" / f"{nct_id}.json") or {}


# ---------------------------------------------------------------------------
# Helpers for classifying trial-derived drug entries
# ---------------------------------------------------------------------------

_RNA_ASO_KEYWORDS = [
    "antisense", "oligonucleotide", "sirna", "rnai", "rna interference",
    "exon skip", "exon-skip", "morpholino", "gapmer", "locked nucleic",
    "dyne-", "dyne ", "aoc ", "aoc-", "tominersen", "wve-",
    "eteplirsen", "golodirsen", "viltolarsen", "casimersen", "rg6042",
]

_GENE_THERAPY_KEYWORDS = [
    "gene therapy", "gene transfer", "aav", "adeno-associated", "raav",
    "lentiviral", "lenti-", "moxeparvovec", "flt201", "spk-",
    "sgt-", "rgx-", "bbm-d", "ly3884961", "ly-m001", "ins1201",
    "srd-001", "spot-", "gen6050", "ab-1001", "pf-06939926",
]

_SKIP_INTERVENTION_KEYWORDS = [
    "placebo", "sham", "saline", "imitation surgery", "surgery control",
    "methylprednisolone", "sirolimus", "prednisone", "dexamethasone",
    "prednisolone", "brain mri", "lumbar puncture", "lumbar punction",
    "blood sample", "cognitive evaluation", "observation",
    "standard of care", "standard care", "physiotherapy",
]

_ADMIN_PREFIXES = [
    "single dose intravenous of ",
    "single dose of ",
    "intra-striatal ",
    "intrathecal ",
    "intramuscular ",
    "subcutaneous ",
    "intravenous infusion of ",
]


def _strip_admin_prefix(name: str) -> str:
    lower = name.lower()
    for prefix in _ADMIN_PREFIXES:
        if lower.startswith(prefix):
            return name[len(prefix):]
    return name


def _classify_intervention(name: str, title: str, itype: str) -> str | None:
    """
    Classify a trial intervention as gene_therapy, rna_aso, or None (skip).

    For GENETIC-typed interventions, defaults to gene_therapy when no
    keyword matches but the name looks like a drug code.  For DRUG/BIOLOGICAL
    interventions, only returns rna_aso on an explicit keyword match.
    """
    cleaned = _strip_admin_prefix(name)
    text = f"{cleaned} {title}".lower()

    if any(kw in text for kw in _SKIP_INTERVENTION_KEYWORDS):
        return None

    if any(kw in text for kw in _RNA_ASO_KEYWORDS):
        return "rna_aso"

    if any(kw in text for kw in _GENE_THERAPY_KEYWORDS):
        return "gene_therapy"

    # For GENETIC trials: if the name looks like a drug code (e.g. "SPK-10001",
    # "RGX-202", "BI 3720931"), treat as gene_therapy — most GENETIC-tagged
    # trials are viral vector deliveries.
    if itype == "GENETIC" and re.match(r"^[A-Za-z]{2,}[\- ]?\d", cleaned.strip()):
        return "gene_therapy"

    return None


def _insert_trial_derived_drugs(
    conn: sqlite3.Connection, disease_id: str, nct_ids: list
) -> int:
    """
    Scan clinical trial interventions and insert gene_therapy / rna_aso entries.

    - GENETIC-typed interventions → gene_therapy (default) or rna_aso (if keyword match)
    - DRUG/BIOLOGICAL interventions matching RNA_ASO keywords → rna_aso

    Returns the number of drug entries inserted.
    """
    seen_keys: set[str] = set()
    inserted = 0

    for nct_id in nct_ids:
        trial = _trial_node(nct_id)
        if not trial:
            continue

        title = trial.get("title", "")
        names = trial.get("intervention_name") or []
        itypes = trial.get("intervention_type") or []

        for i, name in enumerate(names):
            itype = itypes[i] if i < len(itypes) else "DRUG"

            if itype not in ("GENETIC", "DRUG", "BIOLOGICAL", "OTHER"):
                continue

            # For DRUG/BIOLOGICAL, only keep if RNA_ASO keyword matches;
            # GENETIC trials always run through full classification
            if itype in ("DRUG", "BIOLOGICAL", "OTHER"):
                text = f"{_strip_admin_prefix(name)} {title}".lower()
                if not any(kw in text for kw in _RNA_ASO_KEYWORDS):
                    continue

            category = _classify_intervention(name, title, itype)
            if category is None:
                continue

            clean_name = _strip_admin_prefix(name).strip()
            norm_key = re.sub(r"[^\w]", "", clean_name.lower())
            if not norm_key or norm_key in seen_keys:
                continue
            seen_keys.add(norm_key)

            # Skip if a drug with the same name is already in the DB for this disease
            # (e.g. approved ASO from ChEMBL already inserted before this function runs)
            existing = conn.execute(
                "SELECT d.id FROM drugs d "
                "JOIN disease_drugs dd ON dd.drug_id = d.id "
                "WHERE dd.disease_id = ? AND LOWER(d.name) = LOWER(?)",
                (disease_id, clean_name),
            ).fetchone()
            if existing:
                continue

            drug_id = "CT:" + re.sub(r"[^\w]", "_", clean_name.upper()[:50])
            mech = (
                "Gene therapy (viral vector delivery)"
                if category == "gene_therapy"
                else "Antisense oligonucleotide / RNA therapeutic"
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO drugs
                    (id, chembl_id, name, indication, mechanism, drug_category, source)
                VALUES (?, NULL, ?, NULL, ?, ?, 'clinicaltrials')
                """,
                (drug_id, clean_name, json.dumps([mech]), category),
            )
            conn.execute(
                "INSERT OR IGNORE INTO disease_drugs (disease_id, drug_id) VALUES (?, ?)",
                (disease_id, drug_id),
            )
            inserted += 1

    return inserted


# ---------------------------------------------------------------------------
# 2. insert_disease
# ---------------------------------------------------------------------------

def insert_disease(conn: sqlite3.Connection, unified_node: dict) -> None:
    """
    Insert one unified disease node and all its linked entities into the graph.

    Enriches each linked ID by loading the corresponding processed source
    file so the graph tables are fully populated.  Uses INSERT OR IGNORE /
    INSERT OR REPLACE throughout so re-running is safe.

    Args:
        conn:          Open connection from create_schema().
        unified_node:  Dict loaded from data/processed/unified/*.json.
    """
    disease_id = unified_node["id"]
    omim_id    = unified_node.get("omim_id")
    orphacode  = unified_node.get("orphacode")

    # -- Load OMIM text for the disease row ----------------------------------
    omim = _omim_node(omim_id)
    omim_text = omim.get("text_sections", {})

    conn.execute(
        """
        INSERT OR REPLACE INTO disease
            (id, name, omim_id, orphacode, omim_title,
             omim_text_description, omim_text_clinical_features,
             omim_text_molecular_genetics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            disease_id,
            unified_node.get("canonical_name"),
            omim_id,
            orphacode,
            unified_node.get("omim_preferred_title"),
            omim_text.get("description"),
            omim_text.get("clinicalFeatures"),
            omim_text.get("molecularGenetics"),
        ),
    )

    # -- Genes ---------------------------------------------------------------
    # Build a gene→(locus, inheritance, source, association_type) map from
    # all three gene lists in the unified node.

    # OMIM gene metadata keyed by symbol
    omim_gene_meta: dict[str, dict] = {
        g["approved_gene_symbol"]: g
        for g in omim.get("genes", [])
        if g.get("approved_gene_symbol")
    }
    # Orphanet gene metadata keyed by symbol
    orpha = _orphanet_node(orphacode)
    orpha_gene_meta: dict[str, dict] = {
        g["symbol"]: g
        for g in orpha.get("associated_genes", [])
        if g.get("symbol")
    }

    # Collect all gene symbols with their best available metadata + source
    gene_rows: dict[str, dict] = {}

    def _add_gene(symbol: str, source: str, assoc_type: str) -> None:
        if not symbol:
            return
        if symbol not in gene_rows:
            meta = omim_gene_meta.get(symbol, {})
            gene_rows[symbol] = {
                "locus": meta.get("cyto_location"),
                "inheritance": meta.get("inheritance"),
                "source": source,
                "association_type": assoc_type,
            }

    # Primary causal genes (from OMIM phenotype map = most authoritative)
    for sym in unified_node.get("omim_genes", []):
        _add_gene(sym, "omim", "causal")

    # Orphanet associated genes (may include modifiers)
    for g in orpha.get("associated_genes", []):
        sym = g.get("symbol")
        if not sym:
            continue
        atype = "modifying" if "modif" in (g.get("association_type") or "").lower() else "causal"
        if sym not in gene_rows:
            gene_rows[sym] = {
                "locus": None,
                "inheritance": None,
                "source": "orphanet",
                "association_type": atype,
            }

    # ClinVar genes
    for sym in unified_node.get("clinvar_genes", []):
        _add_gene(sym, "clinvar", "causal")

    # Primary genes from registry (catch-all in case other lists are empty)
    for sym in unified_node.get("genes", []):
        _add_gene(sym, "registry", "causal")

    for symbol, meta in gene_rows.items():
        conn.execute(
            "INSERT OR IGNORE INTO genes (id, symbol, locus, inheritance) VALUES (?, ?, ?, ?)",
            (symbol, symbol, meta["locus"], meta["inheritance"]),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO disease_genes
                (disease_id, gene_id, source, association_type)
            VALUES (?, ?, ?, ?)
            """,
            (disease_id, symbol, meta["source"], meta["association_type"]),
        )

    # -- Drugs (gene-target / disease_modifying) ---------------------------------
    canonical_name = unified_node.get("canonical_name", "")
    for chembl_id in unified_node.get("chembl_drug_ids", []):
        drug = _chembl_node(chembl_id)
        if not drug:
            logger.warning("No ChEMBL file for %s — skipping", chembl_id)
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO drugs
                (id, chembl_id, name, indication, mechanism, drug_category, source)
            VALUES (?, ?, ?, ?, ?, ?, 'chembl')
            """,
            (
                chembl_id, chembl_id,
                drug.get("name"),
                json.dumps(drug.get("indication") or []),
                json.dumps(drug.get("mechanism") or []),
                drug.get("drug_category", "disease_modifying"),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO disease_drugs (disease_id, drug_id) VALUES (?, ?)",
            (disease_id, chembl_id),
        )

    # -- Drugs (indication-based: symptomatic | off_label | enzyme_replacement | biologic | rna_aso) --
    safe_name = re.sub(r"[^\w]", "_", canonical_name)
    ind_path = PROCESSED_DIR / "chembl" / f"indication_{safe_name}.json"
    # The indication file is keyed on the name used at ingestion time, which can
    # differ slightly from the OMIM-derived canonical name. Fall back to a
    # tolerant token-subset match so those drugs are not silently dropped.
    if not ind_path.exists():
        ind_path = _find_indication_file(canonical_name) or ind_path
    indication_drugs = _load_json(ind_path) or []
    for drug in indication_drugs:
        cid = drug.get("chembl_id")
        if not cid:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO drugs
                (id, chembl_id, name, indication, mechanism, drug_category, source)
            VALUES (?, ?, ?, ?, ?, ?, 'chembl')
            """,
            (
                cid, cid,
                drug.get("name"),
                json.dumps(drug.get("indication") or []),
                json.dumps(drug.get("mechanism") or []),
                drug.get("drug_category"),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO disease_drugs (disease_id, drug_id) VALUES (?, ?)",
            (disease_id, cid),
        )

    # -- Drugs (trial-derived: gene_therapy | rna_aso) -----------------------
    nct_ids = unified_node.get("clinicaltrials_nct_ids", [])
    n_trial_drugs = _insert_trial_derived_drugs(conn, disease_id, nct_ids)

    # -- Pathways ------------------------------------------------------------
    for pathway_id in unified_node.get("reactome_pathway_ids", []):
        pw = _reactome_node(pathway_id)
        if not pw:
            logger.warning("No Reactome file for %s — skipping", pathway_id)
            continue
        top = pw.get("top_level_pathway") or {}
        conn.execute(
            """
            INSERT OR IGNORE INTO pathways
                (id, reactome_id, name, top_level_pathway)
            VALUES (?, ?, ?, ?)
            """,
            (
                pathway_id,
                pathway_id,
                pw.get("pathway_name"),
                top.get("name"),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO disease_pathways (disease_id, pathway_id) VALUES (?, ?)",
            (disease_id, pathway_id),
        )

    # -- Trials --------------------------------------------------------------
    for nct_id in unified_node.get("clinicaltrials_nct_ids", []):
        trial = _trial_node(nct_id)
        if not trial:
            logger.warning("No ClinicalTrials file for %s — skipping", nct_id)
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO trials
                (id, nct_id, title, status, phase, sponsor, enrollment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nct_id,
                nct_id,
                trial.get("title"),
                trial.get("status"),
                trial.get("phase"),
                trial.get("sponsor"),
                trial.get("enrollment"),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO disease_trials (disease_id, trial_id) VALUES (?, ?)",
            (disease_id, nct_id),
        )

    conn.commit()
    chembl_drug_count = len(unified_node.get("chembl_drug_ids", [])) + len(indication_drugs)
    logger.info(
        "Inserted disease %r: %d genes, %d ChEMBL drugs + %d trial-derived, "
        "%d pathways, %d trials",
        canonical_name,
        len(gene_rows),
        chembl_drug_count,
        n_trial_drugs,
        len(unified_node.get("reactome_pathway_ids", [])),
        len(nct_ids),
    )


# ---------------------------------------------------------------------------
# 3. build_full_graph
# ---------------------------------------------------------------------------

def build_full_graph(
    unified_dir: Path = UNIFIED_DIR,
    db_path: Path = DB_PATH,
) -> sqlite3.Connection:
    """
    Build the complete knowledge graph from all unified disease nodes.

    Drops and recreates the database so every run produces a clean,
    reproducible graph.  Call this after (re-)running entity_resolution.

    Args:
        unified_dir: Directory containing *.json unified nodes.
        db_path:     Output SQLite path.

    Returns:
        Open sqlite3.Connection to the populated database.
    """
    unified_dir = Path(unified_dir)
    db_path = Path(db_path)

    # Start fresh each build so stale rows never accumulate
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing database at %s", db_path)

    conn = create_schema(db_path)

    node_files = sorted(unified_dir.glob("*.json"))
    if not node_files:
        logger.warning("No unified node files found in %s", unified_dir)
        return conn

    logger.info("Building graph from %d unified nodes", len(node_files))
    for path in node_files:
        node = _load_json(path)
        if node is None:
            logger.warning("Skipping unreadable file %s", path)
            continue
        insert_disease(conn, node)

    # Summary stats
    for table in ("disease", "genes", "drugs", "pathways", "trials"):
        (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        logger.info("  %s rows: %d", table, count)

    return conn


# ---------------------------------------------------------------------------
# 3b. ingest_on_demand — full pipeline for a brand-new (unseeded) disease
# ---------------------------------------------------------------------------

def _genes_from_omim_node(mim: int) -> list[str]:
    """
    Read data/processed/omim_{mim}.json and return its associated HGNC gene
    symbols (deduplicated, order-preserved), or [] if the node is missing.

    Mirrors scripts/seed_diseases._genes_for so on-demand ingestion drives the
    gene-based sources exactly the way the seed pipeline does.
    """
    node = _load_json(PROCESSED_DIR / f"omim_{mim}.json")
    if not isinstance(node, dict):
        return []
    seen: dict[str, None] = {}
    for g in node.get("genes", []):
        sym = g.get("approved_gene_symbol")
        if sym and sym not in seen:
            seen[sym] = None
    return list(seen)


def ingest_on_demand(disease_name: str) -> tuple[bool, str]:
    """
    Ingest a brand-new disease end-to-end and fold it into the knowledge graph.

    Runs the same pipeline as scripts/seed_diseases.py, but for a single disease
    that was never seeded and starting only from its name:

        1. OMIM     — search_omim(name) to resolve the MIM number, then
                      batch_ingest([mim]) to save the OMIM node.
        2. ClinVar  — batch_ingest_by_genes(genes)            (gene-driven)
        3. ChEMBL   — batch_ingest_by_genes(genes)            (gene-target drugs)
                      + batch_ingest_by_indication({name})    (symptomatic/ERT/biologics)
        4. Reactome — batch_ingest_by_genes(genes)
        5. Orphanet — batch_ingest([name])
        6. ClinicalTrials — batch_ingest_by_disease([name])
        7. Entity resolution — build_disease_map() -> export_unified_nodes()
        8. Graph build       — build_full_graph()  (rebuilds data/bioiq.db from
                               ALL unified nodes on disk, including this one)

    The individual source steps are resilient: if one source errors or returns
    nothing it is logged and skipped (many real diseases legitimately have no
    ClinVar/ChEMBL/etc. data), so a partial-but-useful node is still built. Only
    the steps that make ingestion meaningless are fatal — failing to resolve a
    MIM number, saving no OMIM node, the graph rebuild raising, or the disease
    not appearing in the rebuilt graph afterwards.

    A 0.5s pause is inserted between the major source stages (on top of each
    ingestion module's own per-request delay) to respect source-API rate limits.

    Args:
        disease_name: Free-text disease name typed by the user, e.g.
            "Hypertrophic Cardiomyopathy". Need not match any seeded name.

    Returns:
        (True, canonical_name) on success — canonical_name is the name the
        disease was registered under in the graph (derived from the OMIM
        preferred title, so it may differ from the input) and is what
        query_disease / generate_brief should be called with afterwards.
        (False, error_message) if a fatal step failed.
    """
    # Lazy imports keep builder's module-level import graph clean and avoid any
    # import-time cycle with the ingestion / entity-resolution layers.
    import time

    from graph import entity_resolution
    from ingestion import chembl, clinicaltrials, clinvar, omim, orphanet, reactome

    name = disease_name.strip()
    if not name:
        return False, "Empty disease name."

    def _pause() -> None:
        """0.5s courtesy delay between source stages (project rate policy)."""
        time.sleep(0.5)

    def _safe(source: str, fn, *args, **kwargs):
        """Run one source step; log and swallow errors so it is never fatal."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — one source must not abort ingestion
            logger.error("[ingest_on_demand] %s failed for %r: %s", source, name, e)
            return None

    logger.info("[ingest_on_demand] START %r", name)

    # -- Step 1: resolve MIM via OMIM search, then ingest the OMIM node -------
    logger.info("[ingest_on_demand] 1/8 OMIM search for %r", name)
    try:
        results = omim.search_omim(name)
    except Exception as e:  # noqa: BLE001
        return False, f"OMIM search raised for {name!r}: {e}"

    mim = next((r.get("mim_number") for r in results if r.get("mim_number")), None)
    if mim is None:
        return False, f"OMIM search found no MIM number for {name!r}."
    logger.info("[ingest_on_demand] resolved %r -> MIM %s (%s)",
                name, mim, (results[0].get("title") or "").strip())

    _pause()
    logger.info("[ingest_on_demand] 1/8 OMIM batch_ingest MIM %s", mim)
    try:
        omim.batch_ingest([mim])
    except Exception as e:  # noqa: BLE001
        return False, f"OMIM ingest raised for MIM {mim}: {e}"
    if not (PROCESSED_DIR / f"omim_{mim}.json").exists():
        return False, f"OMIM node was not saved for MIM {mim}; cannot continue."

    genes = _genes_from_omim_node(mim)
    logger.info("[ingest_on_demand] MIM %s genes: %s", mim, genes or "none")

    # -- Steps 2-4: gene-driven sources (resilient) --------------------------
    if genes:
        _pause()
        logger.info("[ingest_on_demand] 2/8 ClinVar for %d gene(s)", len(genes))
        _safe("clinvar", clinvar.batch_ingest_by_genes, genes)

        _pause()
        logger.info("[ingest_on_demand] 3/8 ChEMBL (gene targets) for %d gene(s)", len(genes))
        _safe("chembl_gene", chembl.batch_ingest_by_genes, genes)

        _pause()
        logger.info("[ingest_on_demand] 4/8 Reactome for %d gene(s)", len(genes))
        _safe("reactome", reactome.batch_ingest_by_genes, genes)
    else:
        logger.warning(
            "[ingest_on_demand] no OMIM genes for MIM %s — skipping "
            "ClinVar / gene-ChEMBL / Reactome", mim,
        )

    # -- Step 3b: indication-driven ChEMBL (symptomatic / ERT / biologics) ---
    _pause()
    logger.info("[ingest_on_demand] 3/8 ChEMBL (by indication) for %r", name)
    _safe("chembl_indication", chembl.batch_ingest_by_indication, {name: name})

    # -- Step 5: Orphanet (name-driven) --------------------------------------
    _pause()
    logger.info("[ingest_on_demand] 5/8 Orphanet for %r", name)
    _safe("orphanet", orphanet.batch_ingest, [name])

    # -- Step 6: ClinicalTrials (name-driven) --------------------------------
    _pause()
    logger.info("[ingest_on_demand] 6/8 ClinicalTrials for %r", name)
    _safe("clinicaltrials", clinicaltrials.batch_ingest_by_disease, [name])

    # -- Step 7: entity resolution (registers the new OMIM file on disk) ------
    logger.info("[ingest_on_demand] 7/8 entity resolution")
    try:
        disease_map = entity_resolution.build_disease_map()
        entity_resolution.export_unified_nodes(disease_map)
    except Exception as e:  # noqa: BLE001
        return False, f"Entity resolution failed: {e}"

    # -- Step 8: rebuild the graph from every unified node on disk -----------
    logger.info("[ingest_on_demand] 8/8 build_full_graph (rebuilding data/bioiq.db)")
    try:
        conn = build_full_graph()
    except Exception as e:  # noqa: BLE001
        return False, f"Graph rebuild failed: {e}"

    # The disease is registered under an OMIM-derived name; recover it straight
    # from the rebuilt DB by the MIM we ingested (the DB is the source of truth).
    try:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT name FROM disease WHERE omim_id = ?", (mim,)
        ).fetchone()
    finally:
        conn.close()

    if r is None:
        return False, (
            f"Ingested MIM {mim} but no disease row references it after the "
            "graph rebuild."
        )
    canonical = r["name"]

    # Final sanity check: the disease must be queryable end-to-end.
    if query_disease(canonical) is None:
        return False, (
            f"Disease {canonical!r} was ingested but is not queryable in the "
            "rebuilt knowledge graph."
        )

    logger.info("[ingest_on_demand] DONE %r -> canonical %r (MIM %s)", name, canonical, mim)
    return True, canonical


# ---------------------------------------------------------------------------
# 4. query_disease
# ---------------------------------------------------------------------------

def query_disease(disease_name: str, db_path: Path = DB_PATH) -> dict | None:
    """
    Return everything the graph knows about a disease as a single dict.

    Matches disease_name case-insensitively against the canonical name.
    The returned dict is structured for direct hand-off to the synthesis
    layer.

    Args:
        disease_name: Any casing of the canonical disease name.
        db_path:      Path to the SQLite database.

    Returns:
        Dict with keys: disease, genes, drugs, pathways, trials,
        linkage_summary.  None if no matching disease is found.
    """
    if not Path(db_path).exists():
        logger.error("Database not found at %s — run build_full_graph() first", db_path)
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Disease row
    row = conn.execute(
        "SELECT * FROM disease WHERE LOWER(name) = LOWER(?)",
        (disease_name.strip(),),
    ).fetchone()

    if row is None:
        logger.warning("No disease found matching %r", disease_name)
        conn.close()
        return None

    disease_id = row["id"]
    disease_dict = dict(row)

    # Genes
    genes = [
        dict(r)
        for r in conn.execute(
            """
            SELECT g.symbol, g.locus, g.inheritance,
                   dg.source, dg.association_type
            FROM disease_genes dg
            JOIN genes g ON g.id = dg.gene_id
            WHERE dg.disease_id = ?
            ORDER BY dg.association_type, g.symbol
            """,
            (disease_id,),
        ).fetchall()
    ]

    # Drugs
    drugs = [
        {
            **dict(r),
            "indication": json.loads(r["indication"] or "[]"),
            "mechanism": json.loads(r["mechanism"] or "[]"),
        }
        for r in conn.execute(
            """
            SELECT d.chembl_id, d.name, d.indication, d.mechanism,
                   d.drug_category, d.source
            FROM disease_drugs dd
            JOIN drugs d ON d.id = dd.drug_id
            WHERE dd.disease_id = ?
            ORDER BY d.drug_category, d.name
            """,
            (disease_id,),
        ).fetchall()
    ]

    # Pathways
    pathways = [
        dict(r)
        for r in conn.execute(
            """
            SELECT p.reactome_id, p.name, p.top_level_pathway
            FROM disease_pathways dp
            JOIN pathways p ON p.id = dp.pathway_id
            WHERE dp.disease_id = ?
            ORDER BY p.top_level_pathway, p.name
            """,
            (disease_id,),
        ).fetchall()
    ]

    # Trials
    trials = [
        dict(r)
        for r in conn.execute(
            """
            SELECT t.nct_id, t.title, t.status, t.phase,
                   t.sponsor, t.enrollment
            FROM disease_trials dt
            JOIN trials t ON t.id = dt.trial_id
            WHERE dt.disease_id = ?
            ORDER BY t.status, t.phase, t.nct_id
            """,
            (disease_id,),
        ).fetchall()
    ]

    conn.close()

    drugs_by_category: dict = {}
    for drug in drugs:
        cat = drug.get("drug_category") or "unknown"
        drugs_by_category.setdefault(cat, []).append(drug)

    return {
        "disease": disease_dict,
        "genes": genes,
        "drugs": drugs,
        "drugs_by_category": drugs_by_category,
        "pathways": pathways,
        "trials": trials,
        "linkage_summary": {
            "gene_count": len(genes),
            "drug_count": len(drugs),
            "pathway_count": len(pathways),
            "trial_count": len(trials),
            "drug_categories": {k: len(v) for k, v in drugs_by_category.items()},
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = build_full_graph()
    conn.close()

    test_diseases = [
        "Marfan Syndrome",
        "Cystic Fibrosis",
        "Huntington Disease",
        "Gaucher Disease",
        "Duchenne Muscular Dystrophy",
    ]
    for name in test_diseases:
        result = query_disease(name)
        if result:
            s = result["linkage_summary"]
            cats = s.get("drug_categories", {})
            cat_str = " | ".join(f"{k}={v}" for k, v in sorted(cats.items()))
            print(
                f"\n{name}: {s['gene_count']} genes | {s['drug_count']} drugs | "
                f"{s['pathway_count']} pathways | {s['trial_count']} trials"
            )
            if cat_str:
                print(f"  drug breakdown: {cat_str}")
            for drug in result["drugs"]:
                print(
                    f"  [{drug.get('drug_category','?')}] {drug.get('name')} "
                    f"({drug.get('source','?')})"
                )
