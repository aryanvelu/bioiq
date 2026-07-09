"""
scripts/seed_diseases.py

Seed the BioIQ knowledge graph with the full disease corpus.

For every disease in DISEASES this script runs the complete ingestion pipeline
and then rebuilds the graph:

    1. OMIM         — batch_ingest(all MIM numbers)         [one call, up front]
    then, per disease (driven by the genes OMIM found for it):
    2. ClinVar      — batch_ingest_by_genes(disease genes)
    3. ChEMBL       — batch_ingest_by_genes(disease genes)   (gene-target drugs)
                      + batch_ingest_by_indication({name})   (symptomatic/ERT/biologics)
    4. Reactome     — batch_ingest_by_genes(disease genes)
    5. Orphanet     — batch_ingest([disease name])
    6. ClinicalTrials — batch_ingest_by_disease([disease name])
    finally, once for the whole corpus:
    7. Entity resolution — build_disease_map() → export_unified_nodes()
    8. Graph build       — build_full_graph()  (drops & recreates data/bioiq.db)

Every source call is wrapped so one failure never aborts the run. Progress is
logged per disease, and a machine-readable record of what succeeded and what
failed is written to data/processed/seed_summary.json (checkpointed after each
disease, so an interrupted run still leaves a partial summary).

IMPORTANT — this is a long, network-heavy job:
  • It hits six external APIs (OMIM needs OMIM_API_KEY, NCBI/ClinVar prefers
    NCBI_API_KEY) with the modules' built-in rate-limit delays. A full 58-disease
    run takes a long time and consumes API quota.
  • Step 8 (build_full_graph) DELETES and recreates data/bioiq.db from ALL
    processed files on disk — the existing corpus plus everything just seeded.

Usage:
    python scripts/seed_diseases.py                 # full run
    python scripts/seed_diseases.py --dry-run       # print the plan, touch nothing
    python scripts/seed_diseases.py --limit 3       # ingest only the first 3 diseases
    python scripts/seed_diseases.py --skip-graph    # ingest only; don't rebuild the DB
    python scripts/seed_diseases.py --trials-per-disease 50   # cap ClinicalTrials fetch
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Ingestion modules use relative paths like Path("data/processed"), so anchor
# the working directory to the project root no matter where we're invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion import chembl, clinicaltrials, clinvar, omim, orphanet, reactome
from graph import builder, entity_resolution

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("seed_diseases")

SUMMARY_PATH = Path("data/processed/seed_summary.json")

# ---------------------------------------------------------------------------
# The corpus — (disease name, OMIM MIM number, category)
# NOTE: this is exactly the list provided for seeding — 38 rare + 20 chronic =
# 58 diseases. (MIM 163950 is intentionally shared by Turner and Noonan as
# given.) The counts are surfaced in the summary so any gap versus a 70-disease
# target is visible rather than silently padded with invented entries.
# ---------------------------------------------------------------------------
DISEASES: list[tuple[str, int, str]] = [
    # ---- Rare diseases ----
    ("Marfan Syndrome",               154700, "rare"),
    ("Ehlers-Danlos Syndrome",        130000, "rare"),
    ("Huntington Disease",            143100, "rare"),
    ("Cystic Fibrosis",               219700, "rare"),
    ("Phenylketonuria",               261600, "rare"),
    ("Wilson Disease",                277900, "rare"),
    ("Gaucher Disease",               230800, "rare"),
    ("Fabry Disease",                 301500, "rare"),
    ("Pompe Disease",                 232300, "rare"),
    ("Niemann-Pick Disease",          257200, "rare"),
    ("Tay-Sachs Disease",             272800, "rare"),
    ("Fragile X Syndrome",            300624, "rare"),
    ("Rett Syndrome",                 312750, "rare"),
    ("Angelman Syndrome",             105830, "rare"),
    ("Prader-Willi Syndrome",         176270, "rare"),
    ("Williams Syndrome",             194050, "rare"),
    ("Turner Syndrome",               163950, "rare"),
    ("CHARGE Syndrome",               214800, "rare"),
    ("DiGeorge Syndrome",             188400, "rare"),
    ("Noonan Syndrome",               163950, "rare"),
    ("Alport Syndrome",               301050, "rare"),
    ("Tuberous Sclerosis",            191100, "rare"),
    ("Neurofibromatosis 1",           162200, "rare"),
    ("Von Hippel-Lindau",             193300, "rare"),
    ("Li-Fraumeni Syndrome",          151623, "rare"),
    ("Hereditary Hemochromatosis",    235200, "rare"),
    ("Spinal Muscular Atrophy",       253300, "rare"),
    ("Duchenne Muscular Dystrophy",   310200, "rare"),
    ("Becker Muscular Dystrophy",     300376, "rare"),
    ("Myotonic Dystrophy",            160900, "rare"),
    ("Charcot-Marie-Tooth",           118220, "rare"),
    ("Friedreich Ataxia",             229300, "rare"),
    ("Spinocerebellar Ataxia",        164400, "rare"),
    ("TTR Amyloidosis",               105210, "rare"),
    ("Alpha-1 Antitrypsin Deficiency", 613490, "rare"),
    ("Adrenoleukodystrophy",          300100, "rare"),
    ("Mucopolysaccharidosis Type 1",  607016, "rare"),
    ("Mucopolysaccharidosis Type 2",  309900, "rare"),
    # ---- Chronic diseases ----
    ("Type 2 Diabetes",               125853, "chronic"),
    ("Type 1 Diabetes",               222100, "chronic"),
    ("Hypertension",                  145500, "chronic"),
    ("Coronary Artery Disease",       608320, "chronic"),
    ("Heart Failure",                 142000, "chronic"),
    ("Atrial Fibrillation",           608583, "chronic"),
    ("Chronic Kidney Disease",        612623, "chronic"),
    ("Obesity",                       601665, "chronic"),
    ("Metabolic Syndrome",            605552, "chronic"),
    ("Non-Alcoholic Fatty Liver Disease", 613282, "chronic"),
    ("Rheumatoid Arthritis",          180300, "chronic"),
    ("Systemic Lupus Erythematosus",  152700, "chronic"),
    ("Multiple Sclerosis",            126200, "chronic"),
    ("Parkinson Disease",             168600, "chronic"),
    ("Alzheimer Disease",             104300, "chronic"),
    ("Asthma",                        600807, "chronic"),
    ("COPD",                          606963, "chronic"),
    ("Inflammatory Bowel Disease",    266600, "chronic"),
    ("Psoriasis",                     177900, "chronic"),
    ("Osteoporosis",                  166710, "chronic"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _genes_for(mim: int) -> list[str]:
    """
    Read the OMIM node saved by omim.batch_ingest and return its associated
    gene symbols (deduplicated, order-preserved).

    Args:
        mim: MIM number whose processed node to read.

    Returns:
        List of approved HGNC gene symbols, or [] if the node is missing/empty.
    """
    path = Path("data/processed") / f"omim_{mim}.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            node = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    seen: dict[str, None] = {}
    for g in node.get("genes", []):
        sym = g.get("approved_gene_symbol")
        if sym and sym not in seen:
            seen[sym] = None
    return list(seen)


def _run_step(rec: dict, source: str, fn, *args, **kwargs):
    """
    Execute one ingestion call, recording its outcome on the disease record.

    Any exception is caught and logged so a single source failure never stops
    the run. The raw return value is handed back to the caller for counting.

    Args:
        rec:    The per-disease record dict to annotate on failure.
        source: Source label, e.g. "clinvar", used in logs and error entries.
        fn:     The batch function to call.
        *args, **kwargs: Passed through to fn.

    Returns:
        (result, error): result is fn's return value or None on failure;
        error is None on success or the exception message string on failure.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001 — deliberately broad: never abort the run
        msg = f"{type(e).__name__}: {e}"
        logger.error("  %s failed: %s", source, msg)
        rec.setdefault("errors", []).append(f"{source}: {msg}")
        return None, msg


def _status(result, error, *, count: int | None = None) -> dict:
    """
    Classify a step outcome into a compact status dict for the summary.

    status is "fail" (raised), "ok" (ran and produced data), or "empty" (ran
    but found nothing). An optional count is attached when meaningful.
    """
    if error is not None:
        return {"status": "fail", "error": error}
    if count is None:
        # No count semantics — treat a truthy result as ok.
        return {"status": "ok" if result else "empty"}
    return {"status": "ok" if count > 0 else "empty", "count": count}


# ---------------------------------------------------------------------------
# Per-disease ingestion
# ---------------------------------------------------------------------------

def ingest_disease(name: str, mim: int, category: str, *, trials_per_disease: int) -> dict:
    """
    Run ClinVar, ChEMBL, Reactome, Orphanet, and ClinicalTrials ingestion for
    a single disease (OMIM is ingested in bulk beforehand), returning a record
    of what each source produced.

    Gene-driven sources (ClinVar, gene-target ChEMBL, Reactome) use the genes
    OMIM found for this disease; if OMIM yielded none, those steps are marked
    "skipped" and only the name-driven sources run.

    Args:
        name: Canonical disease name.
        mim:  OMIM MIM number.
        category: "rare" or "chronic".
        trials_per_disease: Cap on ClinicalTrials results.

    Returns:
        A per-disease summary record.
    """
    genes = _genes_for(mim)
    rec: dict = {
        "mim": mim,
        "category": category,
        "genes": genes,
        "omim": "ok" if Path(f"data/processed/omim_{mim}.json").exists() else "fail",
        "steps": {},
        "errors": [],
    }
    logger.info("── %s (MIM %s, %s) — genes: %s", name, mim, category, genes or "none")

    # -- Gene-driven sources ------------------------------------------------
    if genes:
        res, err = _run_step(rec, "clinvar", clinvar.batch_ingest_by_genes, genes)
        count = sum(len(v) for v in res.values()) if res else 0
        rec["steps"]["clinvar"] = _status(res, err, count=count)

        res, err = _run_step(rec, "chembl_gene", chembl.batch_ingest_by_genes, genes)
        count = len({n["id"] for v in res.values() for n in v}) if res else 0
        rec["steps"]["chembl_gene"] = _status(res, err, count=count)

        res, err = _run_step(rec, "reactome", reactome.batch_ingest_by_genes, genes)
        count = len({n["pathway_id"] for v in res.values() for n in v}) if res else 0
        rec["steps"]["reactome"] = _status(res, err, count=count)
    else:
        for src in ("clinvar", "chembl_gene", "reactome"):
            rec["steps"][src] = {"status": "skipped", "reason": "no OMIM genes"}

    # -- Indication-driven ChEMBL (symptomatic / ERT / biologics) -----------
    res, err = _run_step(
        rec, "chembl_indication", chembl.batch_ingest_by_indication, {name: name}
    )
    count = len(res.get(name, [])) if res else 0
    rec["steps"]["chembl_indication"] = _status(res, err, count=count)

    # -- Name-driven sources ------------------------------------------------
    res, err = _run_step(rec, "orphanet", orphanet.batch_ingest, [name])
    rec["steps"]["orphanet"] = _status(res, err, count=len(res) if res is not None else 0)

    res, err = _run_step(
        rec, "clinicaltrials", clinicaltrials.batch_ingest_by_disease,
        [name], max_results=trials_per_disease,
    )
    count = len(res.get(name, [])) if res else 0
    rec["steps"]["clinicaltrials"] = _status(res, err, count=count)

    return rec


# ---------------------------------------------------------------------------
# Summary assembly
# ---------------------------------------------------------------------------

def _tally(per_disease: dict) -> dict:
    """Aggregate per-source ok/empty/fail/skipped counts across all diseases."""
    sources = ["omim", "clinvar", "chembl_gene", "chembl_indication",
               "reactome", "orphanet", "clinicaltrials"]
    totals = {s: {"ok": 0, "empty": 0, "fail": 0, "skipped": 0} for s in sources}
    for rec in per_disease.values():
        # OMIM is stored as a flat "ok"/"fail" string.
        totals["omim"]["ok" if rec.get("omim") == "ok" else "fail"] += 1
        for src, info in rec.get("steps", {}).items():
            status = info.get("status", "fail")
            totals[src][status] = totals[src].get(status, 0) + 1
    return totals


def _failures(per_disease: dict) -> list[dict]:
    """Flatten every recorded error into a scannable failures list."""
    out = []
    for name, rec in per_disease.items():
        for err in rec.get("errors", []):
            source, _, msg = err.partition(": ")
            out.append({"disease": name, "source": source, "error": msg})
    return out


def write_summary(summary: dict) -> None:
    """Persist the summary to data/processed/seed_summary.json."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(diseases: list[tuple[str, int, str]], *, skip_graph: bool,
        trials_per_disease: int) -> dict:
    """
    Run the full seed pipeline over `diseases` and return the summary dict.

    OMIM is ingested in bulk first, then each disease's remaining sources are
    ingested in turn (checkpointing the summary after each), then entity
    resolution and the graph rebuild run once at the end.

    Args:
        diseases: List of (name, mim, category) tuples.
        skip_graph: If True, skip entity resolution + build_full_graph.
        trials_per_disease: Cap on ClinicalTrials results per disease.

    Returns:
        The summary dict (also written to disk).
    """
    started = _now()
    t0 = time.time()

    n_rare = sum(1 for _, _, c in diseases if c == "rare")
    n_chronic = sum(1 for _, _, c in diseases if c == "chronic")
    summary: dict = {
        "started_at": started,
        "finished_at": None,
        "duration_seconds": None,
        "total_diseases": len(diseases),
        "rare": n_rare,
        "chronic": n_chronic,
        "note": (
            "This corpus is the list provided for seeding "
            f"({n_rare} rare + {n_chronic} chronic = {len(diseases)}). "
            "MIM 163950 is shared by Turner and Noonan as supplied."
        ),
        "diseases": {},
        "totals": {},
        "failures": [],
        "graph": None,
    }

    # -- Step 1: OMIM in bulk ----------------------------------------------
    mims = [mim for _, mim, _ in diseases]
    logger.info("Step 1/8 — OMIM batch_ingest for %d MIM numbers", len(mims))
    try:
        omim_nodes = omim.batch_ingest(mims)
        logger.info("OMIM ingest complete: %d/%d nodes saved", len(omim_nodes), len(mims))
    except Exception as e:  # noqa: BLE001
        logger.error("OMIM batch_ingest raised: %s — continuing per-disease", e)

    # -- Steps 2-6: per disease --------------------------------------------
    for i, (name, mim, category) in enumerate(diseases, start=1):
        logger.info("[%d/%d] Ingesting %s", i, len(diseases), name)
        rec = ingest_disease(name, mim, category, trials_per_disease=trials_per_disease)
        summary["diseases"][name] = rec
        # Checkpoint after every disease so an interrupted run keeps progress.
        summary["totals"] = _tally(summary["diseases"])
        summary["failures"] = _failures(summary["diseases"])
        write_summary(summary)

    # -- Steps 7-8: entity resolution + graph ------------------------------
    if skip_graph:
        logger.info("Skipping entity resolution + graph build (--skip-graph)")
        summary["graph"] = {"status": "skipped"}
    else:
        logger.info("Step 7/8 — entity resolution (build_disease_map → export_unified_nodes)")
        try:
            disease_map = entity_resolution.build_disease_map()
            saved = entity_resolution.export_unified_nodes(disease_map)
            logger.info("Entity resolution exported %d unified nodes", len(saved))

            logger.info("Step 8/8 — build_full_graph (rebuilds data/bioiq.db)")
            conn = builder.build_full_graph()
            tables = {}
            for table in ("disease", "genes", "drugs", "pathways", "trials"):
                (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                tables[table] = count
            conn.close()
            summary["graph"] = {"status": "ok", "unified_nodes": len(saved), "tables": tables}
            logger.info("Graph build complete: %s", tables)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            logger.error("Graph stage failed: %s", msg)
            summary["graph"] = {"status": "fail", "error": msg}

    summary["finished_at"] = _now()
    summary["duration_seconds"] = round(time.time() - t0, 1)
    write_summary(summary)
    return summary


def print_report(summary: dict) -> None:
    """Print a concise end-of-run report of successes and failures."""
    bar = "=" * 72
    print(f"\n{bar}\n  SEED COMPLETE — {summary['total_diseases']} diseases "
          f"({summary['rare']} rare, {summary['chronic']} chronic) "
          f"in {summary['duration_seconds']}s\n{bar}")
    print("\nPer-source outcomes (ok / empty / fail / skipped):")
    for src, t in summary["totals"].items():
        print(f"  {src:20} ok={t['ok']:<3} empty={t['empty']:<3} "
              f"fail={t['fail']:<3} skipped={t['skipped']:<3}")
    fails = summary["failures"]
    print(f"\nFailures: {len(fails)}")
    for f in fails[:25]:
        print(f"  • {f['disease']} / {f['source']}: {f['error']}")
    if len(fails) > 25:
        print(f"  … and {len(fails) - 25} more (see {SUMMARY_PATH})")
    g = summary.get("graph") or {}
    print(f"\nGraph: {g.get('status')}"
          + (f" — {g.get('tables')}" if g.get("tables") else ""))
    print(f"\nFull summary written to {SUMMARY_PATH}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args and run (or preview) the seed pipeline."""
    parser = argparse.ArgumentParser(description="Seed the BioIQ knowledge graph.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without touching any API or file.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N diseases (for testing).")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Run ingestion only; do not rebuild data/bioiq.db.")
    parser.add_argument("--trials-per-disease", type=int, default=200,
                        help="Max ClinicalTrials results per disease (default 200).")
    args = parser.parse_args()

    diseases = DISEASES[: args.limit] if args.limit else DISEASES

    if args.dry_run:
        print(f"DRY RUN — would seed {len(diseases)} diseases "
              f"({sum(1 for *_, c in diseases if c == 'rare')} rare, "
              f"{sum(1 for *_, c in diseases if c == 'chronic')} chronic).")
        print("Per disease: OMIM → ClinVar → ChEMBL(gene+indication) → "
              "Reactome → Orphanet → ClinicalTrials")
        print("Then: entity resolution → build_full_graph "
              f"({'SKIPPED' if args.skip_graph else 'rebuilds data/bioiq.db'})\n")
        for i, (name, mim, cat) in enumerate(diseases, 1):
            print(f"  {i:2}. {name:34} MIM {mim}  [{cat}]")
        return

    summary = run(diseases, skip_graph=args.skip_graph,
                  trials_per_disease=args.trials_per_disease)
    print_report(summary)


if __name__ == "__main__":
    main()
