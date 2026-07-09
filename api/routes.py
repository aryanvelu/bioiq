"""
api/routes.py

FastAPI route handlers for BioIQ.

Endpoints:
  GET  /health   — liveness + database reachability check
  POST /query    — generate a research brief for a named disease
  GET  /diseases — list all disease names currently in the knowledge graph
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

DB_PATH = Path(__file__).parent.parent / "data" / "bioiq.db"
PATTERNS_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "cross_disease_patterns.json"
)
BRIEFS_DIR = Path(__file__).parent.parent / "data" / "processed" / "briefs"
HYPOTHESES_DIR = Path(__file__).parent.parent / "data" / "processed" / "hypotheses"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    disease: str


class HypothesisRequest(BaseModel):
    disease: str
    refresh: bool = False


class QueryResponse(BaseModel):
    disease: str
    brief: str
    status: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    database: str


class DiseasesResponse(BaseModel):
    diseases: list[str]


class PatternsResponse(BaseModel):
    disease: str
    drug_overlaps: list[dict]
    mechanism_themes: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db_conn() -> sqlite3.Connection:
    """Open a short-lived SQLite connection to the knowledge graph."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _brief_cache_path(disease_name: str) -> Path:
    """
    Path where synthesis.engine caches this disease's brief.

    Mirrors the engine's own key (spaces and slashes -> underscores) so the
    route can tell a cache hit from a fresh generation without importing engine.
    """
    safe = disease_name.replace(" ", "_").replace("/", "_")
    return BRIEFS_DIR / f"{safe}.json"


def _hypotheses_cache_path(disease_name: str) -> Path:
    """Path where synthesis.engine caches this disease's hypotheses (same key)."""
    safe = disease_name.replace(" ", "_").replace("/", "_")
    return HYPOTHESES_DIR / f"{safe}.json"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — confirms the API is running and the database is reachable."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database file not found")
    try:
        conn = _db_conn()
        conn.execute("SELECT 1")
        conn.close()
    except sqlite3.Error as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
    return {"status": "ok", "database": "reachable"}


@router.get("/diseases", response_model=DiseasesResponse)
def list_diseases():
    """Return all disease names currently stored in the knowledge graph."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database file not found")
    try:
        conn = _db_conn()
        rows = conn.execute("SELECT name FROM disease ORDER BY name").fetchall()
        conn.close()
    except sqlite3.Error as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
    return {"diseases": [row["name"] for row in rows]}


@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query_disease(request: Request, body: QueryRequest):
    """
    Generate a structured research brief for a disease, ingesting it first if new.

    If the disease is already in the knowledge graph, its brief is generated
    (or served from cache). If it is NOT in the graph, it is ingested on demand
    first — OMIM -> ClinVar -> ChEMBL -> Reactome -> Orphanet -> ClinicalTrials
    -> entity resolution -> graph rebuild — which adds 15-30+ seconds before the
    brief is synthesized. This lets ANY disease, not just the seeded corpus, be
    briefed.

    The response carries a `status` field:
      • "cached"                 — brief served from the existing cache
      • "generated"              — disease was in the graph; brief freshly generated
      • "ingested_and_generated" — disease was new; full pipeline ran, then brief
    """
    disease_name = body.disease.strip()
    if not disease_name:
        raise HTTPException(status_code=422, detail="disease field must not be empty")

    # Is the disease already in the knowledge graph? (Also normalises casing.)
    try:
        conn = _db_conn()
        row = conn.execute(
            "SELECT name FROM disease WHERE name = ? COLLATE NOCASE",
            (disease_name,),
        ).fetchone()
        conn.close()
    except sqlite3.Error as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    newly_ingested = False
    if row is not None:
        canonical_name = row["name"]
    else:
        # Unknown disease — ingest it end-to-end before briefing. 15-30+ seconds.
        from graph.builder import ingest_on_demand

        ok, detail = ingest_on_demand(disease_name)
        if not ok:
            raise HTTPException(
                status_code=502,
                detail=f"On-demand ingestion of '{disease_name}' failed: {detail}",
            )
        canonical_name = detail
        newly_ingested = True

    # Did a cached brief already exist BEFORE this call? Determines the status:
    # a pre-existing cache is "cached"; otherwise we generate it now.
    brief_was_cached = _brief_cache_path(canonical_name).exists()

    # Lazy import keeps startup fast and avoids a circular-import at module level
    from synthesis.engine import generate_brief

    brief = generate_brief(canonical_name)
    if brief.startswith("ERROR:"):
        raise HTTPException(status_code=500, detail=brief)

    if newly_ingested:
        status = "ingested_and_generated"
    elif brief_was_cached:
        status = "cached"
    else:
        status = "generated"

    return {
        "disease": canonical_name,
        "brief": brief,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/hypotheses")
@limiter.limit("10/minute")
def hypotheses_endpoint(request: Request, body: HypothesisRequest):
    """
    Generate structured forward-looking research hypotheses for a disease.

    Returns a JSON object with four sections:
      therapeutic_hypotheses  — 5 ranked novel therapeutic approaches not yet in trials
      repurposing_candidates  — 3 approved drugs most worth repurposing
      experimental_priorities — 3 experiments ranked by expected information value
      regulatory_pathway      — FDA designation assessment and strategic implications

    Results are cached to data/processed/hypotheses/ — repeat calls for the
    same disease return instantly without hitting the Claude API.

    If the disease is not yet in the knowledge graph it is ingested on demand
    first (same pipeline as POST /query), adding 15-30+ seconds before the
    hypotheses are generated. The response carries a `status` field:
      • "cached"                 — hypotheses served from the existing cache
      • "generated"              — disease was in the graph; freshly generated
      • "ingested_and_generated" — disease was new; full pipeline ran, then generated
    """
    disease_name = body.disease.strip()
    if not disease_name:
        raise HTTPException(status_code=422, detail="disease field must not be empty")

    # Is the disease already in the knowledge graph? (Also normalises casing.)
    try:
        conn = _db_conn()
        row = conn.execute(
            "SELECT name FROM disease WHERE name = ? COLLATE NOCASE",
            (disease_name,),
        ).fetchone()
        conn.close()
    except sqlite3.Error as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    newly_ingested = False
    if row is not None:
        canonical_name = row["name"]
    else:
        # Unknown disease — ingest it end-to-end before generating. 15-30+ seconds.
        from graph.builder import ingest_on_demand

        ok, detail = ingest_on_demand(disease_name)
        if not ok:
            raise HTTPException(
                status_code=502,
                detail=f"On-demand ingestion of '{disease_name}' failed: {detail}",
            )
        canonical_name = detail
        newly_ingested = True

    # Was a cached result present BEFORE this call? (Determines the status.) A
    # forced refresh always counts as a fresh generation.
    hyp_was_cached = (
        _hypotheses_cache_path(canonical_name).exists() and not body.refresh
    )

    from synthesis.engine import generate_hypotheses

    result = generate_hypotheses(canonical_name, refresh=body.refresh)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    if newly_ingested:
        result["status"] = "ingested_and_generated"
    elif hyp_was_cached:
        result["status"] = "cached"
    else:
        result["status"] = "generated"

    return result


@router.get("/patterns/{disease_name}", response_model=PatternsResponse)
def patterns_endpoint(disease_name: str):
    """
    Return the genuine cross-disease overlaps that involve a given disease.

    Reads the cached data/processed/cross_disease_patterns.json (produced by
    scripts/cross_disease_analysis.py) and returns only the overlaps the
    reasoning pass rated as genuine — filtered to those whose `diseases` list
    contains the queried name (case-insensitive).

    This endpoint never 404s. If the patterns file does not exist yet, cannot be
    read, or simply has no genuine overlaps touching this disease, it returns
    200 with empty arrays. The lists fill in as more diseases are analysed.

    Args:
        disease_name: Disease name to filter overlaps by (case-insensitive).

    Returns:
        {disease, drug_overlaps, mechanism_themes} — the two arrays hold only
        the genuine, disease-relevant entries.
    """
    target = disease_name.strip().lower()
    empty = {"disease": disease_name, "drug_overlaps": [], "mechanism_themes": []}

    if not PATTERNS_PATH.exists():
        return empty

    try:
        with open(PATTERNS_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # A missing or corrupt cache is not a client error — degrade gracefully.
        return empty

    def _touches(entry: dict) -> bool:
        """True if the queried disease is in this overlap's diseases list."""
        return any(
            isinstance(d, str) and d.strip().lower() == target
            for d in entry.get("diseases", [])
        )

    drug_overlaps = [
        o for o in data.get("genuine_drug_overlaps", []) if _touches(o)
    ]
    mechanism_themes = [
        t for t in data.get("genuine_mechanism_themes", []) if _touches(t)
    ]

    return {
        "disease": disease_name,
        "drug_overlaps": drug_overlaps,
        "mechanism_themes": mechanism_themes,
    }
