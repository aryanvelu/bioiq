"""
Reactome ingestion module.

Provides functions to search Reactome for pathways a gene participates in,
fetch full pathway details (participant genes and top-level pathway
category), normalize raw responses into knowledge-graph-ready records, and
batch-process a list of gene symbols into data/processed/reactome/.

Uses the Reactome ContentService REST API
(https://reactome.org/ContentService), which is open, requires no
authentication, and is free to use commercially.
"""

import json
import logging
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://reactome.org/ContentService"

REQUEST_DELAY = 0.5  # seconds between API calls, per project rate-limit policy

HUMAN_TAXON_ID = "9606"

RAW_DIR = Path("data/raw/reactome")
PROCESSED_DIR = Path("data/processed/reactome")
CACHE_DIR = Path("data/cache")

for _dir in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _get(url, params=None):
    """
    Perform a GET request against the Reactome ContentService API with a
    rate-limit delay.

    Args:
        url (str): Full URL to request.
        params (dict | None): Optional query parameters.

    Returns:
        object | None: Parsed JSON response (dict or list), or None if the
            resource was not found (HTTP 404) or the request failed.
    """
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("Reactome request to %s failed: %s", url, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY)


def search_pathways_by_gene(gene_symbol):
    """
    Search Reactome for all human pathways a gene participates in.

    Uses the Reactome mapping endpoint, which maps an HGNC gene symbol to
    every pathway containing a product of that gene, directly or as part of
    a complex or reaction. The raw response is saved to
    data/raw/reactome/search_{gene_symbol}.json.

    Args:
        gene_symbol (str): HGNC gene symbol, e.g. "CFTR".

    Returns:
        list[dict]: One dict per pathway, each with 'pathway_id' (str, e.g.
            "R-HSA-382556"), 'pathway_name' (str), and 'species' (str).
            Returns an empty list if the gene maps to no pathways.
    """
    url = f"{BASE_URL}/data/mapping/HGNC/{gene_symbol}/pathways"
    data = _get(url, params={"species": HUMAN_TAXON_ID})
    if not data:
        logger.info("No Reactome pathways found for gene %r", gene_symbol)
        return []

    raw_path = RAW_DIR / f"search_{gene_symbol}.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)

    return [
        {
            "pathway_id": p.get("stId"),
            "pathway_name": p.get("displayName"),
            "species": p.get("speciesName"),
        }
        for p in data
    ]


def fetch_pathway_detail(pathway_id, use_cache=True):
    """
    Fetch full Reactome detail for a pathway.

    Combines three ContentService endpoints: query (name, species,
    summation), participants (every physical entity in the pathway, used to
    derive participant genes), and ancestors (the chain of parent pathways up
    to a top-level pathway category such as "Metabolism", "Signal
    Transduction", or "Immune System"). The combined raw response is saved to
    data/raw/reactome/{pathway_id}.json and cached at
    data/cache/reactome_{pathway_id}.json.

    Args:
        pathway_id (str): Reactome stable ID, e.g. "R-HSA-382556".
        use_cache (bool): If True (default), return the cached raw response
            when one exists instead of calling the API.

    Returns:
        dict | None: A dict with 'query' (dict), 'participants' (list), and
            'ancestors' (list of ancestor-chain lists, root pathway last).
            Returns None if the pathway does not exist.
    """
    cache_path = CACHE_DIR / f"reactome_{pathway_id}.json"

    if use_cache and cache_path.exists():
        logger.info("Loading Reactome entry %s from cache", pathway_id)
        with open(cache_path) as f:
            return json.load(f)

    query = _get(f"{BASE_URL}/data/query/{pathway_id}")
    if query is None:
        logger.error("No Reactome pathway found for %s", pathway_id)
        return None

    participants = _get(f"{BASE_URL}/data/participants/{pathway_id}") or []
    ancestors = _get(f"{BASE_URL}/data/event/{pathway_id}/ancestors") or []

    entry = {"query": query, "participants": participants, "ancestors": ancestors}

    raw_path = RAW_DIR / f"{pathway_id}.json"
    with open(raw_path, "w") as f:
        json.dump(entry, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(entry, f, indent=2)

    logger.info("Fetched Reactome entry %s (%s)", pathway_id, query.get("displayName"))
    return entry


def normalize_reactome_entry(raw_entry):
    """
    Convert a raw Reactome pathway entry into a clean, structured
    knowledge-graph node.

    Args:
        raw_entry (dict): The raw entry dict returned by
            fetch_pathway_detail, with 'query', 'participants', and
            'ancestors' keys.

    Returns:
        dict | None: A dict with 'id' (pathway_id), 'source', 'pathway_id',
            'pathway_name', 'participant_genes' (sorted list of gene symbols
            for every ReferenceGeneProduct in the pathway), and
            'top_level_pathway' (dict with 'id' and 'name' identifying the
            highest-level ancestor pathway -- e.g. "Metabolism", "Signal
            Transduction", "Immune System" -- or the pathway itself if it has
            no ancestors). Returns None if raw_entry has no query data.
    """
    query = (raw_entry or {}).get("query")
    if not query:
        return None

    participant_genes = sorted(
        {
            parts[-1]
            for entity in raw_entry.get("participants") or []
            for ref in entity.get("refEntities") or []
            if ref.get("schemaClass") == "ReferenceGeneProduct"
            for parts in (ref.get("displayName", "").split(),)
            if len(parts) > 1
        }
    )

    ancestors = raw_entry.get("ancestors") or []
    if ancestors and ancestors[0]:
        top = ancestors[0][-1]
        top_level_pathway = {"id": top.get("stId"), "name": top.get("displayName")}
    else:
        top_level_pathway = {"id": query.get("stId"), "name": query.get("displayName")}

    pathway_id = query.get("stId")
    return {
        "id": pathway_id,
        "source": "Reactome",
        "pathway_id": pathway_id,
        "pathway_name": query.get("displayName"),
        "participant_genes": participant_genes,
        "top_level_pathway": top_level_pathway,
    }


def batch_ingest_by_genes(gene_symbols):
    """
    Search, fetch, normalize, and save Reactome pathway entries for a list of
    genes.

    For each gene symbol, finds every pathway it participates in, fetches the
    full Reactome entry for each pathway, normalizes it, and writes the
    result to data/processed/reactome/{pathway_id}.json. A pathway shared by
    more than one of the given genes is only fetched and saved once. A
    failure on any individual gene or pathway is logged and skipped so it does
    not stop processing of the rest.

    Args:
        gene_symbols (list[str]): HGNC gene symbols to search, e.g.
            ["CFTR", "HTT"].

    Returns:
        dict[str, list[dict]]: Maps each gene symbol to the list of
            normalized pathway nodes it participates in (possibly empty).
    """
    results = {}
    seen_nodes = {}

    for gene_symbol in gene_symbols:
        logger.info("Searching Reactome for pathways containing %r", gene_symbol)

        matches = search_pathways_by_gene(gene_symbol)
        if not matches:
            results[gene_symbol] = []
            continue

        nodes = []
        for match in matches:
            pathway_id = match["pathway_id"]

            if pathway_id in seen_nodes:
                nodes.append(seen_nodes[pathway_id])
                continue

            raw_entry = fetch_pathway_detail(pathway_id)
            if raw_entry is None:
                logger.error("Skipping %s: fetch failed", pathway_id)
                continue

            node = normalize_reactome_entry(raw_entry)
            if node is None:
                logger.error("Skipping %s: could not normalize entry", pathway_id)
                continue

            out_path = PROCESSED_DIR / f"{pathway_id}.json"
            with open(out_path, "w") as f:
                json.dump(node, f, indent=2)
            logger.info("Saved normalized Reactome node to %s", out_path)

            seen_nodes[pathway_id] = node
            nodes.append(node)

        results[gene_symbol] = nodes

    return results


if __name__ == "__main__":
    test_genes = ["FBN1", "CFTR", "HTT", "GBA1", "DMD"]
    results = batch_ingest_by_genes(test_genes)
    for gene_symbol, nodes in results.items():
        print(f"\n{gene_symbol}: {len(nodes)} pathway(s)")
        for node in nodes:
            print(
                f"  {node['pathway_id']} | {node['pathway_name']} | "
                f"top_level: {node['top_level_pathway']['name']} | "
                f"{len(node['participant_genes'])} participant genes"
            )
