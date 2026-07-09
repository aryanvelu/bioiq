"""
Orphanet ingestion module.
#Orphanet is the rare disease registry layer - how common the disease is, how it's inherited, at what age it starts, etc
Provides functions to search Orphanet for a disease by name, fetch a full
ORPHAcode entry (cross-references, epidemiology, natural history, and
associated genes), normalize raw responses into knowledge-graph-ready
records, and batch-process a list of disease names into
data/processed/orphanet/.

Uses the Orphadata REST API (https://api.orphadata.com), which is open and
requires no authentication.
"""

import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.orphadata.com"

REQUEST_DELAY = 0.5  # seconds between API calls, per project rate-limit policy

RAW_DIR = Path("data/raw/orphanet")
PROCESSED_DIR = Path("data/processed/orphanet")
CACHE_DIR = Path("data/cache")

for _dir in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Orphadata endpoints that together make up a "full entry" for an ORPHAcode.
ENTRY_ENDPOINTS = {
    "cross_referencing": "rd-cross-referencing",
    "epidemiology": "rd-epidemiology",
    "natural_history": "rd-natural_history",
    "associated_genes": "rd-associated-genes",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def search_orphanet(disease_name):
    """
    Search Orphanet for a disease by name.

    Uses the Orphadata cross-referencing endpoint, which performs a
    case-insensitive match against an Orphanet disorder's preferred name.

    Args:
        disease_name (str): The disease name to search for, e.g.
            "Marfan syndrome".

    Returns:
        list[dict]: One dict per match, each with 'orphacode' (int) and
            'preferred_name' (str). Returns an empty list if the request
            fails or no disorder matches that name.
    """
    url = f"{BASE_URL}/rd-cross-referencing/orphacodes/names/{quote(disease_name)}"

    try:
        response = requests.get(url, params={"lang": "en"}, timeout=20)
        if response.status_code == 404:
            logger.warning("No Orphanet entry found for name %r", disease_name)
            return []
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("Orphanet search for %r failed: %s", disease_name, e)
        return []
    finally:
        time.sleep(REQUEST_DELAY)

    results = data.get("data", {}).get("results")
    if not results:
        return []

    if isinstance(results, dict):
        results = [results]

    safe_name = re.sub(r"[^\w-]", "_", disease_name)
    raw_path = RAW_DIR / f"search_{safe_name}.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)

    return [
        {"orphacode": item.get("ORPHAcode"), "preferred_name": item.get("Preferred term")}
        for item in results
    ]


def _fetch_subtype_gene_associations(preferred_name):
    """
    Fallback for umbrella-level ORPHAcodes that return HTTP 404 from the
    rd-associated-genes endpoint.

    Orphanet stores gene associations on disease subtypes rather than the
    parent code.  This function tries searching for "[preferred_name] type N"
    variants (N = 1–6), fetches gene associations for each found subtype,
    and returns the aggregated unique set.

    Args:
        preferred_name (str): The disease's preferred name, e.g.
            "Gaucher disease" or "Marfan syndrome".

    Returns:
        list[dict]: Aggregated DisorderGeneAssociation records (same
            structure as the rd-associated-genes results), de-duplicated by
            gene symbol.  Empty list if no subtypes are found.
    """
    seen_symbols = set()
    assocs = []

    for n in range(1, 7):
        subtype_name = f"{preferred_name} type {n}"
        url = f"{BASE_URL}/rd-cross-referencing/orphacodes/names/{quote(subtype_name)}"
        try:
            r = requests.get(url, params={"lang": "en"}, timeout=20)
        except requests.RequestException as e:
            logger.warning("Subtype search for %r failed: %s", subtype_name, e)
            time.sleep(REQUEST_DELAY)
            continue
        finally:
            time.sleep(REQUEST_DELAY)

        if r.status_code == 404:
            continue
        if r.status_code != 200:
            logger.warning("Subtype search %r returned HTTP %s", subtype_name, r.status_code)
            continue

        results = r.json().get("data", {}).get("results")
        if not results:
            continue
        if isinstance(results, dict):
            results = [results]

        for item in results:
            subcode = item.get("ORPHAcode")
            if not subcode:
                continue
            gene_url = f"{BASE_URL}/rd-associated-genes/orphacodes/{subcode}"
            try:
                r2 = requests.get(gene_url, params={"lang": "en"}, timeout=20)
            except requests.RequestException as e:
                logger.warning("Gene fetch for ORPHA:%s failed: %s", subcode, e)
                time.sleep(REQUEST_DELAY)
                continue
            finally:
                time.sleep(REQUEST_DELAY)

            if r2.status_code != 200:
                continue
            sub_results = r2.json().get("data", {}).get("results") or {}
            for assoc in sub_results.get("DisorderGeneAssociation") or []:
                sym = (assoc.get("Gene") or {}).get("Symbol")
                if sym and sym not in seen_symbols:
                    seen_symbols.add(sym)
                    assocs.append(assoc)

    if assocs:
        logger.info(
            "Subtype fallback for %r found %d unique gene(s): %s",
            preferred_name,
            len(assocs),
            sorted(seen_symbols),
        )
    return assocs


def fetch_orphanet_entry(orphacode, use_cache=True):
    """
    Fetch the full Orphanet entry for an ORPHAcode.

    Combines four Orphadata endpoints: cross-references (incl. OMIM/UMLS/
    MONDO mappings), epidemiology (prevalence), natural history
    (inheritance, age of onset), and associated genes.

    For umbrella-level ORPHAcodes (e.g. ORPHA:355 Gaucher disease,
    ORPHA:558 Marfan syndrome), the rd-associated-genes endpoint returns
    404 because gene associations live on subtypes.  In that case a
    subtype search is performed automatically.

    The combined raw response is saved to data/raw/orphanet/{orphacode}.json
    and cached at data/cache/orphanet_{orphacode}.json.

    Args:
        orphacode (str | int): The Orphanet ORPHAcode (ORPHA number).
        use_cache (bool): If True (default), return the cached raw
            response when one exists instead of calling the API.

    Returns:
        dict | None: A dict with 'cross_referencing', 'epidemiology',
            'natural_history', and 'associated_genes' keys, each holding
            the raw 'results' object from the corresponding endpoint (or
            None if that endpoint had no data for this ORPHAcode). Returns
            None if the core cross-referencing lookup fails or the
            ORPHAcode does not exist.
    """
    orphacode = str(orphacode)
    cache_path = CACHE_DIR / f"orphanet_{orphacode}.json"

    if use_cache and cache_path.exists():
        logger.info("Loading Orphanet entry %s from cache", orphacode)
        with open(cache_path) as f:
            return json.load(f)

    entry = {}
    for key, path in ENTRY_ENDPOINTS.items():
        url = f"{BASE_URL}/{path}/orphacodes/{orphacode}"
        try:
            response = requests.get(url, params={"lang": "en"}, timeout=20)
            if response.status_code == 404:
                logger.info("No %s data for ORPHAcode %s", key, orphacode)
                entry[key] = None
            else:
                response.raise_for_status()
                entry[key] = response.json().get("data", {}).get("results")
        except (requests.RequestException, ValueError) as e:
            logger.error("Failed to fetch %s for ORPHAcode %s: %s", key, orphacode, e)
            entry[key] = None
        finally:
            time.sleep(REQUEST_DELAY)

    if entry.get("cross_referencing") is None:
        logger.error("No Orphanet entry found for ORPHAcode %s", orphacode)
        return None

    # Fallback for umbrella diseases: the rd-associated-genes endpoint returns
    # 404 for parent ORPHAcodes; gene data lives on subtype codes instead.
    if entry.get("associated_genes") is None:
        preferred_name = entry["cross_referencing"].get("Preferred term", "")
        logger.info(
            "ORPHA:%s (%s): rd-associated-genes returned 404 — trying subtype fallback",
            orphacode, preferred_name,
        )
        subtype_assocs = _fetch_subtype_gene_associations(preferred_name)
        if subtype_assocs:
            entry["associated_genes"] = {"DisorderGeneAssociation": subtype_assocs}

    raw_path = RAW_DIR / f"{orphacode}.json"
    with open(raw_path, "w") as f:
        json.dump(entry, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(entry, f, indent=2)

    logger.info("Fetched Orphanet entry %s", orphacode)
    return entry


def normalize_orphanet_entry(raw_entry):
    """
    Convert a raw Orphanet entry into a clean, structured knowledge-graph
    node.

    Args:
        raw_entry (dict): The raw entry dict returned by
            fetch_orphanet_entry, with 'cross_referencing', 'epidemiology',
            'natural_history', and 'associated_genes' keys.

    Returns:
        dict | None: A dict with 'id' ("ORPHA:<orphacode>"), 'source',
            'orphacode', 'preferred_name', 'prevalence' (list of records
            with type/class/geographic area/value), 'inheritance' (list of
            inheritance patterns), 'onset_age' (list of typical ages of
            onset), 'associated_genes' (list of gene dicts with symbol,
            name, association type/status, and OMIM/Ensembl/HGNC IDs), and
            'cross_refs' (dict with 'omim' and 'omim_gene' lists plus other
            database cross-references, for entity resolution). Returns
            None if raw_entry has no cross-referencing data.
    """
    cross_ref = (raw_entry or {}).get("cross_referencing")
    if not cross_ref:
        return None

    prevalence = [
        {
            "type": p.get("PrevalenceType"),
            "class": p.get("PrevalenceClass"),
            "geographic_area": p.get("PrevalenceGeographic"),
            "value": p.get("ValMoy"),
            "validation_status": p.get("PrevalenceValidationStatus"),
        }
        for p in (raw_entry.get("epidemiology") or {}).get("Prevalence") or []
    ]

    natural_history = raw_entry.get("natural_history") or {}
    inheritance = natural_history.get("TypeOfInheritance") or []
    onset_age = natural_history.get("AverageAgeOfOnset") or []

    associated_genes = []
    for assoc in (raw_entry.get("associated_genes") or {}).get("DisorderGeneAssociation") or []:
        gene = assoc.get("Gene", {})
        gene_refs = {ref.get("Source"): ref.get("Reference") for ref in gene.get("ExternalReference") or []}
        associated_genes.append(
            {
                "symbol": gene.get("Symbol"),
                "name": gene.get("name"),
                "association_type": assoc.get("DisorderGeneAssociationType"),
                "association_status": assoc.get("DisorderGeneAssociationStatus"),
                "omim_gene_id": gene_refs.get("OMIM"),
                "ensembl_id": gene_refs.get("Ensembl"),
                "hgnc_id": gene_refs.get("HGNC"),
            }
        )

    omim_refs = set()
    other_refs = {}
    for ref in cross_ref.get("ExternalReference") or []:
        source, reference = ref.get("Source"), ref.get("Reference")
        if not source or not reference:
            continue
        if source == "OMIM":
            omim_refs.add(str(reference))
        else:
            other_refs.setdefault(source.lower(), []).append(str(reference))

    cross_refs = {
        "omim": sorted(omim_refs),
        "omim_gene": sorted({g["omim_gene_id"] for g in associated_genes if g.get("omim_gene_id")}),
        **other_refs,
    }

    orphacode = cross_ref.get("ORPHAcode")
    return {
        "id": f"ORPHA:{orphacode}",
        "source": "Orphanet",
        "orphacode": orphacode,
        "preferred_name": cross_ref.get("Preferred term"),
        "prevalence": prevalence,
        "inheritance": inheritance,
        "onset_age": onset_age,
        "associated_genes": associated_genes,
        "cross_refs": cross_refs,
    }


def batch_ingest(disease_names):
    """
    Search, fetch, normalize, and save Orphanet entries for a list of
    disease names.

    For each name, finds the matching ORPHAcode, fetches its full entry,
    normalizes it, and writes the result to
    data/processed/orphanet/{orphacode}.json. A failure on any individual
    disease is logged and skipped so it does not stop processing of the
    rest.

    Args:
        disease_names (list[str]): Disease names to ingest, e.g.
            ["Marfan syndrome", "Cystic fibrosis"].

    Returns:
        list[dict]: Normalized nodes that were successfully fetched,
            normalized, and saved.
    """
    nodes = []
    for disease_name in disease_names:
        logger.info("Processing Orphanet entry for %r", disease_name)

        matches = search_orphanet(disease_name)
        if not matches:
            logger.error("Skipping %r: no Orphanet match found", disease_name)
            continue

        orphacode = matches[0]["orphacode"]
        raw_entry = fetch_orphanet_entry(orphacode)
        if raw_entry is None:
            logger.error("Skipping %r (ORPHA:%s): fetch failed", disease_name, orphacode)
            continue

        node = normalize_orphanet_entry(raw_entry)
        if node is None:
            logger.error("Skipping %r (ORPHA:%s): could not normalize entry", disease_name, orphacode)
            continue

        out_path = PROCESSED_DIR / f"{orphacode}.json"
        with open(out_path, "w") as f:
            json.dump(node, f, indent=2)
        logger.info("Saved normalized Orphanet node to %s", out_path)

        nodes.append(node)

    return nodes


if __name__ == "__main__":
    test_diseases = [
        "Marfan syndrome",
        "Cystic fibrosis",
        "Huntington disease",
        "Gaucher disease",
        "Duchenne muscular dystrophy",
    ]
    results = batch_ingest(test_diseases)
    for node in results:
        print(
            f"{node['id']} | {node['preferred_name']} | "
            f"inheritance: {node['inheritance']} | "
            f"genes: {[g['symbol'] for g in node['associated_genes']]}"
        )
