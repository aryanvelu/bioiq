"""
OMIM (Online Mendelian Inheritance in Man) ingestion module.
#Ingestion pipeline is the document collection step 
#Has one overall job - go to a specific databse, pull out the data for a disease, clean the data, and save it in a consistent format for the overall backend system to use later
#OMIM gives core biology - what genes cause the disease, what mechanisms look like, what the clinical picture looks like
#Once all sources are ingested, the graph layer connects them all that way system knows that one disease is classified differently in each source, then syntheiss layer reads all connected data and feeds it to the Claude API to generate research brief
Provides functions to search OMIM by disease name, fetch full entries by
MIM number, normalize raw entries into knowledge-graph-ready nodes, and
batch-process a list of MIM numbers into data/processed/.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OMIM_API_KEY = os.getenv("OMIM_API_KEY")
OMIM_BASE_URL = "https://api.omim.org/api"

REQUEST_DELAY = 0.5  # seconds between API calls, per project rate-limit policy

RAW_DIR = Path("data/raw/omim")
PROCESSED_DIR = Path("data/processed")
CACHE_DIR = Path("data/cache")

for _dir in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# OMIM entry prefixes and what they mean, per OMIM's documentation.
ENTRY_TYPE_BY_PREFIX = {
    "*": "gene",
    "#": "phenotype_known_molecular_basis",
    "+": "gene_and_phenotype",
    "%": "phenotype_unknown_molecular_basis",
    "^": "moved_or_removed",
    "": "other",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def search_omim(disease_name, limit=10):
    """
    Search OMIM for entries matching a disease, phenotype, or gene name.

    Args:
        disease_name (str): The disease, phenotype, or gene name to search for.
        limit (int): Maximum number of results to return. Defaults to 10.

    Returns:
        list[dict]: One dict per match, each with 'mim_number' (int),
            'title' (str), and 'matches' (str describing matched terms).
            Returns an empty list if the request fails or no entry exists.
    """
    params = {
        "search": disease_name,
        "format": "json",
        "apiKey": OMIM_API_KEY,
        "limit": limit,
    }

    try:
        response = requests.get(
            f"{OMIM_BASE_URL}/entry/search", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("OMIM search for %r failed: %s", disease_name, e)
        return []
    finally:
        time.sleep(REQUEST_DELAY)

    search_response = data.get("omim", {}).get("searchResponse", {})
    results = []
    for item in search_response.get("entryList", []):
        entry = item.get("entry", {})
        mim_number = entry.get("mimNumber")
        results.append(
            {
                "mim_number": int(mim_number) if mim_number is not None else None,
                "title": entry.get("titles", {}).get("preferredTitle"),
                "matches": entry.get("matches"),
            }
        )

    logger.info(
        "OMIM search for %r returned %d total results (showing %d)",
        disease_name,
        search_response.get("totalResults", 0),
        len(results),
    )
    return results


def fetch_omim_entry(mim_number, use_cache=True):
    """
    Fetch the full OMIM entry for a given MIM number, including text
    sections, gene-phenotype mappings, external database links, literature
    references, and clinical synopsis.

    The raw JSON response is saved to data/raw/omim/{mim_number}.json and
    cached at data/cache/omim_{mim_number}.json so the same MIM number is
    not re-fetched on later runs.

    Args:
        mim_number (str | int): The OMIM MIM number to fetch.
        use_cache (bool): If True (default), return the cached raw response
            when one exists instead of calling the API.

    Returns:
        dict | None: The raw OMIM API JSON response (the full
            '{"omim": {...}}' object), or None if the request failed or no
            entry was found for this MIM number.
    """
    mim_number = str(mim_number)
    cache_path = CACHE_DIR / f"omim_{mim_number}.json"

    if use_cache and cache_path.exists():
        logger.info("Loading OMIM entry %s from cache", mim_number)
        with open(cache_path) as f:
            return json.load(f)

    params = {
        "mimNumber": mim_number,
        "include": "text,geneMap,externalLinks,referenceList,clinicalSynopsis",
        "format": "json",
        "apiKey": OMIM_API_KEY,
    }

    try:
        response = requests.get(
            f"{OMIM_BASE_URL}/entry", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("Failed to fetch OMIM entry %s: %s", mim_number, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY)

    if not data.get("omim", {}).get("entryList"):
        logger.warning("No OMIM entry found for MIM number %s", mim_number)
        return None

    raw_path = RAW_DIR / f"{mim_number}.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Fetched OMIM entry %s", mim_number)
    return data


def _split_csv(raw):
    """
    Split a comma-separated string from OMIM's externalLinks into a list
    of trimmed, non-empty tokens.

    Args:
        raw (str | None): A comma-separated string, or None.

    Returns:
        list[str]: The individual tokens, or an empty list if raw is falsy.
    """
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_orphanet_refs(raw):
    """
    Parse the OMIM externalLinks 'orphanetDiseases' field into a list of
    Orphanet cross-references.

    OMIM encodes one or more Orphanet records in a single string, with
    fields separated by ';;' and records separated by ';;;', e.g.
    "284963;;20628;;Marfan syndrome type 1;;;558;;109;;Marfan syndrome".
    Numeric fields are ORPHA codes; the remaining field is the disease name.

    Args:
        raw (str | None): The raw 'orphanetDiseases' value from
            entry['externalLinks'].

    Returns:
        list[dict]: One dict per Orphanet record, each with 'orpha_ids'
            (list[str]) and 'name' (str | None). Returns an empty list if
            raw is falsy.
    """
    if not raw:
        return []

    refs = []
    for group in raw.split(";;;"):
        tokens = [token for token in group.split(";;") if token]
        if not tokens:
            continue
        orpha_ids = [token for token in tokens if token.isdigit()]
        names = [token for token in tokens if not token.isdigit()]
        refs.append({"orpha_ids": orpha_ids,
                    "name": names[-1] if names else None})
    return refs


def _extract_genes(entry):
    """
    Extract a deduplicated list of gene records from an OMIM entry's
    phenotype-gene map.

    Args:
        entry (dict): The raw 'entry' object from the OMIM API.

    Returns:
        list[dict]: One dict per unique gene (deduplicated by gene MIM
            number), each with 'gene_mim_number', 'approved_gene_symbol',
            'gene_symbols' (list[str]), 'entrez_gene_ids' (list[str]),
            'ensembl_gene_ids' (list[str]), 'cyto_location', 'chromosome',
            and 'inheritance'.
    """
    genes = []
    seen_gene_mims = set()

    for item in entry.get("phenotypeMapList") or []:
        gene_map = item.get("phenotypeMap", {})
        gene_mim = gene_map.get("mimNumber")
        if gene_mim in seen_gene_mims:
            continue
        seen_gene_mims.add(gene_mim)

        gene_symbols = _split_csv(gene_map.get("geneSymbols"))
        entrez_gene_ids = _split_csv(gene_map.get("geneIDs"))
        ensembl_gene_ids = sorted(
            set(re.findall(r"ENSG\d+", gene_map.get("ensemblIDs") or "")))

        genes.append(
            {
                "gene_mim_number": gene_mim,
                "approved_gene_symbol": gene_map.get("approvedGeneSymbols"),
                "gene_symbols": gene_symbols,
                "entrez_gene_ids": entrez_gene_ids,
                "ensembl_gene_ids": ensembl_gene_ids,
                "cyto_location": gene_map.get("cytoLocation"),
                "chromosome": gene_map.get("chromosomeSymbol"),
                "inheritance": gene_map.get("phenotypeInheritance"),
            }
        )

    return genes


def normalize_omim_entry(raw_response):
    """
    Convert a raw OMIM API response into a normalized knowledge-graph node.

    Args:
        raw_response (dict): The raw OMIM API JSON response, as returned by
            fetch_omim_entry (the full '{"omim": {...}}' object).

    Returns:
        dict | None: A normalized dict with canonical IDs ('id',
            'mim_number'), entry metadata ('entry_type', 'status',
            'preferred_title', 'alternative_titles'), a deduplicated
            'genes' list, 'text_sections' keyed by section name, the raw
            'clinical_synopsis', 'cross_references' to Orphanet, UMLS,
            MONDO, MeSH, SNOMED CT, ICD-10-CM, and Disease Ontology, and a
            list of literature 'references'. Returns None if raw_response
            contains no entry.
    """
    entry_list = (raw_response or {}).get("omim", {}).get("entryList", [])
    if not entry_list:
        return None
    entry = entry_list[0].get("entry", {})

    titles = entry.get("titles", {})
    alternative_titles = [
        title.strip()
        for title in (titles.get("alternativeTitles") or "").split(";;")
        if title.strip()
    ]

    text_sections = {}
    for item in entry.get("textSectionList") or []:
        section = item.get("textSection", {})
        name = section.get("textSectionName")
        if name:
            text_sections[name] = section.get("textSectionContent")

    references = []
    for item in entry.get("referenceList") or []:
        ref = item.get("reference", {})
        references.append(
            {
                "pubmed_id": ref.get("pubmedID"),
                "title": ref.get("title"),
                "authors": ref.get("authors"),
                "source": ref.get("source"),
                "doi": ref.get("doi"),
            }
        )

    ext_links = entry.get("externalLinks") or {}
    cross_references = {
        "orphanet": _parse_orphanet_refs(ext_links.get("orphanetDiseases")),
        "umls": _split_csv(ext_links.get("umlsIDs")),
        "mesh": _split_csv(ext_links.get("meshIDs")),
        "mondo": [f"MONDO:{mid}" for mid in _split_csv(ext_links.get("mondoIDs"))],
        "snomedct": _split_csv(ext_links.get("snomedctIDs")),
        "icd10cm": _split_csv(ext_links.get("icd10cmIDs")),
        "disease_ontology": [f"DOID:{did}" for did in _split_csv(ext_links.get("diseaseOntologyIDs"))],
    }

    mim_number = entry.get("mimNumber")
    prefix = entry.get("prefix", "")

    return {
        "id": f"OMIM:{mim_number}",
        "source": "OMIM",
        "mim_number": mim_number,
        "prefix": prefix,
        "entry_type": ENTRY_TYPE_BY_PREFIX.get(prefix, "other"),
        "status": entry.get("status"),
        "preferred_title": titles.get("preferredTitle"),
        "alternative_titles": alternative_titles,
        "genes": _extract_genes(entry),
        "text_sections": text_sections,
        "clinical_synopsis": entry.get("clinicalSynopsis", {}),
        "cross_references": cross_references,
        "references": references,
    }


def batch_ingest(mim_numbers):
    """
    Fetch, normalize, and save OMIM entries for a list of MIM numbers.

    Each normalized node is written to
    data/processed/omim_{mim_number}.json. A failure on any individual MIM
    number is logged and skipped so it does not stop processing of the
    remaining numbers.

    Args:
        mim_numbers (list[str | int]): MIM numbers to ingest.

    Returns:
        list[dict]: Normalized nodes that were successfully fetched,
            normalized, and saved (in the same order as mim_numbers, minus
            any that failed).
    """
    nodes = []
    for mim_number in mim_numbers:
        logger.info("Processing MIM number %s", mim_number)

        raw = fetch_omim_entry(mim_number)
        if raw is None:
            logger.error("Skipping MIM %s: fetch failed", mim_number)
            continue

        node = normalize_omim_entry(raw)
        if node is None:
            logger.error(
                "Skipping MIM %s: could not normalize entry", mim_number)
            continue

        out_path = PROCESSED_DIR / f"omim_{mim_number}.json"
        with open(out_path, "w") as f:
            json.dump(node, f, indent=2)
        logger.info("Saved normalized OMIM node to %s", out_path)

        nodes.append(node)

    return nodes


if __name__ == "__main__":
    test_mim_numbers = [154700, 219700, 143100, 125853, 300100]
    results = batch_ingest(test_mim_numbers)
    for node in results:
        print(
            f"{node['id']} | {node['preferred_title']} | "
            f"genes: {[g['approved_gene_symbol'] for g in node['genes']]}"
        )
