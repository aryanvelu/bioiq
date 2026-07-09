"""
ClinVar ingestion module.
#ClinVar - gives genetics layer such as specific gene variants that are known to cause or contribute to the disease, with their clinical significance rated by experts
Provides functions to search ClinVar for pathogenic/likely pathogenic
variants by gene symbol, fetch full classification detail for a single
variant by variation ID, normalize raw responses into knowledge-graph-ready
records, and batch-process a list of genes into data/processed/clinvar/.

Uses the NCBI Entrez E-utilities (esearch/esummary) against db=clinvar.
"""

import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

REQUEST_DELAY = 0.5  # seconds between API calls, per project rate-limit policy

RAW_DIR = Path("data/raw/clinvar")
PROCESSED_DIR = Path("data/processed/clinvar")
CACHE_DIR = Path("data/cache")

for _dir in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ClinVar germline classification values treated as "pathogenic or likely pathogenic".
PATHOGENIC_CLASSIFICATIONS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def search_clinvar_by_gene(gene_symbol, limit=10):
    """
    Search ClinVar for pathogenic and likely pathogenic variants in a gene.

    Uses the NCBI Entrez esearch endpoint to find candidate variation IDs
    for the gene, then esummary to retrieve a classification summary for
    each, keeping only variants classified as "Pathogenic",
    "Likely pathogenic", or "Pathogenic/Likely pathogenic".

    The combined raw esearch/esummary response is saved to
    data/raw/clinvar/search_{gene_symbol}.json.

    Args:
        gene_symbol (str): HGNC gene symbol, e.g. "FBN1". Note that ClinVar
            indexes the current HGNC symbol, which may differ from older
            aliases (e.g. "GBA" was renamed "GBA1").
        limit (int): Maximum number of variants to return. Defaults to 10.

    Returns:
        list[dict]: One dict per variant, each with 'variation_id' (str),
            'variant_name' (str), 'gene_symbol' (str),
            'clinical_significance' (str), and 'review_status' (str).
            Returns an empty list if the request fails or no matching
            variants are found.
    """
    search_params = {
        "db": "clinvar",
        "term": f'{gene_symbol}[gene] AND ("clinsig pathogenic"[Filter] OR "likely pathogenic"[clinsig])',
        "retmode": "json",
        "retmax": limit * 3,
        "api_key": NCBI_API_KEY,
    }

    try:
        response = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=search_params, timeout=20)
        response.raise_for_status()
        search_data = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("ClinVar search for gene %r failed: %s", gene_symbol, e)
        return []
    finally:
        time.sleep(REQUEST_DELAY)

    variation_ids = search_data.get("esearchresult", {}).get("idlist", [])
    if not variation_ids:
        logger.warning("No ClinVar variants found for gene %r", gene_symbol)
        return []

    summary_params = {
        "db": "clinvar",
        "id": ",".join(variation_ids),
        "retmode": "json",
        "api_key": NCBI_API_KEY,
    }

    try:
        response = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=summary_params, timeout=20)
        response.raise_for_status()
        summary_data = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("ClinVar summary fetch for gene %r failed: %s", gene_symbol, e)
        return []
    finally:
        time.sleep(REQUEST_DELAY)

    raw_path = RAW_DIR / f"search_{gene_symbol}.json"
    with open(raw_path, "w") as f:
        json.dump({"esearch": search_data, "esummary": summary_data}, f, indent=2)

    results = []
    summary_result = summary_data.get("result", {})
    for vid in summary_result.get("uids", []):
        record = summary_result.get(vid, {})
        germline = record.get("germline_classification") or {}
        classification = germline.get("description")
        if classification not in PATHOGENIC_CLASSIFICATIONS:
            continue

        genes = record.get("genes") or []
        results.append(
            {
                "variation_id": vid,
                "variant_name": record.get("title"),
                "gene_symbol": genes[0].get("symbol") if genes else gene_symbol,
                "clinical_significance": classification,
                "review_status": germline.get("review_status"),
            }
        )
        if len(results) >= limit:
            break

    logger.info(
        "ClinVar search for %r: %s total variants, %d pathogenic/likely pathogenic returned",
        gene_symbol,
        search_data.get("esearchresult", {}).get("count"),
        len(results),
    )
    return results


def fetch_variant_detail(variation_id, use_cache=True):
    """
    Fetch full classification detail for a single ClinVar variant.

    Uses the NCBI Entrez esummary endpoint, which for ClinVar returns the
    full classification record: clinical significance, associated
    condition(s) with cross-references (including OMIM where available),
    review status, molecular consequence, and gene information.

    The raw record is saved to data/raw/clinvar/variant_{variation_id}.json
    and cached at data/cache/clinvar_{variation_id}.json so the same
    variant is not re-fetched on later runs.

    Args:
        variation_id (str | int): ClinVar VariationID (the 'variation_id'
            returned by search_clinvar_by_gene).
        use_cache (bool): If True (default), return the cached raw record
            when one exists instead of calling the API.

    Returns:
        dict | None: The raw esummary record for this variant, or None if
            the request failed or the variant was not found.
    """
    variation_id = str(variation_id)
    cache_path = CACHE_DIR / f"clinvar_{variation_id}.json"

    if use_cache and cache_path.exists():
        logger.info("Loading ClinVar variant %s from cache", variation_id)
        with open(cache_path) as f:
            return json.load(f)

    params = {
        "db": "clinvar",
        "id": variation_id,
        "retmode": "json",
        "api_key": NCBI_API_KEY,
    }

    try:
        response = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("Failed to fetch ClinVar variant %s: %s", variation_id, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY)

    record = data.get("result", {}).get(variation_id)
    if not record:
        logger.warning("No ClinVar record found for variation ID %s", variation_id)
        return None

    raw_path = RAW_DIR / f"variant_{variation_id}.json"
    with open(raw_path, "w") as f:
        json.dump(record, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info("Fetched ClinVar variant %s", variation_id)
    return record


def normalize_clinvar_entry(raw_variant):
    """
    Convert a raw ClinVar esummary record into a clean, structured dict.

    Args:
        raw_variant (dict): A raw per-variant esummary record, as returned
            by fetch_variant_detail.

    Returns:
        dict | None: A dict with 'variation_id', 'gene_symbol',
            'variant_name', 'clinical_significance', 'condition_name',
            'review_status', 'molecular_consequence', and 'omim_cross_ref'
            (a list of "OMIM:<mim_number>" strings for linking back to
            OMIM disease nodes). Returns None if raw_variant is falsy.
    """
    if not raw_variant:
        return None

    genes = raw_variant.get("genes") or []
    gene_symbol = genes[0].get("symbol") if genes else None

    germline = raw_variant.get("germline_classification") or {}
    trait_set = germline.get("trait_set") or []

    condition_names = [trait.get("trait_name") for trait in trait_set if trait.get("trait_name")]
    molecular_consequences = raw_variant.get("molecular_consequence_list") or []

    omim_cross_ref = set()
    for trait in trait_set:
        for xref in trait.get("trait_xrefs") or []:
            if xref.get("db_source") == "OMIM" and xref.get("db_id"):
                omim_cross_ref.add(f"OMIM:{xref['db_id']}")

    return {
        "variation_id": raw_variant.get("uid"),
        "gene_symbol": gene_symbol,
        "variant_name": raw_variant.get("title"),
        "clinical_significance": germline.get("description"),
        "condition_name": "; ".join(condition_names) if condition_names else None,
        "review_status": germline.get("review_status"),
        "molecular_consequence": "; ".join(molecular_consequences) if molecular_consequences else None,
        "omim_cross_ref": sorted(omim_cross_ref),
    }


def batch_ingest_by_genes(gene_symbols, limit_per_gene=10):
    """
    Search, fetch, normalize, and save ClinVar pathogenic/likely pathogenic
    variants for a list of genes.

    For each gene, finds up to `limit_per_gene` pathogenic or likely
    pathogenic variants, fetches full detail for each, normalizes it, and
    writes the results to data/processed/clinvar/{gene_symbol}.json. A
    failure on any individual gene or variant is logged and skipped so it
    does not stop processing of the rest.

    Args:
        gene_symbols (list[str]): HGNC gene symbols, e.g. ["FBN1", "CFTR"].
        limit_per_gene (int): Maximum variants to process per gene.
            Defaults to 10.

    Returns:
        dict[str, list[dict]]: Maps each gene symbol to its list of
            normalized variant entries.
    """
    results_by_gene = {}

    for gene_symbol in gene_symbols:
        logger.info("Processing ClinVar variants for gene %s", gene_symbol)

        candidates = search_clinvar_by_gene(gene_symbol, limit=limit_per_gene)
        if not candidates:
            logger.error("Skipping gene %s: no candidate variants found", gene_symbol)
            results_by_gene[gene_symbol] = []
            continue

        normalized_variants = []
        for candidate in candidates:
            variation_id = candidate["variation_id"]

            raw_variant = fetch_variant_detail(variation_id)
            if raw_variant is None:
                logger.error("Skipping variant %s for gene %s: fetch failed", variation_id, gene_symbol)
                continue

            normalized = normalize_clinvar_entry(raw_variant)
            if normalized is None:
                continue
            normalized_variants.append(normalized)

        out_path = PROCESSED_DIR / f"{gene_symbol}.json"
        with open(out_path, "w") as f:
            json.dump(normalized_variants, f, indent=2)
        logger.info("Saved %d normalized ClinVar variants to %s", len(normalized_variants), out_path)

        results_by_gene[gene_symbol] = normalized_variants

    return results_by_gene


if __name__ == "__main__":
    test_genes = ["FBN1", "CFTR", "HTT", "GBA1", "DMD"]
    results = batch_ingest_by_genes(test_genes)
    for gene, variants in results.items():
        print(f"\n{gene}: {len(variants)} variants")
        for v in variants:
            print(f"   {v['variant_name']} | {v['clinical_significance']} | OMIM: {v['omim_cross_ref']}")
