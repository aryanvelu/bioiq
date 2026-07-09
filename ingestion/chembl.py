"""
ChEMBL ingestion module.

Provides functions to search ChEMBL for approved drugs targeting a given
gene, fetch full drug details (molecule info, indications, and mechanisms),
normalize raw responses into knowledge-graph-ready records, and batch-process
a list of gene symbols into data/processed/chembl/.

Uses the ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data), which is
open, requires no authentication, and is free to use commercially.
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

REQUEST_DELAY = 0.5  # seconds between API calls, per project rate-limit policy

APPROVED_MAX_PHASE = 4  # ChEMBL max_phase 4 means an approved drug

SMALL_MOLECULE_TYPES = frozenset({"Small molecule", "Peptide", "Unknown", "Oligopeptide"})
RNA_THERAPY_TYPES = frozenset({"Oligonucleotide", "RNA"})
LARGE_MOLECULE_TYPES = frozenset({"Protein", "Enzyme", "Antibody", "Antibody fragment"})

RAW_DIR = Path("data/raw/chembl")
PROCESSED_DIR = Path("data/processed/chembl")
CACHE_DIR = Path("data/cache")

for _dir in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _get(url, params=None):
    """
    Perform a GET request against the ChEMBL API with a rate-limit delay.

    Args:
        url (str): Full URL to request.
        params (dict | None): Optional query parameters.

    Returns:
        dict | None: Parsed JSON response, or None if the request failed.
    """
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("ChEMBL request to %s failed: %s", url, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY)


def _fetch_target_genes(target_chembl_id):
    """
    Look up the gene symbols associated with a ChEMBL target.

    Results are cached at data/cache/chembl_target_{target_chembl_id}.json
    since the same target is often shared by multiple drugs.

    Args:
        target_chembl_id (str | None): ChEMBL target ID, e.g. "CHEMBL4051".

    Returns:
        list[str]: GENE_SYMBOL synonyms from the target's components, sorted
            and deduplicated. Returns an empty list if target_chembl_id is
            None or the lookup fails.
    """
    if not target_chembl_id:
        return []

    cache_path = CACHE_DIR / f"chembl_target_{target_chembl_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    target = _get(f"{BASE_URL}/target/{target_chembl_id}.json")
    if not target:
        return []

    genes = sorted(
        {
            syn.get("component_synonym")
            for component in target.get("target_components", [])
            for syn in component.get("target_component_synonyms", [])
            if syn.get("syn_type") == "GENE_SYMBOL"
        }
    )

    with open(cache_path, "w") as f:
        json.dump(genes, f, indent=2)

    return genes


def search_by_gene(gene_symbol):
    """
    Search ChEMBL for approved drugs whose mechanism targets the given gene.

    Finds the human single-protein ChEMBL target whose GENE_SYMBOL synonym
    matches gene_symbol, looks up its drug mechanisms, and keeps only those
    with max_phase == 4 (approved). The raw target and mechanism responses are
    saved to data/raw/chembl/search_{gene_symbol}.json.

    Args:
        gene_symbol (str): HGNC gene symbol, e.g. "CFTR".

    Returns:
        list[dict]: One dict per approved drug, each with 'chembl_id' (str),
            'name' (str, ChEMBL preferred name), 'target_chembl_id' (str),
            'action_type' (str), and 'mechanism_of_action' (str). Returns an
            empty list if no matching human target, or no approved drugs,
            are found for it.
    """
    target_data = _get(
        f"{BASE_URL}/target.json",
        params={"target_synonym__icontains": gene_symbol, "limit": 50},
    )
    if not target_data:
        return []

    target_chembl_id = None
    for target in target_data.get("targets", []):
        if target.get("target_type") != "SINGLE PROTEIN" or target.get("organism") != "Homo sapiens":
            continue
        gene_symbols = {
            syn.get("component_synonym", "").upper()
            for component in target.get("target_components", [])
            for syn in component.get("target_component_synonyms", [])
            if syn.get("syn_type") == "GENE_SYMBOL"
        }
        if gene_symbol.upper() in gene_symbols:
            target_chembl_id = target["target_chembl_id"]
            break

    if target_chembl_id is None:
        logger.warning("No human single-protein ChEMBL target found for gene %r", gene_symbol)
        return []

    mechanism_data = _get(
        f"{BASE_URL}/mechanism.json",
        params={"target_chembl_id": target_chembl_id, "limit": 100},
    )
    if not mechanism_data:
        return []

    approved = [m for m in mechanism_data.get("mechanisms", []) if m.get("max_phase") == APPROVED_MAX_PHASE]
    if not approved:
        logger.info("No approved drugs found targeting %s (%s)", gene_symbol, target_chembl_id)
        return []

    molecule_ids = sorted({m["molecule_chembl_id"] for m in approved})
    molecule_data = _get(
        f"{BASE_URL}/molecule.json",
        params={"molecule_chembl_id__in": ",".join(molecule_ids), "limit": len(molecule_ids)},
    )
    names = {m["molecule_chembl_id"]: m.get("pref_name") for m in (molecule_data or {}).get("molecules", [])}

    raw_path = RAW_DIR / f"search_{gene_symbol}.json"
    with open(raw_path, "w") as f:
        json.dump(
            {"target_chembl_id": target_chembl_id, "mechanisms": approved, "molecules": (molecule_data or {}).get("molecules", [])},
            f,
            indent=2,
        )

    return [
        {
            "chembl_id": m["molecule_chembl_id"],
            "name": names.get(m["molecule_chembl_id"]),
            "target_chembl_id": target_chembl_id,
            "action_type": m.get("action_type"),
            "mechanism_of_action": m.get("mechanism_of_action"),
        }
        for m in approved
    ]


def fetch_drug_detail(chembl_id, use_cache=True):
    """
    Fetch full ChEMBL drug detail for a molecule.

    Combines three ChEMBL endpoints: molecule (name, max phase, type, first
    approval), drug indication (approved and investigational indications),
    and mechanism (targets and mechanism of action, each annotated with the
    gene symbols of its target). The combined raw response is saved to
    data/raw/chembl/{chembl_id}.json and cached at
    data/cache/chembl_{chembl_id}.json.

    Args:
        chembl_id (str): ChEMBL molecule ID, e.g. "CHEMBL2010601".
        use_cache (bool): If True (default), return the cached raw response
            when one exists instead of calling the API.

    Returns:
        dict | None: A dict with 'molecule', 'indications' (list), and
            'mechanisms' (list, each with a 'target_genes' list added).
            Returns None if the molecule lookup fails.
    """
    cache_path = CACHE_DIR / f"chembl_{chembl_id}.json"

    if use_cache and cache_path.exists():
        logger.info("Loading ChEMBL entry %s from cache", chembl_id)
        with open(cache_path) as f:
            return json.load(f)

    molecule = _get(f"{BASE_URL}/molecule/{chembl_id}.json")
    if molecule is None:
        logger.error("No ChEMBL molecule found for %s", chembl_id)
        return None

    indication_data = _get(f"{BASE_URL}/drug_indication.json", params={"molecule_chembl_id": chembl_id, "limit": 100})
    indications = (indication_data or {}).get("drug_indications", [])

    mechanism_data = _get(f"{BASE_URL}/mechanism.json", params={"molecule_chembl_id": chembl_id, "limit": 100})
    mechanisms = (mechanism_data or {}).get("mechanisms", [])
    for mechanism in mechanisms:
        mechanism["target_genes"] = _fetch_target_genes(mechanism.get("target_chembl_id"))

    entry = {"molecule": molecule, "indications": indications, "mechanisms": mechanisms}

    raw_path = RAW_DIR / f"{chembl_id}.json"
    with open(raw_path, "w") as f:
        json.dump(entry, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(entry, f, indent=2)

    logger.info("Fetched ChEMBL entry %s (%s)", chembl_id, molecule.get("pref_name"))
    return entry


def normalize_chembl_entry(raw_entry, drug_category="disease_modifying"):
    """
    Convert a raw ChEMBL drug entry into a clean, structured knowledge-graph
    node.

    Args:
        raw_entry (dict): The raw entry dict returned by fetch_drug_detail,
            with 'molecule', 'indications', and 'mechanisms' keys.
        drug_category (str): One of disease_modifying | symptomatic | off_label |
            enzyme_replacement | biologic | rna_aso.  Defaults to disease_modifying
            (used for gene-target mechanism searches).

    Returns:
        dict | None: A dict with 'id' ("CHEMBL:<chembl_id>"), 'source',
            'chembl_id', 'name', 'indication', 'mechanism', 'target_genes',
            and 'drug_category'.  Returns None if raw_entry has no molecule data.
    """
    molecule = (raw_entry or {}).get("molecule")
    if not molecule:
        return None

    indications = raw_entry.get("indications") or []
    approved_terms = {ind.get("efo_term") for ind in indications if ind.get("max_phase_for_ind") == "4.0"}
    all_terms = {ind.get("efo_term") for ind in indications}
    indication = sorted(filter(None, approved_terms)) or sorted(filter(None, all_terms))

    mechanisms = raw_entry.get("mechanisms") or []
    mechanism = sorted(filter(None, {m.get("mechanism_of_action") for m in mechanisms}))
    target_genes = sorted({gene for m in mechanisms for gene in m.get("target_genes") or []})

    chembl_id = molecule.get("molecule_chembl_id")
    return {
        "id": f"CHEMBL:{chembl_id}",
        "source": "ChEMBL",
        "chembl_id": chembl_id,
        "name": molecule.get("pref_name"),
        "indication": indication,
        "mechanism": mechanism,
        "target_genes": target_genes,
        "drug_category": drug_category,
    }


def batch_ingest_by_genes(gene_symbols):
    """
    Search, fetch, normalize, and save ChEMBL entries for approved drugs
    targeting a list of genes.

    For each gene symbol, finds approved drugs whose mechanism targets that
    gene, fetches the full ChEMBL entry for each drug, normalizes it, and
    writes the result to data/processed/chembl/{chembl_id}.json. A drug that
    targets more than one of the given genes is only fetched and saved once.
    A failure on any individual gene or drug is logged and skipped so it does
    not stop processing of the rest.

    Args:
        gene_symbols (list[str]): HGNC gene symbols to search, e.g.
            ["CFTR", "HTT"].

    Returns:
        dict[str, list[dict]]: Maps each gene symbol to the list of
            normalized drug nodes found for it (possibly empty).
    """
    results = {}
    seen_nodes = {}

    for gene_symbol in gene_symbols:
        logger.info("Searching ChEMBL for approved drugs targeting %r", gene_symbol)

        matches = search_by_gene(gene_symbol)
        if not matches:
            results[gene_symbol] = []
            continue

        nodes = []
        for match in matches:
            chembl_id = match["chembl_id"]

            if chembl_id in seen_nodes:
                nodes.append(seen_nodes[chembl_id])
                continue

            raw_entry = fetch_drug_detail(chembl_id)
            if raw_entry is None:
                logger.error("Skipping %s: fetch failed", chembl_id)
                continue

            node = normalize_chembl_entry(raw_entry, drug_category="disease_modifying")
            if node is None:
                logger.error("Skipping %s: could not normalize entry", chembl_id)
                continue

            out_path = PROCESSED_DIR / f"{chembl_id}.json"
            with open(out_path, "w") as f:
                json.dump(node, f, indent=2)
            logger.info("Saved normalized ChEMBL node to %s", out_path)

            seen_nodes[chembl_id] = node
            nodes.append(node)

        results[gene_symbol] = nodes

    return results


def search_drugs_by_indication(disease_name):
    """
    Search ChEMBL by disease indication for small-molecule and oligonucleotide drugs.

    Queries the drug_indication endpoint by EFO term and returns approved
    small molecules tagged symptomatic or off_label, and approved/late-stage
    oligonucleotides tagged rna_aso.  Large-molecule biologics are excluded
    (use search_drugs_by_indication_name for those).

    Saves individual drug detail files to data/processed/chembl/ and the raw
    indication query to data/raw/chembl/indication_{safe_name}.json.

    Args:
        disease_name (str): ChEMBL-style disease name, e.g. "Huntington disease".

    Returns:
        list[dict]: Normalized drug nodes with drug_category set.
    """
    data = _get(f"{BASE_URL}/drug_indication.json",
                params={"efo_term__icontains": disease_name, "limit": 200})
    if not data:
        return []

    mol_phases = {}
    for ind in data.get("drug_indications", []):
        cid = ind.get("molecule_chembl_id")
        try:
            phase = float(ind.get("max_phase_for_ind") or 0)
        except (TypeError, ValueError):
            phase = 0.0
        if cid and phase > mol_phases.get(cid, 0):
            mol_phases[cid] = phase

    if not mol_phases:
        return []

    ids = sorted(mol_phases.keys())
    molecules = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        mol_data = _get(f"{BASE_URL}/molecule.json",
                        params={"molecule_chembl_id__in": ",".join(chunk), "limit": len(chunk)})
        molecules.extend((mol_data or {}).get("molecules", []))

    safe_query = re.sub(r"[^\w]", "_", disease_name)
    raw_path = RAW_DIR / f"indication_{safe_query}.json"
    with open(raw_path, "w") as f:
        json.dump({"drug_indications": data.get("drug_indications", []),
                   "molecules": molecules}, f, indent=2)

    nodes = []
    for mol in molecules:
        cid = mol.get("molecule_chembl_id")
        mol_type = mol.get("molecule_type", "")
        try:
            mol_max_phase = float(mol.get("max_phase") or 0)
        except (TypeError, ValueError):
            mol_max_phase = 0.0
        ind_phase = mol_phases.get(cid, 0)

        if mol_type in SMALL_MOLECULE_TYPES:
            if mol_max_phase < 4:
                continue  # only globally-approved small molecules
            category = "symptomatic" if ind_phase >= 4.0 else "off_label"
        elif mol_type in RNA_THERAPY_TYPES:
            if mol_max_phase < 3:
                continue  # skip very early-stage oligonucleotides
            category = "rna_aso"
        else:
            continue  # large molecules handled by search_drugs_by_indication_name

        raw_entry = fetch_drug_detail(cid)
        if not raw_entry:
            continue

        node = normalize_chembl_entry(raw_entry, drug_category=category)
        if not node:
            continue

        out_path = PROCESSED_DIR / f"{cid}.json"
        with open(out_path, "w") as f:
            json.dump(node, f, indent=2)
        logger.info("Saved %s drug %s (%s)", category, cid, node.get("name"))
        nodes.append(node)

    logger.info("search_drugs_by_indication(%r): %d nodes", disease_name, len(nodes))
    return nodes


def search_drugs_by_indication_name(disease_name):
    """
    Search ChEMBL by disease indication for large-molecule drugs.

    Returns enzyme_replacement (Protein/Enzyme molecules) and biologic
    (Antibody/Antibody fragment) entries.  Only includes globally-approved
    molecules (max_phase == 4).

    Args:
        disease_name (str): ChEMBL-style disease name.

    Returns:
        list[dict]: Normalized drug nodes with drug_category set.
    """
    data = _get(f"{BASE_URL}/drug_indication.json",
                params={"efo_term__icontains": disease_name, "limit": 200})
    if not data:
        return []

    mol_phases = {}
    for ind in data.get("drug_indications", []):
        cid = ind.get("molecule_chembl_id")
        try:
            phase = float(ind.get("max_phase_for_ind") or 0)
        except (TypeError, ValueError):
            phase = 0.0
        if cid and phase > mol_phases.get(cid, 0):
            mol_phases[cid] = phase

    if not mol_phases:
        return []

    ids = sorted(mol_phases.keys())
    molecules = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        mol_data = _get(f"{BASE_URL}/molecule.json",
                        params={"molecule_chembl_id__in": ",".join(chunk), "limit": len(chunk)})
        molecules.extend((mol_data or {}).get("molecules", []))

    nodes = []
    for mol in molecules:
        cid = mol.get("molecule_chembl_id")
        mol_type = mol.get("molecule_type", "")
        try:
            mol_max_phase = float(mol.get("max_phase") or 0)
        except (TypeError, ValueError):
            mol_max_phase = 0.0

        if mol_type not in LARGE_MOLECULE_TYPES:
            continue
        if mol_max_phase < 4:
            continue  # only approved biologics

        category = "enzyme_replacement" if mol_type in {"Protein", "Enzyme"} else "biologic"

        raw_entry = fetch_drug_detail(cid)
        if not raw_entry:
            continue

        node = normalize_chembl_entry(raw_entry, drug_category=category)
        if not node:
            continue

        out_path = PROCESSED_DIR / f"{cid}.json"
        with open(out_path, "w") as f:
            json.dump(node, f, indent=2)
        logger.info("Saved %s drug %s (%s)", category, cid, node.get("name"))
        nodes.append(node)

    logger.info("search_drugs_by_indication_name(%r): %d nodes", disease_name, len(nodes))
    return nodes


def batch_ingest_by_indication(disease_name_to_query):
    """
    Run indication-based ingestion for multiple diseases.

    For each disease calls search_drugs_by_indication (small molecules + rna_aso)
    and search_drugs_by_indication_name (enzyme replacement + biologics), combines
    the results, and saves the full list to
    data/processed/chembl/indication_{safe_disease_name}.json.

    The builder reads this file to populate the drugs table with categories beyond
    disease_modifying.

    Args:
        disease_name_to_query (dict[str, str]): Maps canonical disease name to the
            ChEMBL indication query string, e.g.
            {"Gaucher Disease": "Gaucher disease"}.

    Returns:
        dict[str, list[dict]]: Maps each canonical disease name to the list of
            normalized drug nodes found for it.
    """
    results = {}
    for disease_name, query in disease_name_to_query.items():
        logger.info("Indication ingest for %r (ChEMBL query=%r)", disease_name, query)

        small_nodes = search_drugs_by_indication(query)
        large_nodes = search_drugs_by_indication_name(query)
        all_nodes = small_nodes + large_nodes

        safe_name = re.sub(r"[^\w]", "_", disease_name)
        out_path = PROCESSED_DIR / f"indication_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(all_nodes, f, indent=2)

        logger.info(
            "  %r: %d small/rna_aso + %d biologic/ERT = %d total — saved to %s",
            disease_name, len(small_nodes), len(large_nodes), len(all_nodes), out_path,
        )
        results[disease_name] = all_nodes

    return results


if __name__ == "__main__":
    test_genes = ["FBN1", "CFTR", "HTT", "GBA1", "DMD"]
    results = batch_ingest_by_genes(test_genes)
    for gene_symbol, nodes in results.items():
        if not nodes:
            print(f"{gene_symbol}: no approved drugs found targeting this gene")
            continue
        for node in nodes:
            print(
                f"{gene_symbol} -> {node['id']} | {node['name']} | "
                f"mechanism: {node['mechanism']} | indication: {node['indication']} | "
                f"target_genes: {node['target_genes']}"
            )
