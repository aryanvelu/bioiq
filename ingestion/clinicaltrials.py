"""
ClinicalTrials.gov ingestion module.

Provides functions to search ClinicalTrials.gov for interventional trials by
disease name, fetch full trial detail (interventions, eligibility criteria,
primary endpoints), normalize raw responses into knowledge-graph-ready records,
and batch-process a list of disease names into data/processed/clinicaltrials/.

Uses the ClinicalTrials.gov REST API v2 (https://clinicaltrials.gov/api/v2/),
which is open, requires no authentication, and is free to use commercially.
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://clinicaltrials.gov/api/v2"

REQUEST_DELAY = 0.5  # seconds between API calls, per project rate-limit policy

# Active trial statuses that represent current or imminent clinical activity
ACTIVE_STATUSES = ",".join([
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
])

# Intervention-only trials (excludes observational studies)
INTERVENTIONAL_FILTER = "studyType:int"

MAX_RESULTS_PER_DISEASE = 200  # hard cap to keep batch runs bounded

# Human-readable phase labels for normalized output
PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "N/A",
}

RAW_DIR = Path("data/raw/clinicaltrials")
PROCESSED_DIR = Path("data/processed/clinicaltrials")
CACHE_DIR = Path("data/cache")

for _dir in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _get(url, params=None):
    """
    Perform a GET request against the ClinicalTrials.gov v2 API with a
    rate-limit delay.

    Args:
        url (str): Full URL to request.
        params (dict | None): Optional query parameters.

    Returns:
        dict | None: Parsed JSON response, or None if the request failed.
    """
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("ClinicalTrials request to %s failed: %s", url, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY)


def _parse_eligibility(criteria_text):
    """
    Split a raw eligibilityCriteria string into inclusion and exclusion
    criteria.

    The text follows a standard format where "Exclusion Criteria:" marks the
    boundary. Content before it is inclusion criteria; content after is
    exclusion criteria.

    Args:
        criteria_text (str): Raw eligibilityCriteria field value.

    Returns:
        tuple[str, str]: (inclusion_text, exclusion_text). Either may be an
            empty string if the respective section is absent.
    """
    if not criteria_text:
        return "", ""
    parts = re.split(r"(?i)exclusion\s+criteria[:\s]*", criteria_text, maxsplit=1)
    inclusion = re.sub(r"(?i)^key?\s*inclusion\s+criteria[:\s]*", "", parts[0]).strip()
    exclusion = parts[1].strip() if len(parts) > 1 else ""
    return inclusion, exclusion


def search_trials_by_disease(disease_name, max_results=MAX_RESULTS_PER_DISEASE):
    """
    Search ClinicalTrials.gov for active interventional trials for a disease.

    Returns trials in statuses RECRUITING, ACTIVE_NOT_RECRUITING,
    ENROLLING_BY_INVITATION, and NOT_YET_RECRUITING, sorted by start date
    (most recent first). Each full study JSON from the response is cached at
    data/cache/clinicaltrials_{nctId}.json so that fetch_trial_detail can
    serve it without a separate API call. The raw search response is saved
    to data/raw/clinicaltrials/search_{safe_name}.json.

    Args:
        disease_name (str): Disease name to search for, e.g.
            "Cystic Fibrosis".
        max_results (int): Maximum number of trials to return. Defaults to
            MAX_RESULTS_PER_DISEASE (200).

    Returns:
        list[dict]: One dict per trial, each with 'nct_id', 'title',
            'status', 'phase', 'sponsor', and 'start_date'. Returns an empty
            list if no active interventional trials are found.
    """
    all_studies = []
    page_token = None
    base_params = {
        "query.cond": disease_name,
        "aggFilters": INTERVENTIONAL_FILTER,
        "filter.overallStatus": ACTIVE_STATUSES,
        "sort": "StartDate:desc",
        "countTotal": "true",
    }

    while len(all_studies) < max_results:
        page_size = min(200, max_results - len(all_studies))
        params = {**base_params, "pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token

        data = _get(f"{BASE_URL}/studies", params=params)
        if not data:
            break

        batch = data.get("studies", [])
        if not batch:
            break

        if not all_studies:
            logger.info(
                "Found %s active interventional trials for %r",
                data.get("totalCount", "?"),
                disease_name,
            )

        all_studies.extend(batch)
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    if not all_studies:
        logger.info("No active interventional trials found for %r", disease_name)
        return []

    # Cache each study so fetch_trial_detail can serve it without re-fetching
    for study in all_studies:
        ps = study.get("protocolSection", {})
        nct_id = ps.get("identificationModule", {}).get("nctId")
        if nct_id:
            cache_path = CACHE_DIR / f"clinicaltrials_{nct_id}.json"
            with open(cache_path, "w") as f:
                json.dump(study, f, indent=2)

    safe_name = re.sub(r"[^\w-]", "_", disease_name)
    raw_path = RAW_DIR / f"search_{safe_name}.json"
    with open(raw_path, "w") as f:
        json.dump(all_studies, f, indent=2)
    logger.info("Saved %d raw search results for %r to %s", len(all_studies), disease_name, raw_path)

    summaries = []
    for study in all_studies:
        ps = study.get("protocolSection", {})
        im = ps.get("identificationModule", {})
        sm = ps.get("statusModule", {})
        dm = ps.get("designModule", {})
        sc = ps.get("sponsorCollaboratorsModule", {})
        raw_phases = dm.get("phases") or []
        phase = "/".join(PHASE_LABELS.get(p, p) for p in raw_phases) or "N/A"
        summaries.append({
            "nct_id": im.get("nctId"),
            "title": im.get("briefTitle"),
            "status": sm.get("overallStatus"),
            "phase": phase,
            "sponsor": sc.get("leadSponsor", {}).get("name"),
            "start_date": sm.get("startDateStruct", {}).get("date"),
        })

    return summaries


def fetch_trial_detail(nct_id, use_cache=True):
    """
    Fetch full ClinicalTrials.gov detail for a trial.

    Returns the complete study JSON including all protocol sections. If the
    trial was included in a prior search_trials_by_disease call, the result
    is served from cache without an additional API call. The raw response is
    saved to data/raw/clinicaltrials/{nct_id}.json and cached at
    data/cache/clinicaltrials_{nct_id}.json.

    Args:
        nct_id (str): ClinicalTrials.gov identifier, e.g. "NCT05844449".
        use_cache (bool): If True (default), return the cached response when
            one exists instead of calling the API.

    Returns:
        dict | None: The full study JSON dict, or None if the trial does not
            exist.
    """
    cache_path = CACHE_DIR / f"clinicaltrials_{nct_id}.json"

    if use_cache and cache_path.exists():
        logger.info("Loading trial %s from cache", nct_id)
        with open(cache_path) as f:
            return json.load(f)

    data = _get(f"{BASE_URL}/studies/{nct_id}")
    if data is None:
        logger.error("No ClinicalTrials.gov entry found for %s", nct_id)
        return None

    raw_path = RAW_DIR / f"{nct_id}.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Fetched trial %s", nct_id)
    return data


def normalize_trial_entry(raw_entry):
    """
    Convert a raw ClinicalTrials.gov study JSON into a clean, structured
    knowledge-graph node.

    Args:
        raw_entry (dict): The full study JSON returned by fetch_trial_detail.

    Returns:
        dict | None: A dict with 'nct_id', 'title', 'status', 'phase',
            'sponsor', 'enrollment' (int or None), 'intervention_name'
            (list of intervention names), 'intervention_type' (list of
            unique intervention type strings), 'conditions' (list of
            condition strings), 'inclusion_criteria' (str),
            'exclusion_criteria' (str), and 'primary_endpoints' (list of
            primary outcome measure strings). Returns None if raw_entry has
            no protocolSection.
    """
    ps = (raw_entry or {}).get("protocolSection")
    if not ps:
        return None

    im = ps.get("identificationModule", {})
    sm = ps.get("statusModule", {})
    dm = ps.get("designModule", {})
    sc = ps.get("sponsorCollaboratorsModule", {})
    ai = ps.get("armsInterventionsModule", {})
    el = ps.get("eligibilityModule", {})
    om = ps.get("outcomesModule", {})
    cm = ps.get("conditionsModule", {})

    raw_phases = dm.get("phases") or []
    phase = "/".join(PHASE_LABELS.get(p, p) for p in raw_phases) or "N/A"

    enrollment_info = dm.get("enrollmentInfo") or {}
    enrollment = enrollment_info.get("count")

    interventions = ai.get("interventions") or []
    intervention_names = [inv.get("name") for inv in interventions if inv.get("name")]
    intervention_types = sorted({inv.get("type") for inv in interventions if inv.get("type")})

    inclusion, exclusion = _parse_eligibility(el.get("eligibilityCriteria", ""))

    primary_endpoints = [
        o.get("measure")
        for o in om.get("primaryOutcomes") or []
        if o.get("measure")
    ]

    return {
        "nct_id": im.get("nctId"),
        "title": im.get("briefTitle"),
        "status": sm.get("overallStatus"),
        "phase": phase,
        "sponsor": sc.get("leadSponsor", {}).get("name"),
        "enrollment": enrollment,
        "intervention_name": intervention_names,
        "intervention_type": intervention_types,
        "conditions": cm.get("conditions") or [],
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "primary_endpoints": primary_endpoints,
    }


def batch_ingest_by_disease(disease_names, max_results=MAX_RESULTS_PER_DISEASE):
    """
    Search, fetch, normalize, and save ClinicalTrials.gov entries for a list
    of diseases.

    For each disease, retrieves active interventional trials, normalizes each
    trial entry, and writes the result to
    data/processed/clinicaltrials/{nct_id}.json. Search results pre-populate
    the cache, so fetch_trial_detail calls within the same run do not make
    additional API requests. A trial that appears under more than one of the
    given diseases (e.g. a multi-indication study) is only saved once. A
    failure on any individual trial is logged and skipped without stopping
    the rest.

    Args:
        disease_names (list[str]): Disease names to process, e.g.
            ["Cystic Fibrosis", "Huntington Disease"].
        max_results (int): Maximum trials to retrieve per disease. Defaults
            to MAX_RESULTS_PER_DISEASE (200).

    Returns:
        dict[str, list[dict]]: Maps each disease name to the list of
            normalized trial nodes ingested for it (possibly empty).
    """
    results = {}
    seen_nct_ids = set()

    for disease_name in disease_names:
        logger.info("Ingesting ClinicalTrials.gov data for %r", disease_name)

        summaries = search_trials_by_disease(disease_name, max_results=max_results)
        if not summaries:
            results[disease_name] = []
            continue

        nodes = []
        for summary in summaries:
            nct_id = summary["nct_id"]
            if not nct_id:
                continue

            raw_entry = fetch_trial_detail(nct_id)
            if raw_entry is None:
                logger.error("Skipping %s: fetch failed", nct_id)
                continue

            if nct_id in seen_nct_ids:
                # Already written to disk; skip normalize+save but still count
                nodes.append(summary)
                continue

            node = normalize_trial_entry(raw_entry)
            if node is None:
                logger.error("Skipping %s: could not normalize entry", nct_id)
                continue

            out_path = PROCESSED_DIR / f"{nct_id}.json"
            with open(out_path, "w") as f:
                json.dump(node, f, indent=2)

            seen_nct_ids.add(nct_id)
            nodes.append(node)

        logger.info(
            "Ingested %d trials for %r", len(nodes), disease_name
        )
        results[disease_name] = nodes

    return results


if __name__ == "__main__":
    test_diseases = [
        "Marfan Syndrome",
        "Cystic Fibrosis",
        "Huntington Disease",
        "Gaucher Disease",
        "Duchenne Muscular Dystrophy",
    ]
    results = batch_ingest_by_disease(test_diseases)

    for disease_name, nodes in results.items():
        if not nodes:
            print(f"\n{disease_name}: no active interventional trials found")
            continue
        print(f"\n{disease_name}: {len(nodes)} trial(s)")
        for node in nodes[:5]:
            phase = node.get("phase") if isinstance(node, dict) and "phase" in node else node.get("phase", "?")
            status = node.get("status", "?")
            title = node.get("title", "?")
            sponsor = node.get("sponsor", "?")
            endpoints = node.get("primary_endpoints", [])
            print(f"  [{status}] {node.get('nct_id')} | {phase} | {sponsor}")
            print(f"    {title}")
            if endpoints:
                print(f"    Endpoints: {endpoints[0][:80]}" + ("..." if len(endpoints[0]) > 80 else ""))
        if len(nodes) > 5:
            print(f"  ... and {len(nodes)-5} more")
