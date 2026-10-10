# BioIQ

**Rare and chronic disease research intelligence.** Give BioIQ a disease name and it returns a structured, cited research brief: mechanism, genetic basis, pathway disruptions, drug repurposing candidates, active clinical trials, and open research questions.

Existing tools such as DisGeNET, Monarch, and Open Targets return association tables. BioIQ connects several public databases into one knowledge graph, then uses the Anthropic API to reason over that graph and write the synthesis in plain language.

## How it works

```
Public databases ──► ingestion/ ──► graph/ ──► synthesis/ ──► api/ ──► frontend/
 OMIM, ClinVar,       one module    entity       retrieval +    FastAPI   simple web UI
 Orphanet, ChEMBL,    per source    resolution   Claude API
 Reactome,                          + SQLite
 ClinicalTrials.gov                 graph
```

1. **Ingestion** (`ingestion/`): one module per source. Raw responses are cached, so re-running costs no new API calls.
   - OMIM: genes, mechanisms, clinical text
   - ClinVar: genetic variant evidence
   - Orphanet: rare disease registry
   - ChEMBL: drug and target data
   - Reactome: biological pathways
   - ClinicalTrials.gov: active trials
2. **Graph** (`graph/`): entity resolution maps the same disease or gene across sources (including symbol drift such as `GBA` → `GBA1`), and the builder assembles unified disease nodes into a SQLite database.
3. **Synthesis** (`synthesis/`): retrieves context from the graph, then calls the Claude API to write the cited brief and generate drug-repurposing hypotheses.
4. **API** (`api/`): FastAPI backend serving briefs, hypotheses, and cross-disease patterns.
5. **Frontend** (`frontend/`): a plain HTML/CSS/JS interface on top of the API.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status |
| `GET /diseases` | Diseases available in the graph |
| `POST /query` | Generate (or return cached) research brief for a disease |
| `POST /hypotheses` | Drug repurposing and mechanism hypotheses for a disease |
| `GET /patterns/{disease_name}` | Cross-disease drug, pathway, and mechanism overlaps |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is gitignored):

```
OMIM_API_KEY=...        # free, register at omim.org/api
NCBI_API_KEY=...        # free, for ClinVar / PubMed
ANTHROPIC_API_KEY=...   # for the synthesis layer
```

Orphanet, Reactome, ChEMBL, and ClinicalTrials.gov need no key.

## Usage

```bash
# 1. Ingest sources and build the graph (cached, safe to re-run)
python scripts/seed_diseases.py

# 2. Generate hypotheses for every disease in the graph (calls Claude, cached to disk)
python scripts/generate_all_hypotheses.py

# 3. Build cross-disease pattern analysis
python scripts/cross_disease_analysis.py --refresh

# 4. Start the API
uvicorn api.main:app --reload
```

To add diseases, add the name and OMIM MIM number to the `DISEASES` list in `scripts/seed_diseases.py` and re-run steps 1–3. Only the new diseases make fresh API calls.

## Known limitations

- ChEMBL misses enzyme replacement therapies (for example for Gaucher disease), because they replace the enzyme rather than pharmacologically targeting it.
- Large structural variants in ClinVar span multiple genes and carry no OMIM cross-reference, so only single-gene variants link reliably.
- Cached briefs, hypotheses, and cross-disease patterns are static snapshots with no automatic invalidation yet.

## Status

Working MVP: the full pipeline runs end to end across a seed corpus of rare and chronic diseases. Planned next steps are a PostgreSQL + pgvector backend and broader disease coverage. However, any disease can be given to BioIQ and the output will be the research brief (on-demand ingestion) but latency is quite high

## Tech stack

Python 3.10+, FastAPI, SQLite, Anthropic Claude API.
