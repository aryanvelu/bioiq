"""
Runs generate_brief for all 5 diseases and writes output to data/briefs_output.txt.
Each brief is flushed immediately so progress is visible in the output file.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis.engine import generate_brief

DISEASES = [
    "Marfan Syndrome",
    "Cystic Fibrosis",
    "Huntington Disease",
    "Gaucher Disease",
    "Duchenne Muscular Dystrophy",
]

OUT = "data/briefs_output.txt"

with open(OUT, "w") as f:
    for disease in DISEASES:
        header = f"\n{'='*80}\n  RESEARCH BRIEF: {disease.upper()}\n{'='*80}\n"
        f.write(header)
        f.flush()
        print(f"Generating brief for {disease}...", flush=True)
        brief = generate_brief(disease)
        f.write(brief + "\n")
        f.flush()
        char_count = len(brief)
        separator = f"\n--- {disease} brief: {char_count} chars ---\n"
        f.write(separator)
        f.flush()
        print(f"  Done: {char_count} chars written", flush=True)

print(f"\nAll 5 briefs written to {OUT}", flush=True)
