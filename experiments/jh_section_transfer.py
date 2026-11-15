"""Offline retrieval diagnostics on previously generated public paraphrases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.kb import load_chunks
from core.search import Searcher
from pipelines.kimjaehak.learned_sections import train_section_classifier
from pipelines.kimjaehak.kb_expanded import ExpandedIndex


def run(source):
    provenance = json.loads((source.parent / "derivation.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if provenance["scope"] != "public_paraphrase_input_diagnostic" or digest != provenance["dataset_sha256"]:
        raise ValueError("Derived input provenance mismatch")
    public = ROOT / "public/tutor.jsonl"
    originals = {row["id"]:row for row in map(json.loads,public.read_text().splitlines())}
    rows = [row for row in map(json.loads,source.read_text().splitlines()) if row["source_id"] in originals
            and len(originals[row["source_id"]]["labels"].get("evidence") or []) == 1]
    searcher = Searcher(load_chunks(ROOT / "kb"))
    questions = ROOT / "pipelines/kimjaehak/data/kb_questions.jsonl"
    expanded_index = ExpandedIndex(searcher,list(map(json.loads,questions.read_text().splitlines())))
    vectorizer,model = train_section_classifier(1000)
    probabilities = model.predict_proba(vectorizer.transform([row["messages"][-1]["content"] for row in rows]))
    predicted,confidence = model.classes_[probabilities.argmax(axis=1)],probabilities.max(axis=1)
    gold = [originals[row["source_id"]]["labels"]["evidence"][0] for row in rows]
    baseline,expanded = [],[]
    for row in rows:
        query = "\n".join(message["content"] for message in row["messages"])
        found = searcher.search(query)
        baseline.append(found[0].section_id if found else "")
        found = expanded_index.best_chunk(query,.1)
        expanded.append(found.section_id if found else "")
    result = {"scope":"public_derived_section_transfer","independent_source_holdout":False,
        "partition":provenance.get("paraphrase_partition","primary"),"count":len(rows),"source_sha256":digest,
        "training_source_sha256":hashlib.sha256(public.read_bytes()).hexdigest(),
        "bm25_hits":sum(a==b for a,b in zip(baseline,gold)),"expanded_hits":sum(a==b for a,b in zip(expanded,gold)),"gates":{}}
    for threshold in (.95,.98,.995):
        selected = [i for i in range(len(rows)) if confidence[i]>=threshold]
        result["gates"][str(threshold)] = {"coverage":len(selected),
            "selected_hits":sum(predicted[i]==gold[i] for i in selected),
            "combined_hits":sum((predicted[i] if confidence[i]>=threshold else baseline[i])==gold[i] for i in range(len(rows)))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        parser.error("Use recorded public-derived inputs")
    result = run(source)
    (source.parent / "section_diagnostic.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
