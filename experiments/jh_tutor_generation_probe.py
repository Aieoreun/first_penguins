"""Freeze own tutor routing/source predictions to isolate answer generation.

Only bundled-public or provenance-checked local public-derived diagnostics are
accepted. Gold categories/evidence never enter generation; judge runs afterward.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jh_compare import evaluator_hashes, load_public, score_rows
from core.kb import load_chunks
from core.llm import LLM
from core.schema import FALLBACK_ANSWER, Label
from core.search import Searcher
from pipelines import load


class CaptureChoices:
    """Development-only recording of our own public draft candidates."""
    def __init__(self, llm, selection_system):
        self.llm, self.selection_system = llm, selection_system
        self.answers = []

    async def chat(self, system, user, **kwargs):
        if self.selection_system and system == self.selection_system:
            self.answers = json.loads(user)["후보"]
        return await self.llm.chat(system, user, **kwargs)


def source_data(source):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["evaluator_hashes"] != evaluator_hashes() or manifest["model"] != os.environ.get("OPENAI_MODEL"):
        raise ValueError("Source model/evaluator differs")
    if manifest.get("scope") in {"public_paraphrase_full_diagnostic", "provided_kb_fact_diagnostic"}:
        dataset = source.parent / "derived_inputs.jsonl"
        provenance = json.loads((source.parent / "derivation.json").read_text())
        if hashlib.sha256(dataset.read_bytes()).hexdigest() != provenance["dataset_sha256"]:
            raise ValueError("Derived dataset changed")
        items = list(map(json.loads, dataset.read_text().splitlines()))
        result_name = "derived_result.json"
    elif not manifest.get("scope"):
        items = load_public(endpoints=manifest.get("endpoints", ["tutor", "guard"]))
        result_name = "result.json"
    else:
        raise ValueError("Unsupported source scope")
    records = list(map(json.loads, (source / "responses.jsonl").read_text().splitlines()))
    prior = {r["id"]: r for r in records}
    if len(items) != manifest["count"] or len(records) != len(items) or set(prior) != {i["id"] for i in items}:
        raise ValueError("Incomplete or duplicate source items")
    return manifest, [i for i in items if i["endpoint"] == "tutor"], prior, json.loads((source/result_name).read_text())


def eligible(response, allowed):
    citations = response.get("citations")
    return (response.get("action") == "answer" and isinstance(citations, list)
            and len(citations) == 1 and citations[0] in allowed)


def section_changes(items, prior, vectorizer, model, allowed):
    """Choose interventions from own predictions; diagnostic gold is never read."""
    changes = {}
    for item in items:
        old = prior[item['id']]
        response = old.get('response') or {}
        if old.get('error') or not eligible(response, allowed):
            continue
        query = next(m['content'] for m in reversed(item['messages']) if m['role']=='user')
        probabilities = model.predict_proba(vectorizer.transform([query]))[0]
        index = probabilities.argmax()
        predicted = str(model.classes_[index])
        if (.90 <= probabilities[index] < .98 and predicted in allowed
                and predicted != response['citations'][0]):
            changes[item['id']] = {'old':response['citations'][0], 'new':predicted,
                                   'confidence':float(probabilities[index])}
    return changes


async def collect(items, prior, variant, workers, dest, source_selection='frozen', changed_sources=None):
    cls = load(variant)
    # No classification or retraining is needed: predictions come from the
    # source's own responses, not from the evaluator's labels.
    pipeline = cls.__new__(cls)
    pipeline.llm = LLM()
    pipeline.searcher = Searcher(load_chunks(ROOT / "kb"))
    allowed = set(pipeline.searcher.section_ids())
    slots = asyncio.Semaphore(workers)
    completed = 0
    with (dest / "responses.jsonl").open("w") as stream:
        async def one(item):
            nonlocal completed
            old = prior[item["id"]]
            row = copy.deepcopy(old)
            response = row.get("response") or {}
            row.update(generation_attempted=False, generation_errors=[], generation_debug={}, seconds=0,
                       source_seconds=old["seconds"])
            selected = source_selection == 'frozen' or item['id'] in (changed_sources or {})
            if not old["error"] and eligible(response, allowed) and selected:
                async with slots:
                    current = copy.copy(pipeline)
                    current.llm = CaptureChoices(pipeline.llm, getattr(pipeline, 'answer_choice_system', None))
                    state = current._new_state({"messages": item["messages"]}, False)
                    # NORMAL/PII use the same generation prompt in these variants.
                    state.label, state.section = Label.NORMAL, response["citations"][0]
                    if source_selection != 'frozen':
                        row['source_change_prediction'] = changed_sources[item['id']]
                        if source_selection == 'section90_changed':
                            state.section = changed_sources[item['id']]['new']
                    row["generation_attempted"] = True
                    start = time.monotonic()
                    try:
                        await asyncio.wait_for(current._generate(state), current.deadline)
                    except TimeoutError:
                        state.answer, state.citations = FALLBACK_ANSWER, []
                        state.errors.append("generation_probe:Deadline")
                    row.update(response=state.to_tutor_response(), generation_errors=state.errors,
                               generation_debug=state.debug, seconds=round(time.monotonic()-start, 3),
                               generation_candidates=current.llm.answers or [state.answer])
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
            completed += 1
            if completed % 25 == 0 or completed == len(items):
                print(f"generated {completed}/{len(items)}", flush=True)
            return row
        return await asyncio.gather(*(one(i) for i in items))


def score(dest):
    meta = json.loads((dest / "manifest.json").read_text())
    if (dest / "result.json").exists():
        raise ValueError("Already scored")
    source = ROOT / meta["source"]
    if hashlib.sha256((source / "responses.jsonl").read_bytes()).hexdigest() != meta["source_response_sha256"]:
        raise ValueError("Source response changed")
    _, items, prior, _ = source_data(source)
    rows = list(map(json.loads, (dest / "responses.jsonl").read_text().splitlines()))
    if len(rows) != len(items) or {r['id'] for r in rows} != {i['id'] for i in items}:
        raise ValueError("Incomplete collection")
    journal = dest / "judge.jsonl"
    if not journal.exists():
        shutil.copyfile(source / "judge.jsonl", journal)
    baseline = score_rows(items, [prior[i["id"]] for i in items], meta["judge_workers"], journal=journal)
    revised = score_rows(items, rows, meta["judge_workers"], journal=journal)
    first_valid = None
    if meta.get('record_candidate_comparison'):
        first_rows = copy.deepcopy(rows)
        for row in first_rows:
            if row.get('generation_candidates'):
                row['response']['answer'] = row['generation_candidates'][0]
        first_valid = score_rows(items, first_rows, meta["judge_workers"], journal=journal)
    report = {**meta, "baseline_tutor": baseline["tutor"], "revised_tutor": revised["tutor"],
              "attempted": sum(r["generation_attempted"] for r in rows),
              "error_requests": sum(bool(r["generation_errors"]) for r in rows),
              "baseline": baseline, "revised": revised,
              "first_valid_tutor":first_valid['tutor'] if first_valid else None, "first_valid":first_valid}
    (dest / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("variant","baseline_tutor","revised_tutor","first_valid_tutor","attempted","error_requests")}), flush=True)


def reuse_journal(reference, metadata, dest):
    """Reuse exact local judge payloads for a controlled follow-up, not HTTP runs."""
    reference = reference.resolve()
    if not reference.is_relative_to(ROOT / 'experiments/.cache'):
        raise ValueError('Expected local generation reference')
    prior = json.loads((reference / 'manifest.json').read_text())
    if (prior.get('scope') != metadata['scope'] or not (reference / 'result.json').exists()
            or any(prior.get(key) != metadata[key] for key in
                   ('source', 'source_response_sha256', 'evaluator_hashes'))
            or prior.get('source_manifest', {}).get('model') != metadata['source_manifest']['model']):
        raise ValueError('Judge reference source/model/evaluator differs or is incomplete')
    journal = reference / 'judge.jsonl'
    metadata['judge_reused_from'] = str(reference.relative_to(ROOT))
    metadata['judge_reused_sha256'] = hashlib.sha256(journal.read_bytes()).hexdigest()
    shutil.copyfile(journal, dest / 'judge.jsonl')


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--variant", choices=("jh_guard_fitted_contextfast", "jh_r3_answer_choice", "jh_r3_answer_pair", "jh_r2_english_answer",
                                          "jh_r3_quote_answer", "jh_r3_complete_answer",
                                          "jh_r3_checked_answer", "jh_r3_sentence_answer", "jh_r3_answer_support",
                                          "jh_r3_extract_section", "jh_r3_extract_chunk",
                                          "jh_r3_extract_union", "jh_r3_extract_union_tables"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=32)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument('--judge-cache-from', type=Path,
                    help='Completed component run with the same own source; reuse identical judge calls')
    ap.add_argument('--source-selection',choices=('frozen','section90_control','section90_changed'),default='frozen')
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    if args.resume:
        dest = args.resume.resolve()
        if not dest.is_relative_to(ROOT / "experiments/.cache"):
            ap.error("Only local public diagnostics")
        score(dest)
        return
    if (not args.source or not args.variant or not args.name
            or not re.fullmatch(r"[A-Za-z0-9_-]+", args.name)
            or not 1 <= args.workers <= 64 or not 1 <= args.judge_workers <= 64):
        ap.error("Invalid arguments")
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        ap.error("Only local public diagnostics")
    manifest, items, prior, old_result = source_data(source)
    changed_sources = None
    if args.source_selection != 'frozen':
        if args.variant != 'jh_guard_fitted_contextfast':
            ap.error('Section interventions use the unchanged base answer generator')
        from pipelines.kimjaehak.learned_sections import train_section_classifier
        vectorizer, model = train_section_classifier(1000)
        allowed = set(Searcher(load_chunks(ROOT/'kb')).section_ids())
        changed_sources = section_changes(items, prior, vectorizer, model, allowed)
        if not changed_sources:
            ap.error('No eligible own source predictions change; no API calls made')
    dest = ROOT / "experiments/.cache" / ("jh_generation_" + args.name)
    dest.mkdir(exist_ok=False)
    cls = load(args.variant)
    metadata = {"scope":"frozen_own_routing_source_generation_component_not_http", "variant":args.variant,
                "source":str(source.relative_to(ROOT)), "source_manifest":manifest,
                "source_response_sha256":hashlib.sha256((source/'responses.jsonl').read_bytes()).hexdigest(),
                "source_tutor_score":old_result["tutor"], "workers":args.workers, "judge_workers":args.judge_workers,
                "count":len(items), "evaluator_hashes":evaluator_hashes(),
                "commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                "dirty":bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                "answer_choices":getattr(cls,'answer_choices',1),
                "tutor_extract_scope":getattr(cls,'tutor_extract_scope',None),
                "answer_prompt_sha256":hashlib.sha256(cls.answer_systems[Label.NORMAL].encode()).hexdigest(),
                "selection_prompt_sha256":hashlib.sha256(getattr(cls,'answer_choice_system','').encode()).hexdigest(),
                "runtime_limit":"generation_only_26s_not_end_to_end",
                "record_candidate_comparison":True,
                "runtime_labels":"own_answer_action_and_own_citation_never_gold"}
    metadata['source_selection'] = args.source_selection
    if changed_sources is not None:
        metadata['source_change_predictions'] = changed_sources
    if args.judge_cache_from:
        reuse_journal(args.judge_cache_from, metadata, dest)
    (dest/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    asyncio.run(collect(items,prior,args.variant,args.workers,dest,args.source_selection,changed_sources))
    if args.collect_only:
        print('SCORING_PENDING',dest,flush=True)
    else:
        score(dest)


if __name__ == "__main__":
    main()
