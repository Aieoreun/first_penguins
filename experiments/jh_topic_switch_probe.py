"""Controlled tutor diagnostic: unrelated prior lesson, unchanged final KB question."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from jh_compare import ROOT, run_variant
from jh_generate_kb_facts import sections_and_hash
from pipelines.kimjaehak.config import VARIANTS


def transform(rows, sections):
    rows=sorted(rows,key=lambda r:r['id'])
    result=[]
    for index,target in enumerate(rows):
        if (target['endpoint']!='tutor' or target['labels']['category']!='normal'
                or len(target['messages'])!=1 or len(target['labels']['evidence'])!=1):
            raise ValueError('Expected reviewed, single-turn, normal KB facts')
        sid=target['labels']['evidence'][0]
        for mode in ('same_subject','different_subject'):
            prior=None
            for offset in range(1,len(rows)):
                candidate=rows[(index+offset)%len(rows)]
                other=candidate['labels']['evidence'][0]
                quote=candidate.get('label_support_quote','')
                same=other.split('-')[0]==sid.split('-')[0]
                if (other==sid or same!=(mode=='same_subject') or not quote or quote not in sections[other]
                        or re.search(r'\b[A-Z]{2,4}-\d{2}-\d{2}\b',quote)):
                    continue
                if target['labels']['gold_facts'][0] in quote:
                    continue
                prior=candidate
                break
            if prior is None:
                raise ValueError('No eligible distractor for '+target['id'])
            item=copy.deepcopy(target)
            item['id']=target['id']+'-switch-'+mode
            item['source_id']=target['id']
            item['context_source_id']=prior['id']
            item['context_mode']=mode
            item['messages']=[copy.deepcopy(prior['messages'][0]),
                              {'role':'assistant','content':prior['label_support_quote']},
                              {'role':'user','content':'이제 주제를 바꿔서 질문할게요. '+target['messages'][0]['content']}]
            result.append(item)
    return result


def followup(rows, sections):
    """Keep the original question in history; final request must resolve it."""
    result = []
    for target in sorted(rows, key=lambda r: r['id']):
        if (target['endpoint'] != 'tutor' or target['labels']['category'] != 'normal'
                or len(target['messages']) != 1 or len(target['labels']['evidence']) != 1
                or target['label_support_quote'] not in sections[target['labels']['evidence'][0]]):
            raise ValueError('Expected reviewed, single-turn, normal KB facts')
        for mode, request in (
                ('explain', '방금 질문한 내용이 잘 이해되지 않아. 그 내용을 설명해 줄래?'),
                ('core', '앞서 물어본 내용에서 핵심이 무엇인지 알고 싶어.')):
            item = copy.deepcopy(target)
            item['id'] = target['id'] + '-followup-' + mode
            item['source_id'] = target['id']
            item['context_mode'] = mode
            item['messages'] = [copy.deepcopy(target['messages'][0]),
                                {'role':'assistant', 'content':'질문하신 주제를 교재에서 함께 살펴볼게요.'},
                                {'role':'user', 'content':request}]
            result.append(item)
    return result


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=True)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--variant',choices=VARIANTS,default='jh_guard_fitted_contextfast')
    ap.add_argument('--build-only',action='store_true')
    ap.add_argument('--context-kind',choices=('topic_switch','followup'),default='topic_switch')
    ap.add_argument('--workers',type=int,default=64)
    ap.add_argument('--request-slots',type=int,default=64)
    ap.add_argument('--judge-workers',type=int,default=32)
    ap.add_argument('--embedding-mode',choices=('env','off'),default='env')
    args=ap.parse_args()
    source=args.source.resolve()
    if (not source.is_relative_to(ROOT/'experiments/.cache') or not re.fullmatch(r'[A-Za-z0-9_-]+',args.name)
            or not all(1<=n<=64 for n in (args.workers,args.request_slots,args.judge_workers))):
        ap.error('Invalid source, name, or concurrency')
    meta=json.loads((source.parent/'derivation.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    sections,kb_hash=sections_and_hash()
    if (meta['scope']!='provided_kb_fact_diagnostic' or meta['dataset_sha256']!=digest
            or meta['kb_sha256']!=kb_hash or not meta.get('review') or not meta.get('curation')):
        ap.error('Expected reviewed and curated KB diagnostic')
    rows=list(map(json.loads,source.read_text().splitlines()))
    items=(transform if args.context_kind=='topic_switch' else followup)(rows,sections)
    root=ROOT/'experiments/.cache'/('jh_topic_'+args.name)
    root.mkdir(exist_ok=False)
    dataset=''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in items)
    (root/'derived_inputs.jsonl').write_text(dataset)
    provenance={'scope':'provided_kb_fact_diagnostic','diagnostic_kind':'controlled_'+args.context_kind,
                'transformation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source':str(source.relative_to(ROOT)),'source_sha256':digest,'kb_sha256':kb_hash,
                'dataset_sha256':hashlib.sha256(dataset.encode()).hexdigest(),'count':len(items),
                'generation_model':meta['generation_model'],'source_provenance':meta,
                'labels':('inherited_unchanged_final_question_with_other_lesson_context' if args.context_kind=='topic_switch'
                          else 'inherited_original_question_in_history_with_anaphoric_explanation_request'),
                'transformation_uses_candidate_responses':False,'independent_source_holdout':False,
                'context_assistant_content':('exact_other_section_quote_from_provided_kb' if args.context_kind=='topic_switch'
                                             else 'fixed_neutral_acknowledgment_without_answer'),
                'score_interpretation':'tutor_only_controlled_stress_diagnostic_not_private_score'}
    (root/'derivation.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    if args.build_only:
        print(json.dumps({'count':len(items),'dataset_sha256':provenance['dataset_sha256']}))
        return
    args.no_judge=False
    args.collect_only=True
    args.endpoints=['tutor']
    run_variant(args,args.variant,1,items,root)
    dest=root/(args.variant+'_1')
    manifest=json.loads((dest/'manifest.json').read_text())
    manifest.update(provenance)
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('DERIVED_SCORING_PENDING',dest,flush=True)


if __name__=='__main__':main()
