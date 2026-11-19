"""Fixed public-derived tutor comparisons with privacy-safe aggregate counters."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from jh_compare import ROOT, load_public, score_rows
from jh_transfer_probe import build_items
from experiments.jh_environment import load_environment
from experiments import jh_guard_fusion_compare as runner

SOURCE = ROOT / 'experiments/.cache/jh_augment_public2_1/paraphrases.jsonl'
DEFAULT_CHOICES = ('cm_char_mlp50', 'tm_selective', 'tm_replace')
CHOICES = DEFAULT_CHOICES + ('r7_binary_guard75', 'r7_tax_scope', 'r7_tax_concrete', 'r7_binary_guard', 'r7_calc_review', 'r7_numeric_verdict', 'r7_numeric_compact', 'r7_numeric_direct', 'r7_material_verdict', 'r7_calc_noquote', 'r7_calc_compact', 'r7_early_consensus', 'r7_task_examples1', 'r7_task_examples3', 'r7_adaptive3', 'r7_structured_calc', 'r7_scope_structured', 'r7_examples1', 'r7_examples3', 'r7_material_calculator', 'r7_calculator', 'r7_minimal_span', 'r7_adaptive_span', 'r7_blend_adaptive', 'r7_cancel_guard', 'r7_cancel_tutor', 'r7_cancel_both', 'r7_material_scope', 'r7_apply_scoped', 'r7_ground3', 'r7_ground_adaptive', 'r7_apply_concrete', 'r7_apply_joint', 'r7_ground_input', 'r7_ground_union', 'r7_apply_numeric', 'r7_apply_all', 'r7_batchconcat', 'r7_batchblend', 'r7_concat', 'r7_blend', 'r7_fastblend', 'r7_boundary', 'r7_verdict', 'r7_product', 'r7_other', 'r7_scaffold', 're_c_groundlow2k', 're_c_input2k', 're_c_ground2k', 're_c_input', 're_c_ground', 're_c_safety', 're_a_input', 're_a_source', 're_a_answer', 'np_parallel','np_timeouts','np_peace_block','np_unlearned','np_input_budgets','np_block_only')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True)
    ap.add_argument('--order', nargs='+', choices=CHOICES, default=list(DEFAULT_CHOICES))
    ap.add_argument('--alternative', action='store_true')
    ap.add_argument('--mixed', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not args.name.replace('_','').replace('-','').isalnum() or len(set(args.order)) != len(args.order):
        ap.error('Use a unique simple name and nonduplicated candidates')
    load_environment()
    dest = ROOT / 'experiments/.cache' / ('jh_tutor_mlp_' + args.name)
    originals = {r['id']:r for r in load_public()}
    derived, _ = build_items(SOURCE, args.alternative)
    items = [{**originals[r['source_id']], 'id':r['id'], 'source_id':r['source_id'],
              'messages':r['messages']} for r in derived
             if args.mixed or originals[r['source_id']]['endpoint']=='tutor']
    encoded = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in items)
    current = {'scope':'public_derived_tutor_mlp_development_not_independent_holdout',
        'source_sha256':runner.digest(SOURCE.read_bytes()), 'dataset_sha256':runner.digest(encoded.encode()),
        'protected':runner.protected_hashes(ROOT), 'pipelines':runner.pipeline_hashes(ROOT),
        'bootstrap_sha256':runner.digest(runner.BOOTSTRAP.encode()), 'order':args.order,
        'mixed':args.mixed, 'alternative':args.alternative,
        'chat_model':os.environ['OPENAI_MODEL'], 'embedding_model':os.environ.get('EMBEDDING_MODEL'),
        'inherited_labels':True, 'count':len(items)}
    if args.resume:
        recorded = json.loads((dest/'plan.json').read_text())
        if any(recorded.get(k)!=v for k,v in current.items()):
            raise ValueError('Dataset, code or experiment configuration changed')
    else:
        dest.mkdir()
        current['commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        runner.write_json(dest/'plan.json',current)
        (dest/'inputs.jsonl').write_text(encoded)
    env = runner.server_env(os.environ, ROOT, 'env')
    # Never overlap collection with scoring or another candidate's load.
    for label in args.order:
        run = dest/label
        if not (run/'runtime.json').exists():
            if run.exists(): raise ValueError('Partial collection: preserve it and use a fresh run name')
            runner.collect_candidate(label,ROOT,items,run,env,32,'env',current)
        print('COLLECTED '+label,flush=True)
    summaries=[]
    for label in args.order:
        run = dest/label
        if not (run/'result.json').exists():
            rows=[json.loads(s) for s in (run/'responses.jsonl').read_text().splitlines()]
            result=score_rows(items,rows,32,journal=run/'judge.jsonl')
            runner.write_json(run/'result.json',result)
        result=json.loads((run/'result.json').read_text())
        runtime=json.loads((run/'runtime.json').read_text())
        summary={'candidate':label,**{k:result[k] for k in ('total','tutor','guard','invalid','judge_errors')},'runtime':runtime}
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    if runner.protected_hashes(ROOT)!=current['protected'] or runner.pipeline_hashes(ROOT)!=current['pipelines']:
        raise ValueError('Protected files or pipeline code changed during run')
    runner.write_json(dest/'summary.json',summaries)
    (dest/'COMPLETE').write_text('All public-derived responses scored.\n')


if __name__=='__main__': main()
