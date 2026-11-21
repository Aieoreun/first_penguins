"""API-free recovery operating points, evaluated only under injected failure."""
import asyncio,json
from copy import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from pipelines import load
from core.kb import load_chunks
from core.search import Searcher
from experiments.jh_round8_fault_probe import FailedLLM,collect
from experiments.jh_guard_fusion_datasets import load_dataset
from experiments.jh_compare import score_rows


def main():
    dest=ROOT/'experiments/.cache/jh_r8_recovery_thresholds1';dest.mkdir()
    base=load('jh_r8_resilient_guard')(Searcher(load_chunks(ROOT/'kb')),FailedLLM())
    reports=[]
    for dataset in ['paraphrase','alternative','novel']:
        items,provenance=load_dataset(dataset,ROOT)
        for complete in [.3,.5,.7]:
            for block in [.5,.7,.9]:
                p=copy(base);p.guard_failure_input_threshold=complete;p.guard_failure_block_threshold=block
                rows,errors,rescues=asyncio.run(collect(p,items))
                result=score_rows(items,rows,1,no_judge=True)
                report={'dataset':dataset,'primary_metric':provenance['primary_metric'],
                        'complete_threshold':complete,'block_threshold':block,
                        'guard_or_input_only_score':result['guard'],
                        'details':result['guard_details'],'invalid':result['invalid'],
                        'fallback_counts':dict(rescues)}
                reports.append(report);print(json.dumps(report),flush=True)
    out={'scope':'posthoc_public_recovery_threshold_screen_all_calls_fail_not_healthy_or_holdout',
         'api_calls':0,'selection_warning':'Do not promote from the highest single dataset score; thresholds affect only cases with missing remote verdicts.',
         'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Recovery threshold grid completed.\n')


if __name__=='__main__':main()
