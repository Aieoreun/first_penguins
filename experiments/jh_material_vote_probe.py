"""PUBLIC-derived, paired MATERIAL component probe; never an HTTP/full score.

collect makes organizer API calls; self-test uses mocks and never reads .env.
The first of the same three material calls is the single-vote counterfactual.
Both arms share retrieval, block/other calls, gates and the original priority.
"""
import argparse
import asyncio
from collections import Counter
import contextvars
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# v1/v2/v4 read environment controls at import time. Import only the lightweight
# configuration module first, before importing any inherited pipeline modules.
from pipelines.kimjaehak.config import BASELINE_ENV, configure

PROBE_ENV = {'V4_PREPARE_LIMIT':'14', 'V4_DEADLINE':'24'}


def configure_probe():
    configure()
    os.environ.update(PROBE_ENV)


configure_probe()
from core.schema import Label, POLICY
from pipelines.v1.pipeline import _json_object
from pipelines.v2.pipeline import _majority, _priority
from pipelines.kimjaehak.material_votes import Pipeline as MaterialVotes, select_material_vote
from pipelines.kimjaehak.voted import BLOCKS, OTHERS

CACHE = (ROOT/'experiments/.cache').resolve()
PHASE = contextvars.ContextVar('development_probe_check_name', default=None)


def effective_flags():
    from pipelines.v1 import pipeline as v1
    from pipelines.v2 import pipeline as v2
    from pipelines.v4 import pipeline as v4
    return {
        'module_v1':{name:getattr(v1,name) for name in (
            'CLASSIFY_EFFORT','CLASSIFY_TIMEOUT','ANSWER_EFFORT','GUARD_EFFORT',
            'EMBED_TIMEOUT','SECOND_TIMEOUT')},
        'module_v2':{name:getattr(v2,name) for name in (
            'REWRITE','REWRITE_EFFORT','REWRITE_TIMEOUT','SECTION_SOURCE','VOTES','SEEK_FIRST',
            'MATERIAL_EFFORT','MATERIAL_INPUT','MATERIAL_SECTIONS','PROMPTS','SAFE_RULE',
            'ANSWER_EFFORT_V2','ANSWER_MAX_TOKENS')},
        'module_v4':{'REQUEST_DEADLINE':v4.REQUEST_DEADLINE,'PREPARE_LIMIT':v4.PREPARE_LIMIT},
        'pipeline':{name:getattr(ProbePipeline,name) for name in (
            'deadline','prepare_limit','tutor_material_votes','short_votes','block_min_votes',
            'seek_first','safe_by_rule','material_max_chunks','learned_gate_c',
            'learned_gate_threshold','learned_section_threshold')},
        'environment':{key:os.environ.get(key) for key in {**BASELINE_ENV,**PROBE_ENV}},
    }


def sha(data): return hashlib.sha256(data).hexdigest()


def confined(path, root, *, must_exist=True):
    path = Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Expected a path inside the local development cache')
    if must_exist and not path.exists(): raise ValueError('Missing local diagnostic file')
    return path


def validate_dataset(source, root=ROOT):
    """Accept only complete tutor paraphrases with verifiable public lineage."""
    root = root.resolve()
    cache = (root/'experiments/.cache').resolve()
    source = confined(source, cache)
    if source.name != 'derived_inputs.jsonl': raise ValueError('Expected derived_inputs.jsonl')
    metadata_path = confined(source.parent/'derivation.json', cache)
    metadata = json.loads(metadata_path.read_text())
    source_bytes = source.read_bytes()
    if (metadata.get('scope') != 'public_paraphrase_full_diagnostic'
            or metadata.get('score_scope') != 'tutor'
            or metadata.get('dataset_sha256') != sha(source_bytes)):
        raise ValueError('Dataset provenance/hash does not match tutor public paraphrases')
    rows = [json.loads(s) for s in source_bytes.decode().splitlines()]
    if not rows or len(rows) != metadata.get('count') or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate dataset')
    generated_path = confined(root/metadata['source'], cache)
    if sha(generated_path.read_bytes()) != metadata.get('source_sha256'):
        raise ValueError('Paraphrase source hash mismatch')
    generation_path = confined(generated_path.parent/'manifest.json', cache)
    generation = json.loads(generation_path.read_text())
    public_hashes = {name: sha((root/'public'/name).read_bytes()) for name in ('tutor.jsonl','guard.jsonl')}
    if (generation.get('scope') != 'provided_llm_public_paraphrases'
            or generation.get('public_hashes') != public_hashes):
        raise ValueError('Generation provenance differs from bundled public data')
    public = {r['id']:r for r in map(json.loads,(root/'public/tutor.jsonl').read_text().splitlines())}
    generated_rows = [json.loads(s) for s in generated_path.read_text().splitlines()]
    generated = {r['source_id']:r for r in generated_rows}
    if len(generated) != len(generated_rows) or set(generated) != set(generation['source_ids']):
        raise ValueError('Incomplete/duplicate paraphrase source IDs')
    for row in rows:
        original = public.get(row.get('source_id'))
        paraphrase = generated.get(row.get('source_id'))
        if row.get('endpoint') != 'tutor' or original is None or paraphrase is None:
            raise ValueError('Only public tutor-derived rows are allowed')
        if row.get('labels') != original.get('labels'):
            raise ValueError('Inherited public labels were modified')
        messages = row.get('messages',[])
        if (not messages or messages[-1].get('role') != 'user'
                or messages[:-1] != original['messages'][:-1]
                or paraphrase.get('error') is not None
                or messages[-1]['content'] not in paraphrase['paraphrases']):
            raise ValueError('Messages are not an unchanged public history plus its verified paraphrase')
    return rows, {'dataset':str(source.relative_to(root)), 'dataset_sha256':sha(source_bytes),
                  'derivation_sha256':sha(metadata_path.read_bytes()),'derivation':metadata,
                  'generation_manifest_sha256':sha(generation_path.read_bytes()),'public_hashes':public_hashes}


def payload_only(row):
    # IDs, labels, references and evidence never enter the pipeline invocation.
    return {'spec_version':'1.0','session_id':'s_'+uuid.uuid4().hex,
            'messages':copy.deepcopy(row['messages'])}


class CaptureLLM:
    """Request-local facade; underlying organizer clients may be shared."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.material = []
        self.chat_counts = Counter()

    def __getattr__(self, key): return getattr(self.delegate, key)

    async def chat(self, system, user, **kwargs):
        phase = PHASE.get()
        self.chat_counts[phase or 'unclassified'] += 1
        trace = None
        if phase == 'material':
            trace = {'call_index':len(self.material),'response':{},'raw_response':None,
                     'error':None,'status':'started','effort':kwargs.get('effort'),
                     'timeout':kwargs.get('timeout'),'max_tokens':kwargs.get('max_tokens'),
                     'system_sha256':sha(system.encode()),'user_sha256':sha(user.encode())}
            self.material.append(trace)
        try:
            text = await self.delegate.chat(system,user,**kwargs)
        except BaseException as exc:
            if trace is not None:
                trace.update(error=type(exc).__name__,status='cancelled' if isinstance(exc,asyncio.CancelledError) else 'failed')
            raise  # Preserve the original parent's error/fallback/cancellation behavior.
        if trace is not None:
            trace['raw_response'] = text
            trace['response'] = _json_object(text) or {}
            trace['status'] = 'returned'
        return text


class ProbePipeline(MaterialVotes):
    async def _ask(self, state, name, system, user, effort, timeout):
        token = PHASE.set(name)
        try:
            return await super()._ask(state,name,system,user,effort,timeout)
        finally:
            PHASE.reset(token)

    async def _collect_checks(self, state, check, material):
        checks = await super()._collect_checks(state,check,material)
        state._probe_checks = checks
        return checks


def replay_label(blocks,others,material,*,seek_first,block_min_votes):
    """Mirror voted.Pipeline._classify aggregation, using its original priority."""
    block_votes = [x['category'] for x in blocks if isinstance(x.get('category'),str) and x['category'] in BLOCKS]
    other_votes = [x['category'] for x in others if isinstance(x.get('category'),str) and x['category'] in OTHERS]
    personal = [x['self_info'] for x in others if isinstance(x.get('self_info'),bool)]
    block = _majority(block_votes) or 'none'
    if block != 'none' and block_votes.count(block) < block_min_votes: block = 'none'
    flags = {block:True,_majority(other_votes) or 'none':True,'self_info':bool(_majority(personal)),
             'out_of_material':isinstance(material.get('quote'),str) and not material['quote'].strip()}
    return next((label for key,label in _priority(seek_first) if flags.get(key)),Label.NORMAL)


def verdict(label):
    # State.policy also falls back to NORMAL if preparation did not set a label.
    effective = label if label is not None else Label.NORMAL
    return {'label':effective.value,'category':POLICY[effective].category.value,'action':POLICY[effective].action.value}


def paired_record(state, wrapper, pipeline):
    actual = verdict(state.label)
    traces = wrapper.material
    checks = getattr(state,'_probe_checks',None)
    result = {'actual':actual,'errors':list(state.errors),'material_calls':traces,
              'chat_counts':dict(wrapper.chat_counts),'debug':copy.deepcopy(state.debug)}
    if checks is None:
        complete = not traces and 'prepare:Deadline' not in state.errors
        result.update(status='no_material' if complete else 'prepare_incomplete',
                      paired_valid=complete,first=actual,majority=actual)
        return result
    if len(traces) != 3 or any(t['status'] in {'started','cancelled'} for t in traces):
        raise ValueError('Completed material checks lack three completed ordered traces')
    blocks,others,selected = checks
    responses = [t['response'] for t in traces]
    voted = select_material_vote(responses)
    if selected != voted: raise ValueError('Captured material responses differ from runtime voting')
    options = {'seek_first':pipeline.seek_first,'block_min_votes':pipeline.block_min_votes}
    first = replay_label(blocks,others,responses[0],**options)  # First failure stays {}, NOT first valid.
    majority = replay_label(blocks,others,voted,**options)
    if verdict(majority) != actual: raise ValueError('Replayed majority does not reproduce actual prepare label/action')
    result.update(status='material_paired',paired_valid=True,first=verdict(first),majority=verdict(majority),
                  block_checks=blocks,other_checks=others,selected_material=voted)
    return result


async def collect_one(base, delegate, row):
    pipeline = copy.copy(base)
    wrapper = CaptureLLM(delegate)
    pipeline.llm = wrapper  # Never mutate the shared base's LLM/client pointer.
    state = pipeline._new_state(payload_only(row),False)
    start = time.monotonic()
    await pipeline._prepare_within(state)  # Original prepare budget; no generation/output guard.
    record = paired_record(state,wrapper,pipeline)
    record.update(id=row['id'],source_id=row['source_id'],seconds=time.monotonic()-start,
                  gold_action=row['labels']['action'],gold_category=row['labels']['category'])
    for arm in ('first','majority'):
        record[arm]['action_correct'] = record[arm]['action'] == row['labels']['action']
        record[arm]['category_correct'] = record[arm]['category'] == row['labels']['category']
    return record


def summarize(records):
    valid = [r for r in records if r['paired_valid']]
    material = [r for r in valid if r['status']=='material_paired']
    result = {'count':len(records),'paired_valid':len(valid),'material_paired':len(material),
              'no_material':sum(r['status']=='no_material' for r in records),
              'prepare_incomplete':sum(r['status']=='prepare_incomplete' for r in records),
              'requests_with_errors':sum(bool(r['errors']) for r in records),
              'material_call_errors':sum(t['error'] is not None for r in records for t in r['material_calls']),
              'material_invalid_quotes':sum(not isinstance(t['response'].get('quote'),str) for r in records for t in r['material_calls']),
              'scope':'input_component_no_generation_no_judge_no_http_or_full_score'}
    for name,rows in [('all_complete',valid),('material_only',material)]:
        result[name]={'count':len(rows),**{arm:{'action_correct':sum(r[arm]['action_correct'] for r in rows),
                        'category_correct':sum(r[arm]['category_correct'] for r in rows)} for arm in ('first','majority')},
                      'action_changed':sum(r['first']['action']!=r['majority']['action'] for r in rows),
                      'action_gains':sum(not r['first']['action_correct'] and r['majority']['action_correct'] for r in rows),
                      'action_losses':sum(r['first']['action_correct'] and not r['majority']['action_correct'] for r in rows)}
    return result


async def collect(args):
    rows,provenance = validate_dataset(args.source)
    output = confined(args.output,CACHE,must_exist=False)
    if output.exists(): raise FileExistsError('Never overwrite component diagnostic')
    if not 1 <= args.workers <= 32: raise ValueError('workers must be 1..32; default 16')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=True)
    # .env may contain experimental V2_* flags. Restore the same fixed controls
    # after loading API settings so metadata and runtime environment agree.
    configure_probe()
    model = os.environ.get('OPENAI_MODEL','')
    if model not in {'gpt-5-nano','openai/gpt-5-nano'}: raise ValueError('Only configured organizer gpt-5-nano is allowed')
    if not all(os.environ.get(k) for k in ('OPENAI_API_KEY','OPENAI_BASE_URL')): raise ValueError('Missing organizer .env configuration')
    if args.embedding_mode == 'off':
        for key in ('EMBEDDING_BASE_URL','EMBEDDING_MODEL','EMBEDDING_API_KEY'):os.environ[key]=''
    from core.kb import load_chunks
    from core.search import Searcher,load_vectors
    from core.llm import LLM
    from threadpoolctl import threadpool_limits
    llm = LLM()
    try:
        chunks = load_chunks(ROOT/'kb')
        index = ROOT/'index/kb_embeddings.json'
        vectors = None
        if args.embedding_mode == 'env':
            if llm.embedding_model not in {'text-embedding-3-small','openai/text-embedding-3-small'} or not llm.can_embed:
                raise ValueError('Missing permitted organizer embedding configuration')
            vectors = load_vectors(index,llm.embedding_model,chunks)
            if vectors is None: raise ValueError('Existing KB embedding index required; this probe never primes embeddings')
        with threadpool_limits(limits=1):base = ProbePipeline(Searcher(chunks,vectors),llm)
        output.mkdir(exist_ok=False)
        modules = sorted({Path(sys.modules[cls.__module__].__file__).resolve() for cls in ProbePipeline.__mro__[1:-1]})
        modules += [ROOT/'core/llm.py',ROOT/'core/search.py',ROOT/'core/state.py',ROOT/'core/schema.py',ROOT/'pipelines/v2/prompts.py']
        manifest = {'scope':'paired_public_derived_material_component','provenance':provenance,
                    'baseline_env':dict(BASELINE_ENV),'probe_env':dict(PROBE_ENV),'effective_flags':effective_flags(),
                    'model':llm.model,'embedding_model':llm.embedding_model or None,'embedding_mode':args.embedding_mode,
                    'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'script_sha256':sha(Path(__file__).read_bytes()),'source_hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in modules},
                    'kb_hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted((ROOT/'kb').rglob('*')) if p.is_file()},
                    'embedding_index_sha256':sha(index.read_bytes()) if vectors is not None else None,
                    'workers':args.workers,'prepare_limit':base.prepare_limit,'request_deadline':base.deadline,
                    'material_votes':base.tutor_material_votes,'judge':False,'generation':False,'http':False,'full_score':False,
                    'gold_passed_to_pipeline':False,'runtime_modified':False,
                    'limitations':['First-vote and majority share identical block/other samples and retrieval.',
                                   'Both arms run under three-material-call load; not a one-call latency or total-score comparison.',
                                   'Prepare-incomplete rows are reported and excluded from paired correctness, never silently discarded.',
                                   'Public-derived labels are inherited; previously inspected development data, not independent holdout.']}
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        sem = asyncio.Semaphore(args.workers)
        records = []
        with (output/'responses.jsonl').open('x') as stream:
            async def one(row):
                async with sem:
                    record = await collect_one(base,llm,row)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                    records.append(record)
            tasks = [asyncio.create_task(one(row)) for row in rows]
            try: await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                raise
        result = summarize(records)
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (output/'script.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(result,ensure_ascii=False))
    finally:
        await llm._chat.close()
        if llm._embed is not None:await llm._embed.close()


def self_test():
    import tempfile
    import unittest
    from types import SimpleNamespace
    from unittest.mock import AsyncMock,patch
    from core.state import State
    from pipelines.kimjaehak.voted import Pipeline as Voted

    class Tests(unittest.IsolatedAsyncioTestCase):
        def pipeline(self,values):
            p = ProbePipeline.__new__(ProbePipeline)
            p.llm = CaptureLLM(SimpleNamespace(chat=AsyncMock(side_effect=values)))
            return p,p._new_state({'messages':[{'role':'user','content':'question'}]},False)

        async def test_first_failure_preserves_original_empty_fallback(self):
            p,s = self.pipeline([TimeoutError(),'{"quote":""}','{"quote":""}'])
            selected = await p._ask(s,'material','system','user','low',10)
            s._probe_checks = ([{'category':'none'}],[{'category':'none'}],selected)
            s.label = Label.OUT_OF_MATERIAL
            record = paired_record(s,p.llm,p)
            self.assertEqual(record['first']['label'],Label.NORMAL.value)
            self.assertEqual(record['majority']['label'],Label.OUT_OF_MATERIAL.value)
            self.assertEqual(s.errors,['material:TimeoutError'])

        async def test_order_tie_and_raw_invalid_capture(self):
            p,s = self.pipeline(['{"quote":"support"}','not json','{"quote":""}'])
            selected = await p._ask(s,'material','system','user','low',10)
            self.assertEqual(selected,{'quote':'support'})
            self.assertEqual([t['call_index'] for t in p.llm.material],[0,1,2])
            self.assertEqual(p.llm.material[1]['raw_response'],'not json')
            self.assertEqual(p.llm.material[1]['response'],{})
            self.assertIsNone(PHASE.get())

        async def test_capture_keeps_completion_independent_order(self):
            started=asyncio.Event();release=[asyncio.Event() for _ in range(3)];calls=[]
            async def chat(*args,**kwargs):
                i=len(calls);calls.append(i)
                if len(calls)==3:started.set()
                await release[i].wait()
                return json.dumps({'quote':str(i)})
            p,s=self.pipeline([]);p.llm=CaptureLLM(SimpleNamespace(chat=chat))
            task=asyncio.create_task(p._ask(s,'material','s','u','low',10))
            try:
                await asyncio.wait_for(started.wait(),1)
                for i in (2,1,0):release[i].set();await asyncio.sleep(0)
                self.assertEqual(await task,{'quote':'0'})
                self.assertEqual([t['response']['quote'] for t in p.llm.material],['0','1','2'])
            finally:task.cancel();await asyncio.gather(task,return_exceptions=True)

        async def test_replay_matches_original_classifier_priority(self):
            p,s=self.pipeline([])
            cases=[([{'category':b}],[{'category':o,'self_info':pii}],{'quote':q})
                   for b in ['none','academic_dishonesty','jailbreak_injection','harmful_request']
                   for o in ['none','answer_seeking','emotional_distress'] for pii in [False,True] for q in ['', 'support']]
            for checks in cases:
                with patch.object(ProbePipeline,'_collect_checks',new=AsyncMock(return_value=checks)):
                    await Voted._classify(p,s)
                self.assertEqual(replay_label(*checks,seek_first=p.seek_first,block_min_votes=p.block_min_votes),s.label)

        async def test_no_material_fastgate_and_low_signal_are_unchanged(self):
            p,s=self.pipeline([])
            for label in (Label.NORMAL,Label.LOW_SIGNAL_INPUT,Label.HARMFUL_REQUEST):
                s.label=label;s.debug={'learned_gate':True}
                r=paired_record(s,p.llm,p)
                self.assertEqual(r['status'],'no_material');self.assertEqual(r['first'],r['majority'])
            s.errors.append('prepare:Deadline')
            self.assertFalse(paired_record(s,p.llm,p)['paired_valid'])

        async def test_nonmaterial_not_captured_and_cancellation_propagates(self):
            p,s=self.pipeline(['{"category":"none"}',asyncio.CancelledError(),'{}','{}'])
            await p._ask(s,'other','s','u','low',10)
            self.assertEqual(p.llm.material,[])
            with self.assertRaises(asyncio.CancelledError):await p._ask(s,'material','s','u','low',10)
            self.assertIsNone(PHASE.get())
            self.assertEqual(s.errors,[])

        async def test_collect_invokes_original_prepare_without_gold_or_generation(self):
            base=ProbePipeline.__new__(ProbePipeline);base.llm=object()
            row={'id':'public','source_id':'parent','messages':[{'role':'user','content':'query'}],
                 'labels':{'action':'answer','category':'normal'},'evidence':['SECRET']}
            seen=[]
            original=ProbePipeline._new_state
            def new_state(self,body,guard):
                seen.append(body);return original(self,body,guard)
            async def prepare(self,state):state.label=Label.NORMAL
            with patch.object(ProbePipeline,'_new_state',new=new_state),patch.object(ProbePipeline,'_prepare_within',new=prepare):
                result=await collect_one(base,SimpleNamespace(),row)
            self.assertEqual(set(seen[0]),{'spec_version','session_id','messages'})
            self.assertEqual(result['status'],'no_material');self.assertIsNot(base.llm,None)
            self.assertEqual(row['messages'],seen[0]['messages'])
            self.assertIsNot(row['messages'],seen[0]['messages'])

        async def test_dataset_hash_lineage_and_tutor_only(self):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'public').mkdir();cache=root/'experiments/.cache';cache.mkdir(parents=True)
                public={'id':'source','messages':[{'role':'user','content':'original'}],
                        'labels':{'category':'normal','action':'answer'}}
                (root/'public/tutor.jsonl').write_text(json.dumps(public)+'\n');(root/'public/guard.jsonl').write_text('')
                gen=cache/'generation';gen.mkdir();generated={'source_id':'source','error':None,'paraphrases':['changed']}
                facts=gen/'paraphrases.jsonl';facts.write_text(json.dumps(generated)+'\n')
                (gen/'manifest.json').write_text(json.dumps({'scope':'provided_llm_public_paraphrases','source_ids':['source'],
                    'public_hashes':{n:sha((root/'public'/n).read_bytes()) for n in ('tutor.jsonl','guard.jsonl')}}))
                ds=cache/'derived';ds.mkdir();source=ds/'derived_inputs.jsonl'
                row={**public,'id':'derived','source_id':'source','endpoint':'tutor','messages':[{'role':'user','content':'changed'}]}
                source.write_text(json.dumps(row)+'\n')
                meta={'scope':'public_paraphrase_full_diagnostic','score_scope':'tutor','count':1,'dataset_sha256':sha(source.read_bytes()),
                      'source':str(facts.relative_to(root)),'source_sha256':sha(facts.read_bytes())}
                (ds/'derivation.json').write_text(json.dumps(meta))
                self.assertEqual(validate_dataset(source,root)[0],[row])
                source.write_text(source.read_text()+' ')
                with self.assertRaises(ValueError):validate_dataset(source,root)
                row['endpoint']='guard';source.write_text(json.dumps(row)+'\n');meta['dataset_sha256']=sha(source.read_bytes())
                (ds/'derivation.json').write_text(json.dumps(meta))
                with self.assertRaises(ValueError):validate_dataset(source,root)

        async def test_dotenv_flags_are_restored_before_any_api_client(self):
            def fake_dotenv(*args,**kwargs):
                os.environ.update({key:'contaminated' for key in {**BASELINE_ENV,**PROBE_ENV}})
                os.environ['OPENAI_MODEL']='deliberately-invalid-no-api'
            args=SimpleNamespace(source=Path('unused'),output=Path('unused'),workers=16,embedding_mode='env')
            with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ):
                output=Path(directory)/'never-created'
                with patch(__name__+'.validate_dataset',return_value=([],{})),patch(__name__+'.confined',return_value=output),patch('dotenv.load_dotenv',side_effect=fake_dotenv):
                    with self.assertRaisesRegex(ValueError,'Only configured organizer'):
                        await collect(args)
                self.assertEqual(effective_flags()['environment'],{**BASELINE_ENV,**PROBE_ENV})
                self.assertFalse(output.exists())

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test',help='Mocks only; no .env loading or API clients')
    sub.add_parser('inspect-config',help='Show pinned/effective flags without loading .env or making API calls')
    runner=sub.add_parser('collect',help='Explicit organizer-API component collection; no generation or judge')
    runner.add_argument('--source',type=Path,required=True)
    runner.add_argument('--output',type=Path,required=True)
    runner.add_argument('--workers',type=int,default=16)
    runner.add_argument('--embedding-mode',choices=('env','off'),default='env')
    args=parser.parse_args()
    if args.command=='self-test':self_test()
    elif args.command=='inspect-config':print(json.dumps({'baseline_env':BASELINE_ENV,'probe_env':PROBE_ENV,'effective_flags':effective_flags()},sort_keys=True))
    else:asyncio.run(collect(args))
