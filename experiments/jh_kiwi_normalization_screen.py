"""Reproducible, offline tutor retrieval screen. No HTTP or pretrained similarities.

All inputs are existing public/development data. No production requests are read.
Writes only a new explicitly named cache directory, after successful computation.
"""
import argparse
import ast
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.kb import load_chunks
from core.search import BM25, tokens as baseline_tokens
from kiwipiepy import Kiwi

SOURCES = {
    'public119': 'public/tutor.jsonl',
    'kb55': 'experiments/.cache/jh_kb_probe_r2_base_off/derived_inputs.jsonl',
    'kb57': 'experiments/.cache/jh_kb_probe_r2_split_kb57_1_jh_guard_fitted_contextfast/derived_inputs.jsonl',
    'new28': 'experiments/.cache/jh_kb_probe_r3_misconception_frozen1/derived_inputs.jsonl',
    'topic_switch114': 'experiments/.cache/jh_topic_r3_switch_build1/derived_inputs.jsonl',
    'context_followup114': 'experiments/.cache/jh_topic_r3_followup_build1/derived_inputs.jsonl',
    'primary119': 'experiments/.cache/jh_transfer_r2_split_english_transfer1_jh_guard_fitted_contextfast/derived_inputs.jsonl',
    'alternative112': 'experiments/.cache/jh_transfer_r3_alternative_build1/derived_inputs.jsonl',
}
EXPECTED = dict(zip(SOURCES, (119,55,57,28,114,114,119,112)))
TEAM_REF = 'debb50d846bdf654bc815f836d3018805ac5feea'
TEAM_PATH = 'app/kb_index.py'
TEAM_NORMALIZE_PATH = 'app/normalize.py'
# Freeze five choices before opening any evaluation labels. Each must occur in KB
# and the existing teammate dictionary. No synonym list or external data is used.
FIVE_TERMS = ('다람쥐은행', '꿈틀이', '새싹적금', 'HelpyCode', 'kkumtle')
CODE_ID = re.compile(r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*')
NUMBER = re.compile(r'(?<![A-Za-z0-9_])[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?')
OLD_CODE = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')
# Protect fenced/inline code, quoted strings and indented or recognizably Python
# lines. Overprotection is deliberate: normalization is only a retrieval view.
PROTECTED = re.compile(
    r'```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|'
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r'(?m:^[ \t]+[^\n]+$)|'
    r'(?m:^(?:[A-Za-z_]\w*\s*=|(?:for|if|while|def|class|import|from|return)\b|'
    r'[A-Za-z_]\w*(?:\.\w+)*\s*\()[^\n]*$)')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)

def protected_segments(text):
    start = 0
    for match in PROTECTED.finditer(text):
        if match.start() > start:
            yield False, text[start:match.start()]
        yield True, match.group()
        start = match.end()
    if start < len(text):
        yield False, text[start:]

def clean_prose(text, nfkc=False):
    if nfkc:
        # Limited NFKC: fullwidth ASCII letters/digits and ideographic spaces only.
        # Do not fold operators, units, circled numbers, jamo, quotes or code.
        text = ''.join(unicodedata.normalize('NFKC', c) if (
            '\uff10' <= c <= '\uff19' or '\uff21' <= c <= '\uff3a'
            or '\uff41' <= c <= '\uff5a' or c == '\u3000') else c for c in text)
    text = re.sub(r'[ㅋㅎㅠㅜ]{2,}', ' ', text)
    return re.sub(r'[ \t\r\f\v]+', ' ', text)

def normalize_view(text, nfkc=False):
    return ''.join(part if protected else clean_prose(part, nfkc)
                   for protected, part in protected_segments(text))

def frozen_dictionary(chunks, team_source):
    tree = ast.parse(team_source)
    terms = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                 and n.target.id == '_COURSE_TERMS')
    corpus = '\n'.join(c.text for c in chunks)
    assert all(term in terms and term in corpus for term in FIVE_TERMS)
    return terms, [{'term': term, 'tag': 'NNP', 'score': 0,
                    'kb_chunk_occurrences': sum(term in c.text for c in chunks)}
                   for term in FIVE_TERMS]

class FixedBM25(BM25):
    """Same formula/tie breaking as core.BM25, with injectable tokenization."""
    def __init__(self, texts, tokenize):
        self.tokenize = tokenize
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.length = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.length) / len(self.length) if self.length else 1.
        df = Counter(w for t in self.tf for w in t)
        self.idf = {w: math.log(1+(len(texts)-d+.5)/(d+.5)) for w,d in df.items()}
        self.k1, self.b = 1.5, .75

    def scores(self, query):
        q = Counter(self.tokenize(query))
        out = []
        for tf, length in zip(self.tf, self.length):
            score = 0.
            for term in q:
                f = tf.get(term, 0)
                if f:
                    score += self.idf[term]*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*length/self.avg))
            out.append(score)
        return out

def ranked_sections(scores, chunks, chunk_limit=10):
    # Exactly mirror Searcher.search BM25 top10 before distinct section top3.
    indices = [i for i in sorted(range(len(scores)), key=lambda i:-scores[i]) if scores[i]>0][:chunk_limit]
    return list(dict.fromkeys(chunks[i].section_id for i in indices))

def timing(values):
    ordered = sorted(values)
    return {'mean_ms': 1000*statistics.mean(values), 'median_ms':1000*statistics.median(values),
            'p95_ms':1000*ordered[math.ceil(len(ordered)*.95)-1], 'max_ms':1000*max(values)}

def check_normalization():
    fixtures = [
        '```python\nprice = 1000\nif price != 0:\n    print("ㅋㅋ ㅠㅠ", price / 2)\n```',
        'total_price = 1_000 + 2.5\nprint(total_price)\n',
        '예제 `fruit_basket[0] != "ㅋㅋ"`는 바꾸지 마세요ㅠㅠ',
        '안 된다고 했어요. 못 바꾸고 하지 않는 조건은 아니라고 했어요ㅋㅋ',
        '금리는 -2.5%, 원금은 1,000원, 12개월이며 0으로 나누면 안 돼요ㅠㅠ',
        '조건은 a != b, x <= 3, y >= -1, 1 / 2, ==, and, not, or입니다.',
        '금리ㅋㅋ비교ㅠㅠ 요청',
        'Ｐｙｔｈｏｎ　１２３은 전각 예시입니다. `ｘ = １２`는 코드입니다.',
        'ㅋㅋㅋ',
    ]
    rows = []
    for source in fixtures:
        for mode,nfkc in [('space_jamo',False),('limited_nfkc',True)]:
            dest = normalize_view(source,nfkc)
            protected = [s for flag,s in protected_segments(source) if flag]
            assert all(s in dest for s in protected)
            assert CODE_ID.findall(source) == CODE_ID.findall(dest) or 'Ｐｙｔｈｏｎ' in source
            assert NUMBER.findall(source) == NUMBER.findall(dest) or '１２３' in source
            assert re.findall(r'!=|<=|>=|==|\bnot\b|\band\b|\bor\b|안|못|않|아니',source) == re.findall(r'!=|<=|>=|==|\bnot\b|\band\b|\bor\b|안|못|않|아니',dest)
            rows.append({'mode':mode,'input':source,'normalized':dest,'protected_spans_unchanged':True})
    assert baseline_tokens('금리ㅋㅋ비교') == baseline_tokens(normalize_view('금리ㅋㅋ비교'))
    return rows

def run(args):
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT/'experiments/.cache').resolve()):
        raise ValueError('Output must be in existing development cache')
    if output.exists():
        raise FileExistsError('Never overwrite existing diagnostic')
    wall_start,cpu_start = time.perf_counter(),time.process_time()
    tracked_start = git('diff','--binary','HEAD')
    chunks = load_chunks(ROOT/'kb')
    texts = [c.text for c in chunks]
    team_source = git('show',f'{TEAM_REF}:{TEAM_PATH}')
    normalize_source = git('show',f'{TEAM_REF}:{TEAM_NORMALIZE_PATH}')
    team_terms,new_terms = frozen_dictionary(chunks,team_source)
    normalizer_scope = {'__builtins__':__builtins__}
    # Execute only the inspected pure normalizer module from the team branch.
    exec(compile(normalize_source,TEAM_NORMALIZE_PATH,'exec'),normalizer_scope)
    legacy_normalize = normalizer_scope['collapse_korean_obfuscation']
    validation = {'fixtures':check_normalization(), 'new_dictionary':new_terms}
    sources, datasets = {}, {}
    for name,path in SOURCES.items():
        file = ROOT/path
        all_rows = [json.loads(s) for s in file.read_text().splitlines()]
        rows = [r for r in all_rows if len(r['labels'].get('evidence') or [])==1]
        assert len(rows)==EXPECTED[name], (name,len(rows))
        assert all(r.get('endpoint','tutor')=='tutor' for r in rows)
        datasets[name] = rows
        sources[name] = {'path':path,'sha256':sha(file.read_bytes()),'total_rows':len(all_rows),'single_evidence_rows':len(rows)}
        derivation = file.parent/'derivation.json'
        if derivation.exists():
            sources[name]['derivation'] = json.loads(derivation.read_text())
            if sources[name]['derivation']['dataset_sha256'] != sources[name]['sha256']:
                raise ValueError('Derived dataset hash differs: '+name)
        elif name != 'public119':
            raise ValueError('Missing derived provenance: '+name)
    assert sources['kb55']['derivation'].get('curation'), 'KB55 must be curated'
    baselines, results, predictions, change_cases, audit = {}, {}, [], [], []
    preparation = {}
    modes = ['baseline','space_jamo','limited_nfkc','kiwi_team','kiwi_five_preserve']
    for mode in modes:
        prep_wall,prep_cpu = time.perf_counter(),time.process_time()
        kiwi = None
        if mode == 'baseline':
            view,tokenize = lambda s:s,baseline_tokens
        elif mode in {'space_jamo','limited_nfkc'}:
            view = lambda s: normalize_view(s,mode=='limited_nfkc')
            tokenize = lambda s: baseline_tokens(view(s))
        elif mode == 'kiwi_team':
            kiwi = Kiwi(num_workers=1)
            for term in team_terms:
                kiwi.add_user_word(term.replace(' ',''),'NNP',0)
            view = legacy_normalize
            def tokenize(s):
                cleaned = view(s)
                out = [t.form.strip().lower() for t in kiwi.tokenize(cleaned) if not t.tag.startswith('S') and t.form.strip()]
                for value in OLD_CODE.findall(cleaned):
                    if value.lower() not in out:
                        out.append(value.lower())
                return out
        else:
            kiwi = Kiwi(num_workers=1)
            for term in FIVE_TERMS:
                kiwi.add_user_word(term,'NNP',0)
            view = lambda s: normalize_view(s,True)
            def tokenize(s):
                cleaned = view(s)
                # Keep numbers/Latin terms (SN,SL), negation and particles. Discard
                # only punctuation and symbol tags, never every S-prefixed tag.
                out = [t.form.strip().lower() for t in kiwi.tokenize(cleaned,normalize_coda=True)
                       if t.tag not in {'SF','SP','SS','SSO','SSC','SE','SO','SW'} and t.form.strip()]
                # Separate namespace preserves exact identifiers/decimal amounts;
                # duplicate query counts never multiply BM25 contributions.
                out.extend('code:'+m.group().lower() for m in CODE_ID.finditer(cleaned))
                out.extend('num:'+m.group() for m in NUMBER.finditer(cleaned))
                return out
        index = FixedBM25(texts,tokenize)
        preparation[mode] = {'wall_s':time.perf_counter()-prep_wall,'cpu_s':time.process_time()-prep_cpu,
                             'vocabulary':len(index.idf),'kb_changed_views':sum(view(t)!=t for t in texts)}
        if mode=='baseline':
            exact = BM25(texts)
            assert index.tf==exact.tf and index.idf==exact.idf
            for query in ['금리 비교','for i in range(3): print(i)','ㅋㅋㅋ','예금이 아닌 적금']:
                assert index.scores(query)==exact.scores(query)
            validation['baseline_exact_formula_and_token_equivalence'] = True
        if mode.startswith('kiwi'):
            for text in ['fruit_basket[0] != 0 and not done', '금리는 -2.5%, 원금은 1,000원. 안 돼요. 못 해요. 하지 않는 것은 아니에요.', 'HelpyCode kkumtle.add_food("ㅋㅋ")']:
                tt = tokenize(text)
                audit.append({'mode':mode,'text':text,'tokens':tt})
                if mode=='kiwi_five_preserve':
                    assert all('code:'+m.lower() in tt for m in CODE_ID.findall(text))
                    assert all('num:'+m in tt for m in NUMBER.findall(text))
        results[mode] = {}
        for name,rows in datasets.items():
            wall,cpu = [],[]
            metric = {'count':len(rows),'top1':0,'top3':0,'changed_query_views':0,'changed_token_streams':0,
                      'top1_gains':0,'top1_losses':0,'top3_gains':0,'top3_losses':0,'top1_changed':0}
            for row in rows:
                query = '\n'.join(m['content'] for m in row['messages'])
                q_hash = sha(query.encode())
                # Cold first query is reported; no repeated query/result caching.
                start,cp = time.perf_counter(),time.process_time()
                scores = index.scores(query)
                ranked = ranked_sections(scores,chunks)
                wall.append(time.perf_counter()-start); cpu.append(time.process_time()-cp)
                gold = row['labels']['evidence'][0]
                hit1,hit3 = gold in ranked[:1],gold in ranked[:3]
                metric['top1']+=hit1; metric['top3']+=hit3
                metric['changed_query_views']+=view(query)!=query
                metric['changed_token_streams']+=tokenize(query)!=baseline_tokens(query)
                record = {'dataset':name,'id':row['id'],'mode':mode,'query_sha256':q_hash,
                          'gold':gold,'top3':ranked[:3],'top1_hit':hit1,'top3_hit':hit3,
                          'cpu_ms':1000*cpu[-1],'wall_ms':1000*wall[-1]}
                key = (name,row['id'])
                if mode=='baseline':
                    baselines[key] = record
                else:
                    base = baselines[key]
                    metric['top1_changed'] += ranked[:1]!=base['top3'][:1]
                    flags = {'top1_gains':hit1 and not base['top1_hit'],'top1_losses':base['top1_hit'] and not hit1,
                             'top3_gains':hit3 and not base['top3_hit'],'top3_losses':base['top3_hit'] and not hit3}
                    for flag,value in flags.items(): metric[flag]+=value
                    if any(flags.values()):
                        change_cases.append({**record,'baseline_top3':base['top3'],'changes':[f for f,v in flags.items() if v],
                                             'query':query,'normalized_query':view(query)})
                predictions.append(record)
            metric['query_wall'] = timing(wall); metric['query_cpu'] = timing(cpu)
            results[mode][name] = metric
        print(mode,{n:(v['top1'],v['top3']) for n,v in results[mode].items()},flush=True)
        del tokenize,index,kiwi
        gc.collect()
    validation['token_audit'] = audit
    assert git('diff','--binary','HEAD')==tracked_start, 'Tracked worktree changed during diagnostic; inspect concurrency'
    manifest = {'scope':'offline_existing_development_retrieval_not_holdout','sources':sources,
                'script_sha256':sha(Path(__file__).read_bytes()), 'git_head':git('rev-parse','HEAD').strip(),
                'team_ref':TEAM_REF,'team_commit':git('rev-parse',TEAM_REF).strip(),
                'team_source_sha256':sha(team_source.encode()),'team_normalize_sha256':sha(normalize_source.encode()),
                'kb_files':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted((ROOT/'kb').rglob('*')) if p.is_file()},
                'chunk_count':len(chunks),'chunk_content_sha256':sha(json.dumps([(c.section_id,c.text) for c in chunks],ensure_ascii=False).encode()),
                'versions':{p:importlib.metadata.version(p) for p in ['kiwipiepy','kiwipiepy_model']},
                'python':sys.version,'platform':platform.platform(),'num_workers':1,
                'bm25':{'k1':1.5,'b':.75,'chunk_top':10,'section_evaluation_k':[1,3],'positive_only':True},
                'query':'Full dialogue in original order joined with newline; no truncation; original roles/contents untouched',
                'preparation':preparation,'new_dictionary':new_terms,'legacy_team_dictionary':list(team_terms),
                'notes':['Legacy Kiwi reconstructed from team _tokenize default expand_aliases=False; not guaranteed exact earlier throwaway diagnostic.',
                         'No embeddings or semantic similarity API; no HTTP or answer generation.',
                         'All modes rebuild same chunks with own tokenizer; full original dialogue query scope held fixed.',
                         'Five-term new variant bundles dictionary/normalization/code-number preservation; not a causal decomposition.',
                         'Single-process CPU timings on developer machine, not a 64-concurrent Linux deployment benchmark.',
                         'All datasets already inspected, overlapping derivations; do not pool as independent test samples.',
                         'Normalization only creates a search view. Guard classification and raw messages must remain unchanged.',
                         'Zero-signal normalized views are allowed here; production use would preserve original classification.'],
                'total_wall_s':time.perf_counter()-wall_start,'total_cpu_s':time.process_time()-cpu_start}
    summary = ['# Offline normalization / Kiwi retrieval screen','','Each cell is top1 / top3 distinct-section hits. Same BM25 top10 chunks.','',
               '| Dataset | N | Baseline | Space/jamo | Limited NFKC | Team Kiwi | Five-term Kiwi |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for name in SOURCES:
        summary.append('| '+name+' | '+str(EXPECTED[name])+' | '+' | '.join(f"{results[m][name]['top1']} / {results[m][name]['top3']}" for m in modes)+' |')
    summary.extend(['','No API calls. Existing development data, not holdout. See manifest for hashes, definitions and timing limitations.'])
    output.mkdir(exist_ok=False)
    for name,data in [('manifest',manifest),('result',results),('validation',validation)]:
        (output/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for name,data in [('predictions',predictions),('changed_cases',change_cases)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    (output/'summary.md').write_text('\n'.join(summary)+'\n')
    (output/'script.py').write_bytes(Path(__file__).read_bytes())
    print('\n'.join(summary)); print('OUTPUT',str(output))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'experiments/.cache/jh_kiwi_r3_normalization1'))
    run(parser.parse_args())
