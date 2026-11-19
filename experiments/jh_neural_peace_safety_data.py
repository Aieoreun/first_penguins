"""Pre-prediction public crossed-output diagnostic; never training/runtime data."""
from copy import deepcopy
import hashlib
import json

PUBLIC_SHA = '4a585c174f15dec205a286ec45e8c14ff28b01bbcedde3097072ca52e3b0c426'
# Explicit harmful endorsement, privacy invasion, insult, or claimed rule bypass.
# Exclude context-dependent off_topic/academic completions and phishing examples.
UNSAFE_IDS = ('pub-6b392a59','pub-1499aec9','pub-3dfcc677','pub-33eab8bb',
              'pub-8af14903','pub-3f1f5041','pub-032699fa','pub-15a22837')


def load(root):
    path=root/'public/guard.jsonl'
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=PUBLIC_SHA: raise ValueError('Public source changed')
    public=[json.loads(line) for line in data.splitlines()]
    by_id={x['id']:x for x in public}
    normal=sorted((x for x in public if x['labels']['category']=='normal'
                   and x.get('candidate_answer') and x['labels']['safe'] is True
                   and x['labels']['grounded'] is True),key=lambda x:x['id'])[:len(UNSAFE_IDS)]
    items=[]
    for question,unsafe_id in zip(normal,UNSAFE_IDS,strict=True):
        unsafe=by_id[unsafe_id]
        if unsafe['labels']['safe'] is not False: raise ValueError('Unsafe parent label changed')
        for kind in ('control','unsafe'):
            item=deepcopy(question)
            item.update(id='cross/'+question['id']+'/'+kind,source_id=question['id'],
                        answer_source_id=question['id'] if kind=='control' else unsafe_id)
            if kind=='unsafe':
                item['candidate_answer']=unsafe['candidate_answer']
                # Adapter placeholders ONLY. Ground/span/total are NOT interpreted.
                item['labels'].update(safe=False,grounded=False,unsupported=[])
            items.append(item)
    encoded=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items)
    return items,{'dataset':'cross_safety','scope':'public_crossed16_safety_only_development',
        'primary_metric':'safe_accuracy','count':len(items),'candidate_count':len(items),
        'source_files':{'public/guard.jsonl':PUBLIC_SHA,
            'experiments/jh_neural_peace_safety_data.py':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()},
        'dataset_sha256':hashlib.sha256(encoded.encode()).hexdigest(),
        'independent_source_holdout':False,'training_allowed':False,'deployment_allowed':False,
        'review':'Pre-prediction manual selection: eight normal grounded-safe controls and eight explicit harmful/bypass outputs; no new model-generated labels.',
        'interpretation':'Safety and input diagnostic only. Grounding/span labels for crossed outputs are adapter placeholders: never interpret aggregate guard or grounding scores. Public-derived, not independent or organizer gold.'}
