"""Pinned, existing public-derived guard diagnostics; never runtime training data."""
import hashlib
import json
from pathlib import Path
import random

PUBLIC_SHA = '4a585c174f15dec205a286ec45e8c14ff28b01bbcedde3097072ca52e3b0c426'
DATASETS = {
    'relational': {
        'path': 'experiments/.cache/jh_guard_relational_v2/inputs.jsonl',
        'proposal': 'experiments/.cache/jh_guard_relational_v2/proposal.json',
        'manifest': 'experiments/.cache/jh_guard_relational_v2/manifest.json',
        'manifest_scope': 'kb_manual_relational_guard_development',
        'review': 'docs/jh/RELATIONAL_REVIEW.json',
        'raw_count': 16, 'count': 16, 'candidate_count': 16,
        'files': {
            'experiments/.cache/jh_guard_relational_v2/inputs.jsonl': '564a771a7ec9e634b30da7007a172d72b7d030003e95c7177629bcc9565a8330',
            'experiments/.cache/jh_guard_relational_v2/proposal.json': '9db3af3575a627389b42867e8d782d161c672b81b9abdd8dc793579e08d98501',
            'experiments/.cache/jh_guard_relational_v2/manifest.json': '46b69a010ef1344ee343e60498494b63c27fe1b86b4134b77e8fd0669a9c74a8',
            'docs/jh/RELATIONAL_REVIEW.json': '7362bcd929f3869b8ea54a469ab6ca2fdf1952e3ecdcfb64de47fbef585c6234',
        },
        'scope': 'kb_relational16_reviewed_grounding_development',
        'primary_metric': 'paired_grounding', 'old_input_predictions_fixed': False,
        'interpretation': 'Reviewed hand-written KB-derived relational development data. Whole-set grounded pair metrics are primary; code/course groups and official guard/span scores are supplementary. Not an independent holdout; training and deployment prohibited.',
    },
    'counterfactual': {
        'path': 'experiments/.cache/jh_guard_counterfactual_v1/inputs.jsonl',
        'manifest': 'experiments/.cache/jh_guard_counterfactual_v1/manifest.json',
        'manifest_scope': 'public_guard_counterfactual16_manual_grounding_development',
        'review': 'docs/jh/COUNTERFACTUAL_REVIEW.json',
        'raw_count': 16, 'count': 16, 'candidate_count': 16,
        'files': {
            'public/guard.jsonl': PUBLIC_SHA,
            'experiments/.cache/jh_guard_counterfactual_v1/inputs.jsonl': 'fe90fb26e9442429db4c1c86389cff574bb4f9517f1ccf64260eed2300d689d3',
            'experiments/.cache/jh_guard_counterfactual_v1/manifest.json': '2bd5d88976e93ff30ad7565fa4a83a666eb02860b921506dfdaa8c367d835559',
            'docs/jh/COUNTERFACTUAL_REVIEW.json': '5ace08fa01e64b7685aada627ac4f4edcfebd96f7a20b03223646283c046280f',
        },
        'scope': 'public_guard_counterfactual16_reviewed_grounding_development',
        'primary_metric': 'paired_grounding', 'old_input_predictions_fixed': False,
        'interpretation': 'Reviewed manual counterfactual development data with public parents. Grounded control/detection/pair accuracy is primary; official guard/span scores are supplementary, and span boundaries are subjective. Not an independent holdout; training and deployment prohibited.',
    },
    'alternative': {
        'path': 'experiments/.cache/jh_transfer_alternative_build1/derived_inputs.jsonl',
        'manifest': 'experiments/.cache/jh_transfer_alternative_build1/derivation.json',
        'manifest_scope': 'public_paraphrase_input_diagnostic',
        'generation_manifest': 'experiments/.cache/jh_augment_public2_1/manifest.json',
        'raw_count': 365, 'count': 184, 'candidate_count': 0,
        'files': {
            'public/guard.jsonl': PUBLIC_SHA,
            'public/tutor.jsonl': '8f8a74a7747281d89b27cb972169f95401b1082ebfa99dce72af55e0f6dc31e0',
            'experiments/.cache/jh_transfer_alternative_build1/derived_inputs.jsonl': 'e92d84b39aa79bb457db1106a929642137a3b879c5b496d30439982cb392798b',
            'experiments/.cache/jh_transfer_alternative_build1/derivation.json': '425f608e65140a4ca784c3f2856a8bd68d341b2cd2ee562234f7fe169d37ab7b',
            'experiments/.cache/jh_augment_public2_1/paraphrases.jsonl': '5113975eb7da88ffde347105550e31834b09e28645c2b2aceb5f4f3a16f6f8c2',
            'experiments/.cache/jh_augment_public2_1/manifest.json': '49a497875888638982a0da2cc40a82d8614d594f2bedd6262a4a2752ae52a8a6',
        },
        'scope': 'public_guard_alternative184_input_only_development',
        'primary_metric': 'input_only', 'old_input_predictions_fixed': False,
        'interpretation': 'Input-only category/decision diagnostic; no candidate answers or output labels. Alternate generated expressions share public training parents, inherit noisy labels, and are not an independent holdout. Previously used in tutor development.',
    },
    'public': {
        'path': 'public/guard.jsonl', 'raw_count': 200, 'count': 200, 'candidate_count': 86,
        'files': {'public/guard.jsonl': PUBLIC_SHA},
        'scope': 'public_guard200_comparison_not_submission_total',
        'primary_metric': 'guard', 'old_input_predictions_fixed': False,
    },
    'novel': {
        'path': 'experiments/.cache/jh_guard_transfer_jh_context_policy_novel1/inputs.jsonl',
        'manifest': 'experiments/.cache/jh_guard_transfer_jh_context_policy_novel1/manifest.json',
        'manifest_scope': 'public_guard_answer_paraphrase_component',
        'generation_manifest': 'experiments/.cache/jh_guard_paraphrases_novel1/manifest.json',
        'curation': 'experiments/.cache/jh_guard_paraphrases_novel1/curation.json',
        'raw_count': 60, 'count': 60, 'candidate_count': 60,
        'files': {
            'public/guard.jsonl': PUBLIC_SHA,
            'experiments/.cache/jh_guard_transfer_jh_context_policy_novel1/inputs.jsonl': 'a9fd246418ec4b4a203b1ec3d02a579cda14104fece3a43b9fda13930a129dec',
            'experiments/.cache/jh_guard_transfer_jh_context_policy_novel1/manifest.json': '908cbb3f5fb59308e5311d35700e2585fff7fd0e4bd0a210c8242ded11b89188',
            'experiments/.cache/jh_guard_paraphrases_novel1/paraphrases.jsonl': '047a86b144b8f528289b78190da1d5e336aabd33c77bc7e0a788e33474a920a6',
            'experiments/.cache/jh_guard_paraphrases_novel1/curation.json': '873454d1743bf94fb1e6ceb7d3194930c3d1a76440976b8754598dc198d73e23',
            'experiments/.cache/jh_guard_paraphrases_novel1/manifest.json': '91ce09a45cd3759b00df524c9b68c6ccb8fe7446b3fc34a0897ac8f98b2762ff',
        },
        'scope': 'public_guard_novel_answer60_full_pipeline_development',
        'primary_metric': 'guard', 'old_input_predictions_fixed': True,
    },
    'refusal': {
        'path': 'experiments/.cache/jh_refusal_probe_baseline1/inputs.jsonl',
        'manifest': 'experiments/.cache/jh_refusal_probe_baseline1/manifest.json',
        'manifest_scope': 'public_refusal_safety_component',
        'generation_manifest': 'experiments/.cache/jh_refusals_safe1/manifest.json',
        'curation': 'experiments/.cache/jh_refusals_safe1/curation.json',
        'raw_count': 30, 'count': 30, 'candidate_count': 30,
        'files': {
            'public/guard.jsonl': PUBLIC_SHA,
            'experiments/.cache/jh_refusal_probe_baseline1/inputs.jsonl': '7f37f291da3aa4ce53f4b9bdf85f0ef7414c286153aeb49bb5d3e56b6e498d29',
            'experiments/.cache/jh_refusal_probe_baseline1/manifest.json': '8335b1079c7b4c71a41d1aa3c6a6525ff398f8f2082d8416805b114a0010f466',
            'experiments/.cache/jh_refusals_safe1/refusals.jsonl': '2bb68f0d4d27e67678fc8cd7f69e7ba2ec1e69de5eb693d7838b140106a2c563',
            'experiments/.cache/jh_refusals_safe1/curation.json': '327124a27e9097185cb8328dda4c656810fdf421202e657d309c5e597545ba96',
            'experiments/.cache/jh_refusals_safe1/manifest.json': '5dee65fb205666fcd41c3bbcfb9dad6d47f28abefb7df57007a89a05b99c658f',
        },
        'scope': 'public_guard_refusal30_safety_diagnostic_full_pipeline',
        'primary_metric': 'safe_accuracy', 'old_input_predictions_fixed': True,
        'interpretation': 'Safety only: grounding=True/spans=[] were forced by the old probe; aggregate guard/grounding/span scores are not independent quality evidence.',
    },
    'paraphrase': {
        'path': 'experiments/.cache/jh_transfer_full_adaptive1/derived_inputs.jsonl',
        'manifest': 'experiments/.cache/jh_transfer_full_adaptive1/derivation.json',
        'manifest_scope': 'public_paraphrase_full_diagnostic',
        'generation_manifest': 'experiments/.cache/jh_augment_public2_1/manifest.json',
        'raw_count': 388, 'count': 196, 'candidate_count': 85,
        'files': {
            'public/guard.jsonl': PUBLIC_SHA,
            'public/tutor.jsonl': '8f8a74a7747281d89b27cb972169f95401b1082ebfa99dce72af55e0f6dc31e0',
            'experiments/.cache/jh_transfer_full_adaptive1/derived_inputs.jsonl': '9bf7a76449ffeb76f1bcd236523220b30c13b62afc5d14f5a76e7c3c2b7d429b',
            'experiments/.cache/jh_transfer_full_adaptive1/derivation.json': '000c5e37299595a807dbc449e15f7593fd9d3953590665efb3e565808d5f4510',
            'experiments/.cache/jh_augment_public2_1/paraphrases.jsonl': '5113975eb7da88ffde347105550e31834b09e28645c2b2aceb5f4f3a16f6f8c2',
            'experiments/.cache/jh_augment_public2_1/manifest.json': '49a497875888638982a0da2cc40a82d8614d594f2bedd6262a4a2752ae52a8a6',
        },
        'scope': 'public_guard_question_paraphrase196_full_pipeline_development',
        'primary_metric': 'guard', 'old_input_predictions_fixed': False,
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_counterfactual(root, spec, rows, manifest):
    from experiments import jh_guard_counterfactual as counterfactual
    rebuilt, rebuilt_manifest = counterfactual.build_items(root)
    data = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in rebuilt).encode()
    rebuilt_manifest['dataset_sha256'] = digest(data)
    if rows != rebuilt or manifest != rebuilt_manifest:
        raise ValueError('Counterfactual regeneration differs from frozen dataset/manifest')
    review = json.loads((root / spec['review']).read_text())
    parents = [pair['source_id'] for pair in manifest['pairs_provenance']]
    if (review.get('scope') != 'parent_source_review_before_counterfactual_scoring'
            or review.get('accepted') is not True
            or review.get('new_counterfactual_predictions_observed') is not False
            or review.get('training_allowed') is not False or review.get('deployment_allowed') is not False
            or review.get('independent_source_holdout') is not False
            or review.get('accepted_source_ids') != parents
            or any(review.get(key) != manifest[key] for key in
                   ('dataset_sha256', 'proposal_sha256', 'kb_manifest_sha256'))):
        raise ValueError('Counterfactual parent review does not approve these exact sources')


def validate_relational(root, spec, rows, manifest):
    from experiments import jh_guard_relational_diagnostic as relational
    proposals, rebuilt, rebuilt_manifest = relational.build(root)
    data = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in rebuilt).encode()
    rebuilt_manifest['dataset_sha256'] = digest(data)
    stored_proposals = json.loads((root / spec['proposal']).read_text())
    if rows != rebuilt or manifest != rebuilt_manifest or stored_proposals != proposals:
        raise ValueError('Relational regeneration differs from frozen proposal/dataset/manifest')
    review = json.loads((root / spec['review']).read_text())
    if (review.get('scope') != 'parent_source_review_before_relational_scoring'
            or review.get('accepted') is not True
            or review.get('new_relational_predictions_observed') is not False
            or review.get('training_allowed') is not False or review.get('deployment_allowed') is not False
            or review.get('independent_source_holdout') is not False
            or review.get('accepted_pair_ids') != [pair['pair_id'] for pair in proposals]
            or any(review.get(key) != manifest[key] for key in
                   ('dataset_sha256', 'proposal_sha256', 'kb_manifest_sha256'))):
        raise ValueError('Relational parent review does not approve these exact pairs')


def relational_metrics(items, rows):
    from experiments.jh_guard_counterfactual import grounding_metrics
    # Compute the whole set first; never average the unequal-sized groups.
    result = grounding_metrics(items, rows)
    groups = {'code_interpretation', 'course_relation'}
    if any(item.get('error_group') not in groups for item in items):
        raise ValueError('Missing relational diagnostic group')
    result['by_group'] = {}
    for group in sorted(groups):
        subset = [item for item in items if item['error_group'] == group]
        if subset:
            identifiers = {item['id'] for item in subset}
            result['by_group'][group] = grounding_metrics(subset, [row for row in rows if row['id'] in identifiers])
    return result


def validate_rows(rows, spec, name, public_parents=None):
    if len(rows) != spec['raw_count']:
        raise ValueError('Unexpected source dataset count')
    if name == 'alternative':
        if public_parents is None:
            raise ValueError('Alternative needs verified public parent endpoints')
        if len({row.get('source_id') for row in rows}) != len(rows):
            raise ValueError('Alternative contains repeated parents')
        for row in rows:
            parent = public_parents.get(row.get('source_id'))
            if (parent is None or row.get('endpoint') != 'guard' or 'candidate_answer' in row
                    or set(row.get('labels', {})) != {'category', 'decision'}
                    or row['labels']['category'] != parent['labels']['category']
                    or not isinstance(row.get('messages'), list) or not row['messages']
                    or row['messages'][:-1] != parent['messages'][:-1]
                    or row['messages'][-1].get('role') != 'user'):
                raise ValueError('Alternative input-only parent contract changed')
            if parent['endpoint'] == 'guard' and row['labels']['decision'] != parent['labels']['decision']:
                raise ValueError('Alternative guard decision differs from parent')
        items = [row for row in rows if public_parents[row['source_id']]['endpoint'] == 'guard']
    else:
        items = [row for row in rows if row.get('endpoint') == 'guard']
    if (len(items) != spec['count'] or len({row['id'] for row in items}) != len(items)
            or sum('candidate_answer' in row for row in items) != spec['candidate_count']):
        raise ValueError('Unexpected guard subset, IDs or candidate count')
    for item in items:
        labels = item['labels']
        if not isinstance(item.get('messages'), list) or not item['messages']:
            raise ValueError('Missing conversation')
        if not isinstance(labels.get('category'), str) or labels.get('decision') not in ('allow', 'block'):
            raise ValueError('Missing input labels')
        if 'candidate_answer' in item:
            answer = item['candidate_answer']
            spans = labels.get('unsupported')
            if (not isinstance(answer, str) or not isinstance(spans, list)
                    or any(not isinstance(span, str) or not span or span not in answer for span in spans)
                    or type(labels.get('grounded')) is not bool or type(labels.get('safe')) is not bool):
                raise ValueError('Invalid candidate output labels or spans')
        if name == 'refusal' and (labels['decision'] != 'block' or labels['grounded'] is not True
                                   or labels['unsupported'] != []):
            raise ValueError('Refusal diagnostic safety-only contract changed')
    if name == 'refusal':
        groups = {}
        for item in items:
            parent, kind = item['id'].rsplit('/', 1)
            if kind not in ('refusal', 'disclaimer') or item['labels']['safe'] is not (kind == 'refusal'):
                raise ValueError('Invalid refusal/disclaimer label')
            groups.setdefault(parent, set()).add(kind)
        if len(groups) * 2 != len(items) or any(kinds != {'refusal', 'disclaimer'} for kinds in groups.values()):
            raise ValueError('Refusal pairs incomplete')
    return items


def load_dataset(name, root):
    root = Path(root).resolve()
    spec = DATASETS[name]
    for filename, expected in spec['files'].items():
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes()) != expected:
            raise ValueError('Pinned diagnostic source changed: ' + filename)
    if spec.get('manifest'):
        manifest = json.loads((root / spec['manifest']).read_text())
        if (manifest.get('scope') != spec['manifest_scope']
                or manifest.get('dataset_sha256') != spec['files'][spec['path']]
                or manifest.get('count') != spec['raw_count']
                or manifest.get('independent_source_holdout') is not False
                or spec['files'].get(manifest.get('source')) != manifest.get('source_sha256')):
            raise ValueError('Derived diagnostic lineage differs')
        if name in ('novel', 'refusal'):
            generation = json.loads((root / spec['generation_manifest']).read_text())
            curation = json.loads((root / spec['curation']).read_text())
            embedded = {key: value for key, value in manifest['curation'].items() if key not in ('path', 'sha256')}
            if (manifest.get('input_predictions_fixed') is not True
                    or manifest['generation']['source_sha256'] != spec['files']['public/guard.jsonl']
                    or generation != manifest['generation'] or embedded != curation
                    or curation.get('source_sha256') != manifest['source_sha256']
                    or manifest['curation'].get('candidate_scores_observed') is not False):
                raise ValueError('Original component provenance differs')
        elif name in ('paraphrase', 'alternative'):
            generation = json.loads((root / spec['generation_manifest']).read_text())
            if any(spec['files'].get('public/' + name) != checksum
                   for name, checksum in generation['public_hashes'].items()):
                raise ValueError('Paraphrase public parent changed')
            if name == 'alternative' and (manifest.get('paraphrase_partition') != 'alternative'
                                           or manifest.get('inherited_labels') is not True):
                raise ValueError('Alternative partition or inherited-label declaration changed')
    rows = [json.loads(line) for line in (root / spec['path']).read_text().splitlines() if line.strip()]
    if name == 'counterfactual':
        validate_counterfactual(root, spec, rows, manifest)
    elif name == 'relational':
        validate_relational(root, spec, rows, manifest)
    parents = None
    if name == 'alternative':
        parents = {}
        for endpoint in ('guard', 'tutor'):
            for line in (root / 'public' / (endpoint + '.jsonl')).read_text().splitlines():
                parent = json.loads(line)
                parents[parent['id']] = {**parent, 'endpoint': endpoint}
    items = validate_rows(rows, spec, name, parents)
    if name == 'public':
        # Preserve jh_compare.load_public's per-endpoint and final shuffles.
        random.Random(20261115).shuffle(items)
        random.Random(20261115).shuffle(items)
    metadata = {'dataset': name, 'scope': spec['scope'], 'source': spec['path'],
                'source_sha256': spec['files'][spec['path']], 'source_files': dict(spec['files']),
                'raw_count': spec['raw_count'], 'count': len(items),
                'candidate_count': spec['candidate_count'], 'primary_metric': spec['primary_metric'],
                'independent_source_holdout': False, 'input_predictions_fixed': False,
                'historical_component_input_predictions_fixed': spec['old_input_predictions_fixed'],
                'interpretation': spec.get('interpretation',
                    'Full own input/output routing on public-derived development data; not a private score estimate.')}
    if name in ('counterfactual', 'relational'):
        metadata['source_files'].update(manifest['kb_files'])
        metadata['parent_review'] = spec['review']
        metadata['parent_review_sha256'] = spec['files'][spec['review']]
        metadata['generation_review_status'] = manifest['review_status']
        metadata['parent_review_accepted'] = True
    return items, metadata


def summarize_result(label, result, metadata):
    summary = {'candidate': label, 'dataset': metadata['dataset'], 'scope': metadata['scope'],
               'primary_metric': metadata['primary_metric'], 'runtime': result['runtime']}
    if metadata['primary_metric'] == 'paired_grounding':
        details = result['relational_details'] if metadata['dataset'] == 'relational' else result['counterfactual_details']
        summary.update({key: details[key] for key in
                        ('control_accuracy', 'counterfactual_detection', 'paired_both_correct', 'invalid')})
        summary['interpretation'] = metadata['interpretation']
    elif metadata['primary_metric'] == 'input_only':
        summary['input_metrics'] = {key: result['guard_details'][key] for key in
                                    ('block_f1', 'category_accuracy', 'false_block', 'missed_block')}
        summary['interpretation'] = metadata['interpretation']
    elif metadata['primary_metric'] == 'safe_accuracy':
        summary['safe_accuracy'] = result['guard_details']['safe_accuracy']
        summary['interpretation'] = metadata['interpretation']
    else:
        summary.update(guard=result['guard'], guard_details=result['guard_details'])
    return summary
