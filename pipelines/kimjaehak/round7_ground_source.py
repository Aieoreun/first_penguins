"""Guard grounding source ablations without changing input retrieval or tutor."""
from .classifier_matrix import CharMLP50


class InputGroundSource(CharMLP50):
    guard_ground_context = 'input_top1'

    def _ground_payload(self, state):
        original = super()._ground_payload(state)
        if getattr(state, '_material_budget_guard', False) is not True:
            return original
        query = '\n'.join(message['content'] for message in state.messages)
        found = self.searcher.search(query, bm25_top=1, embedding_top=0)
        if not found:
            return original
        ids = [found[0].section_id]
        if self.guard_ground_context == 'candidate_input_union':
            ids = list(dict.fromkeys([r['section'] for r in original['교재']] + ids))[:2]
        materials = [{'section':sid,'text':self.searcher.section_text(sid)} for sid in ids]
        state.debug['r7_ground_source_changed'] = materials != original['교재']
        return {**original, '교재': materials}


class UnionGroundSource(InputGroundSource):
    guard_ground_context = 'candidate_input_union'
