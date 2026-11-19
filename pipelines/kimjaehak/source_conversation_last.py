"""Place the unchanged conversation after materials in source-selection JSON."""
from .focused_excerpt import Selected as SelectedFocused


class SourceConversationLast(SelectedFocused):
    source_conversation_last = True

    def _source_payload(self, state, ids):
        original = super()._source_payload(state, ids)
        return {'materials': original['materials'],
                'conversation': original['conversation']}
