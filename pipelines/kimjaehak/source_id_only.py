"""Request only a source ID; retain normal nonstreaming full-JSON parsing."""
from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused


_ORIGINAL_SCHEMA = 'JSON 하나: {"section":"제공된 섹션 ID 하나", "reason":"선택 이유를 짧은 한 문장으로"}.\n'
_ID_SCHEMA = 'JSON 하나: {"section":"제공된 섹션 ID 하나"}.\n'
_BASE_SYSTEM = SelectedFocused.source_extract_system
if not _BASE_SYSTEM.endswith(_ORIGINAL_SCHEMA):
    raise ValueError('Source schema changed; review ID-only prompt before use')


class Pipeline(SelectedFocused):
    source_id_only = True
    source_extract_system = _BASE_SYSTEM[:-len(_ORIGINAL_SCHEMA)] + _ID_SCHEMA
