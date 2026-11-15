"""Keep v4 generation/deadlines and substitute only the measured output guard."""
from pipelines.v4.pipeline import Pipeline as V4
from .guided import GuidedOutput


class Pipeline(V4, GuidedOutput):
    # V4 owns tutor/guard deadlines, generation prompts and classification priority.
    # GuidedOutput supplies _check_output before the shared V2 base in the MRO.
    pass
