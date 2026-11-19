"""Apply scope status only on the numeric path, preserving other tutor checks."""
from .round7_answer import ConcreteApply
from .round7_material_scope import ScopedConcrete
from .round7_material_verdict import MaterialVerdict
from .round7_compact_calculator import CompactCalculator


class NumericMaterialVerdict(MaterialVerdict):
    tutor_numeric_material_verdict=True

    def _scope_needed(self,state):
        return ConcreteApply._numeric_request(self,state)


class NumericVerdictCompact(NumericMaterialVerdict,CompactCalculator):
    pass


class NumericVerdictDirect(NumericMaterialVerdict,ScopedConcrete):
    pass
