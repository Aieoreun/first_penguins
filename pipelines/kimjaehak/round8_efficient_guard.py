"""Cross B with prior grounding schedules; isolate OTHER reasoning effort."""
from .round7_binary_guard import BinaryGuard
from .round7_adaptive_ground import GroundThree, AdaptiveGround


class BinaryGroundThree(BinaryGuard,GroundThree):
    pass


class BinaryAdaptiveGround(BinaryGuard,AdaptiveGround):
    pass


class BinaryOtherMinimal(BinaryGuard):
    guard_other_effort='minimal'

    async def _ask(self,state,name,system,user,effort,timeout):
        if getattr(state,'_material_budget_guard',False) is True and name=='other':
            effort=self.guard_other_effort
        return await super()._ask(state,name,system,user,effort,timeout)
