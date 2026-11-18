"""Independent tutor ablations inspired by the team's 2026-11-18 branches.

bsuk1 de664f2: short evidence in material checks.
seungwon f3e320b / vx: remove unused material section output.
Peace 2f51c1f: evaluate dependence on trained source routing (narrowed here).
"""
from core.state import Chunk
from core.schema import Label
from pipelines.v2 import prompts
from .dual_granularity import All as DualAll
from .material_budget import Pipeline as MaterialBudget


def body_text(chunk):
    lines = chunk.text.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
    if lines and lines[0].startswith('## '):
        lines = lines[1:]
    return '\n'.join(lines).strip()


class ShortMaterial(DualAll):
    tutor_short_material = 4

    def _short_material_chunks(self, state):
        original = list(state.chunks)
        found = list(original)
        added = 0
        routes = getattr(state, '_dual_routes', ())
        short = routes[2] if len(routes) == 3 else ()
        for child in short:
            body = body_text(child)
            if not body or any(c.section_id == child.section_id and body in c.text for c in found):
                continue
            found.append(Chunk(child.section_id, child.text))
            added += 1
            if added >= self.tutor_short_material:
                break
        return found

    async def _collect_checks(self, state, check, material):
        if not getattr(state, '_material_budget_guard', True):
            material = prompts.material_user(self._short_material_chunks(state), state.request,
                                             state.history, state.last_user)
        return await super()._collect_checks(state, check, material)


def without_section(system):
    start = '4. 근거 섹션(section):'
    field = '"section": "<섹션 ID>", '
    if system.count(start) != 1 or system.count(field) != 1:
        raise ValueError('Unexpected material prompt schema')
    before, after = system.split(start, 1)
    return (before.rstrip('\n') + '\n' + after.split('\n', 1)[1]).replace(field, '')


MATERIAL_NO_SECTION = without_section(prompts.MATERIAL_SYSTEM)


class NoSection(MaterialBudget):
    tutor_material_no_section = True

    async def _ask(self, state, name, system, user, effort, timeout):
        if name == 'material' and not getattr(state, '_material_budget_guard', True):
            system = MATERIAL_NO_SECTION
        return await super()._ask(state, name, system, user, effort, timeout)


class ContextVerify(MaterialBudget):
    tutor_verify_context_sources = True

    async def _generate(self, state):
        if (not getattr(state, '_material_budget_guard', True) and len(state.messages) > 1
                and state.label in {Label.NORMAL, Label.PII_DISCLOSURE}):
            # Preserve the learned source as a candidate; only bypass its skip
            # of the ordinary selector. No extra call on single-turn requests.
            for key in ('learned_section', 'tutor_section_gate'):
                state.debug.pop(key, None)
        return await super()._generate(state)
