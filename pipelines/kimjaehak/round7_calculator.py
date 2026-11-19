"""Tutor-only arithmetic expressions; interpretation still belongs to the LLM."""
import ast
from decimal import Decimal, localcontext, ROUND_CEILING
import re
from .round7_material_scope import ScopedConcrete, MaterialScope
from .round7_answer import ConcreteApply


def calculate(expression):
    if not isinstance(expression,str) or not 0<len(expression)<=256:
        raise ValueError('Invalid expression size')
    tree=ast.parse(expression,mode='eval')
    if len(list(ast.walk(tree)))>96:raise ValueError('Expression too complex')
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            # Read the source token, not the binary float held by ast.
            token=ast.get_source_segment(expression,node)
            if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',token):
                raise ValueError('Only decimal constants')
            value=Decimal(token)
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=visit(node.operand)
            if isinstance(node.op,ast.USub):value=-value
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            left,right=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):value=left+right
            elif isinstance(node.op,ast.Sub):value=left-right
            elif isinstance(node.op,ast.Mult):value=left*right
            else:value=left/right
        elif (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and node.func.id=='ceil' and len(node.args)==1 and not node.keywords):
            value=visit(node.args[0]).to_integral_value(rounding=ROUND_CEILING)
        else:raise ValueError('Unsupported expression')
        if not value.is_finite() or abs(value)>Decimal('1e15'):
            raise ValueError('Unbounded number')
        return value
    with localcontext() as context:
        context.prec=28
        value=visit(tree.body)
        # No silent monetary rounding: retain finite decimal results, and reject
        # long fractions unless the model explicitly asks for ceil (whole months).
        result=format(value,'f').rstrip('0').rstrip('.') if '.' in format(value,'f') else format(value,'f')
        if len(result)>32 or ('.' in result and len(result.split('.')[1])>8):
            raise ValueError('Unspecified rounding')
        return result


class CalculatorApply(ScopedConcrete):
    tutor_apply_calculator=True
    tutor_apply_system=ScopedConcrete.tutor_apply_system+"""
이번에는 수치 계산 결과를 직접 쓰지 말고 계산식으로 제출하세요. 별도 제한식 계산기가 값을 계산합니다.
원금·월 납입액·연 이율·기간의 단위를 구별하고, 질문의 조건에 교재의 명시적 규칙만 적용하세요.
허용 식은 십진수, 괄호, + - * / 와 정수 기간 올림 ceil(식)뿐입니다. 변수·코드·함수·단위·쉼표는 식에 넣지 마세요.
정수 기간이 필요한 목표 달성 문제에만 ceil을 사용하세요. 반올림·버림 규칙이 없는 금전 결과를 임의로 반올림하지 마세요.
answer에는 결과 대신 {{0}}, {{1}}처럼 calculations 순번을 넣고 숫자는 자리표시자 밖에 쓰지 마세요.
calculations는 최대 4개이며 각 항목의 expression은 계산식, evidence는 계산 규칙이 있는 교재의 연속된 원문 구절입니다.
질문에서 요구한 결과와 반드시 필요한 중간값만 계산하세요. 교재에 없는 세금·수수료·정책은 계산하지 마세요.
최종 JSON: {"supported":true,"reason":"","answer":"계산 결과는 {{0}}원입니다.","calculations":[{"expression":"허용 계산식","evidence":"교재 원문"}]}.
계산 근거가 부족하면 supported=false와 기존 reason을 쓰고 answer는 빈 문자열, calculations는 빈 목록입니다.
"""

    def _attach_explanation(self,state,obj):
        if obj.get('supported') is not True:
            return super()._attach_explanation(state,obj)
        template=obj.get('answer'); calculations=obj.get('calculations')
        if not isinstance(template,str) or not isinstance(calculations,list) or not 1<=len(calculations)<=4:
            raise ValueError('Invalid calculation template')
        placeholders=re.findall(r'\{\{([0-3])\}\}',template)
        remainder=re.sub(r'\{\{[0-3]\}\}','',template)
        if re.search(r'[0-9{}]',remainder) or set(placeholders)!={str(i) for i in range(len(calculations))}:
            raise ValueError('Unverified numeric text')
        material=self.searcher.section_text(state.section)
        answer=template
        for i,row in enumerate(calculations):
            if (not isinstance(row,dict) or not isinstance(row.get('evidence'),str)
                    or len(row['evidence'].strip())<4 or row['evidence'] not in material):
                raise ValueError('Missing verbatim rule evidence')
            answer=answer.replace('{{'+str(i)+'}}',calculate(row.get('expression')))
        ConcreteApply._attach_explanation(self,state,{**obj,'answer':answer})
        state.debug['r7_calculator_completed']=True


class MaterialCalculator(MaterialScope,CalculatorApply):
    """Also apply the independently tested material-scope input rule."""
    pass
