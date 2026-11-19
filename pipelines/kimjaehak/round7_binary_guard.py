"""Guard-only binary/multiclass factorization; tutor and output remain A."""
from .classifier_matrix import CharMLP50
from .classifier_models import CategoryModel,public_data
from .round7_binary_model import BinaryAdjustedModel,BLOCKED


class BinaryGuard(CharMLP50):
    classifier_binary_weight=1.0
    # Public grouped OOF top-50% nonharmful coverage; not calibrated confidence.
    classifier_threshold=.8776460484899311

    def __init__(self,searcher,llm):
        super().__init__(searcher,llm)
        rows,texts,_,_=public_data()
        labels=['block' if r['labels']['category'] in BLOCKED else 'allow' for r in rows]
        decision=CategoryModel('char_mlp').fit(texts,labels)
        self.matrix_model=BinaryAdjustedModel(self.matrix_model,decision,self.classifier_binary_weight)
