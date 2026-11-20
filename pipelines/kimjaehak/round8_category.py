"""Keep the binary decision head while varying within-group category features."""
from .round7_binary_guard import BinaryGuard
from .round7_binary_model import BinaryAdjustedModel
from .round7_guard import BoundaryModel
from .classifier_models import public_data
from .neural_peace_guard import InputBudgets


class BinaryBoundary(BinaryGuard):
    classifier_representation='binary_char_and_category_char_wb_2_5'
    classifier_threshold=.889530520789236

    def __init__(self,searcher,llm):
        super().__init__(searcher,llm)
        rows,texts,_,_=public_data()
        category=BoundaryModel('char_mlp',self.classifier_seed,self.classifier_width,
            self.classifier_alpha,self.classifier_c).fit(texts,[r['labels']['category'] for r in rows])
        self.matrix_model=BinaryAdjustedModel(category,self.matrix_model.decision_model,self.classifier_binary_weight)


class BinaryBoundaryBudgets(BinaryBoundary,InputBudgets):
    pass


class BinaryDense(BinaryGuard):
    classifier_kind='dense_mlp'
    classifier_representation='binary_char_and_category_dense'
    classifier_threshold=.8820789131634711
