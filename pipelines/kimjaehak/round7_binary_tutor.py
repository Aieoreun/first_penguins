"""Factorial combination of independently tested guard and numeric tutor paths."""
from .round7_binary_guard import BinaryGuard
from .round7_tax_scope import TaxScopedConcrete,TaxBudget12


class BinaryTutor(BinaryGuard,TaxScopedConcrete):
    pass


class BinaryTutor12(BinaryGuard,TaxBudget12):
    pass
