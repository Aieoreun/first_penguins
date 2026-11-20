"""Cross the retained binary input gate with previously isolated safety paths."""
from .round7_binary_guard import BinaryGuard
from .neural_peace_guard import ParallelSafety,CheckBudgets,InputBudgets


class BinaryParallelSafety(BinaryGuard,ParallelSafety):
    pass


class BinaryCheckBudgets(BinaryGuard,CheckBudgets):
    pass


class BinaryInputBudgets(BinaryGuard,InputBudgets):
    pass
