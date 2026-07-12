"""Zero-token local solvers."""

from .logic import solve_logic_puzzle
from .math import solve_math_exact
from .ner import solve_ner
from .sentiment import solve_sentiment

__all__ = ["solve_logic_puzzle", "solve_math_exact", "solve_ner", "solve_sentiment"]
