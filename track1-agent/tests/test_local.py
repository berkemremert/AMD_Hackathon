import sys
from src.local_coder.core import solve_local_coder
try:
    print(solve_local_coder("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.", "code_authoring"))
except Exception as e:
    print(e, file=sys.stderr)
