"""Present so that a bare `pytest` works, not just `python -m pytest`.

pytest prepends the directory containing the rootdir conftest.py to sys.path.
Without this file the `tests/` directory goes on the path instead of the project
root, and `from app import ...` fails with ModuleNotFoundError on a fresh clone.
"""
