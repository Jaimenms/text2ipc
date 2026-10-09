"""text2ipc - map free text to IPC symbols with embeddings and hierarchical heuristics.

>>> from text2ipc import classify
>>> for m in classify("A hand-held hoe with two blades", version="20260101", level="group"):
...     print(m.pretty, round(m.score, 3), m.title)
"""

from .classifier import IpcClassifier, classify
from .config import LEVELS
from .search import Beam, Match, SearchParams, Weights

__all__ = ["LEVELS", "Beam", "IpcClassifier", "Match", "SearchParams", "Weights", "classify"]
__version__ = "0.2.2"
