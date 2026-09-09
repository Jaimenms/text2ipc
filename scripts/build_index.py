"""Build (or refresh) the embedding CSV for an IPC version.

Thin wrapper over ``t2ipc build`` for people who prefer ``python scripts/build_index.py``.

    uv run python scripts/build_index.py --version 20260101 --lang EN
    uv run python scripts/build_index.py --version 20270101          # reuses 20260101 vectors
"""

from text2ipc.cli import app

if __name__ == "__main__":
    app(["build", *__import__("sys").argv[1:]])
