"""
Start the FastAPI chat API (app lives in main.py).

Sets cwd to this package directory so core_logic resolves qdrant_db_clean at repo root.
Replit sets PORT; default 8000 locally.
"""
import os
import sys
from pathlib import Path


def main() -> None:
    src = Path(__file__).resolve().parent
    sys.path.insert(0, str(src))
    os.chdir(src)
    port = int(os.environ.get("PORT", "8000"))
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
