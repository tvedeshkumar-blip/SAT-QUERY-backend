"""
Root entrypoint shim for SatQuery AI Backend.
Forwards to the application in the app package (app.main).
Ensures backward compatibility for both `uvicorn main:app` and `uvicorn app.main:app`.
"""
import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from app.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
