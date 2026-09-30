"""FastAPI application. Run with: uv run uvicorn fundoscope.api.main:app --reload --port 8020"""

from fastapi import FastAPI

from fundoscope import __version__

app = FastAPI(title="Fundoscope API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
