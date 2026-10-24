"""FastAPI application. Run with: uv run uvicorn fundoscope.api.main:app --reload --port 8020"""

from fastapi import FastAPI

from fundoscope import __version__
from fundoscope.api.history import router as history_router

app = FastAPI(title="Fundoscope API", version=__version__)
app.include_router(history_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
