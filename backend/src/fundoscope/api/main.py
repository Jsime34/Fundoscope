"""FastAPI application. Run with: uv run uvicorn fundoscope.api.main:app --reload --port 8020"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from fundoscope import __version__
from fundoscope.api.history import router as history_router
from fundoscope.api.valley import router as valley_router

app = FastAPI(title="Fundoscope API", version=__version__)
app.include_router(history_router)
app.include_router(valley_router)


@app.exception_handler(OperationalError)
def database_unavailable(request: Request, exc: OperationalError) -> JSONResponse:
    """The database is down (or unreachable): say so instead of a bare 500."""
    return JSONResponse(
        status_code=503,
        content={"detail": "The database isn't available. Is the PostGIS container running?"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
