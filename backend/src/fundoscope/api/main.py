"""FastAPI application. Run with: uv run uvicorn fundoscope.api.main:app --reload --port 8020

The background job worker (FR-13) starts and stops with the app.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from fundoscope import __version__
from fundoscope.api import jobs as jobs_api
from fundoscope.api.history import router as history_router
from fundoscope.api.valley import router as valley_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run the job worker while the app runs."""
    worker = jobs_api.make_worker()
    worker.start()
    app.state.worker = worker
    yield
    worker.stop()
    del app.state.worker


app = FastAPI(title="Fundoscope API", version=__version__, lifespan=lifespan)
app.include_router(history_router)
app.include_router(jobs_api.router)
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
