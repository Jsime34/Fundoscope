"""Parcel analyses as background jobs (FR-13).

The API queues a job and answers at once with its id; a worker thread claims queued jobs
one at a time, runs them, and records their progress, so anyone can follow a job with its
id. The flow: queued -> running (progress 0 to 100) -> done, or failed with an error in
plain language. A job that fails never stops the worker or the API.

The worker polls the job store every 2 seconds, and `Worker.wake` starts a job queued from
the same process at once. A job still marked running when the worker starts was cut off by
a restart: it is marked failed, and starting the analysis again resumes from what is stored.
"""

import logging
import threading
from collections.abc import Callable
from uuid import UUID

from fundoscope.db.jobs import ANALYZE, Job, JobStore
from fundoscope.pipeline.parcel_history import Progress
from fundoscope.pipeline.parcel_store import StoredAnalysis
from fundoscope.sentinel import SentinelError

log = logging.getLogger(__name__)

POLL_S = 2.0
INTERRUPTED = "The server restarted while this analysis was running. Start it again."
MAX_ERROR_CHARS = 300

# Runs one analysis of a stored parcel, reporting progress as `history_of` does.
Analyze = Callable[[UUID, Progress], StoredAnalysis]


def start_analysis(parcel_id: UUID, jobs: JobStore) -> Job:
    """Queue an analysis of the parcel, unless one is already queued or running."""
    return jobs.active(parcel_id, ANALYZE) or jobs.create(parcel_id, ANALYZE)


def run_next(jobs: JobStore, analyze: Analyze) -> Job | None:
    """Claim the oldest queued job and run it to the end (done or failed).

    Returns:
        The finished job, or None if no job was queued.
    """
    job = jobs.claim()
    if job is None:
        return None

    def progress(done: int, total: int) -> None:
        jobs.update(job.id, *progress_step(done, total))

    try:
        analysis = analyze(job.parcel_id, progress)
    except Exception as exc:  # any failure ends the job, never the worker
        log.exception("Job %s failed", job.id)
        return jobs.finish(job.id, error=readable_error(exc))
    history = analysis.history
    return jobs.finish(
        job.id, processing_units=history.daily.processing_units, credits=history.credits
    )


def progress_step(done: int, total: int) -> tuple[int, str]:
    """A job's progress (5 to 95 while downloading) and what it is doing."""
    if done >= total:
        return 95, "Building the report"
    return 5 + 90 * done // total, f"Downloading satellite data: {done} of {total} requests done"


def readable_error(exc: Exception) -> str:
    """One plain sentence about what went wrong (the full traceback goes to the log)."""
    detail = next(iter(str(exc).strip().splitlines()), "")  # e.g. not the SQL of a DB error
    if isinstance(exc, SentinelError):
        message = f"The satellite service failed ({detail}). Try again later."
    else:
        message = (
            f"Unexpected error ({type(exc).__name__}): {detail}"
            if detail
            else (f"Unexpected error ({type(exc).__name__}).")
        )
    return message[:MAX_ERROR_CHARS]


class Worker:
    """Runs queued jobs one at a time in a background thread."""

    def __init__(self, jobs: JobStore, analyze: Analyze, poll_s: float = POLL_S) -> None:
        self.jobs = jobs
        self.analyze = analyze
        self.poll_s = poll_s
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recovered = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="fundoscope-jobs", daemon=True)
        self._thread.start()

    def wake(self) -> None:
        """Look for queued jobs now instead of at the next poll."""
        self._wake.set()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop after the current job (a download in progress isn't interrupted)."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def step(self) -> bool:
        """Run one queued job, if any.

        Returns:
            False when there was nothing to run, or the job store failed (e.g. the database
            is down): the loop then waits before trying again.
        """
        try:
            if not self._recovered:
                if count := self.jobs.fail_running(INTERRUPTED):
                    log.warning("%d job(s) were cut off by a restart: marked failed", count)
                self._recovered = True
            return run_next(self.jobs, self.analyze) is not None
        except Exception:
            log.exception("The job worker couldn't reach the job store")
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.step():
                self._wake.wait(self.poll_s)
                self._wake.clear()
