"""Bounded background experiment executor with persisted status and cooperative cancellation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Callable
import uuid

from thermotwin.config import project_path
from thermotwin.services.artifacts import atomic_json


@dataclass
class JobRecord:
    """Persistable background job state with bounded logs."""

    job_id: str
    profile: str
    status: str = "queued"
    progress: int = 0
    progress_events: int = 0
    logs: list[str] = field(default_factory=list)
    run_id: str | None = None
    error: str | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "cancel_event"}


class JobManager:
    """Run at most the configured number of expensive jobs outside request threads."""

    def __init__(self, config: dict[str, Any], runner: Callable[[str, Callable[[str], None], Callable[[], bool]], str]):
        self.config, self.runner = config, runner
        self.executor = ThreadPoolExecutor(max_workers=int(config["service"]["max_jobs"]), thread_name_prefix="thermotwin-job")
        self.jobs: dict[str, JobRecord] = {}
        self.lock = threading.RLock()
        self.directory = project_path(config, "artifacts_dir") / "jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("*.json"):
            try:
                data = __import__("json").loads(path.read_text(encoding="utf-8"))
                if data.get("status") in {"queued", "running", "cancelling"}:
                    data["status"], data["error"] = "interrupted", "application restarted before job completion"
                    atomic_json(path, data)
            except Exception:
                continue

    def _persist(self, job: JobRecord) -> None:
        atomic_json(self.directory / f"{job.job_id}.json", job.as_dict())

    @staticmethod
    def _normalized(data: dict[str, Any]) -> dict[str, Any]:
        """Upgrade older persisted jobs that used a misleading per-log +2% scale."""
        result = dict(data)
        if "progress_events" not in result:
            result["progress_events"] = len(result.get("logs", []))
            if result.get("status") != "complete":
                expected = 45 if result.get("profile") == "smoke" else 225
                result["progress"] = min(95, round(result["progress_events"] / expected * 95))
        return result

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent persisted and live jobs so a refreshed page can reconnect."""
        records: dict[str, dict[str, Any]] = {}
        for path in self.directory.glob("*.json"):
            try:
                data = __import__("json").loads(path.read_text(encoding="utf-8"))
                if data.get("job_id"):
                    records[data["job_id"]] = self._normalized(data)
            except (OSError, ValueError, KeyError):
                continue
        with self.lock:
            records.update({identifier: job.as_dict() for identifier, job in self.jobs.items()})
        return sorted(records.values(), key=lambda item: item.get("created_utc", ""), reverse=True)[:limit]

    def create(self, profile: str) -> dict[str, Any]:
        if profile not in {"smoke", "research"}:
            raise ValueError("profile must be smoke or research")
        job = JobRecord(uuid.uuid4().hex[:12], profile)
        with self.lock:
            active = sum(existing.status in {"queued", "running", "cancelling"} for existing in self.jobs.values())
            if active >= int(self.config["service"]["max_jobs"]):
                raise ValueError("the bounded experiment worker is busy; wait or cancel the active job")
            self.jobs[job.job_id] = job
            self._persist(job)
        self.executor.submit(self._run, job)
        return job.as_dict()

    def _run(self, job: JobRecord) -> None:
        with self.lock:
            job.status = "running"
            self._persist(job)
        def log(message: str) -> None:
            with self.lock:
                job.progress_events += 1
                job.logs.append(message)
                job.logs = job.logs[-200:]
                expected_messages = 45 if job.profile == "smoke" else 225
                job.progress = min(95, max(job.progress, round(job.progress_events / expected_messages * 95)))
                self._persist(job)
        try:
            run_id = self.runner(job.profile, log, job.cancel_event.is_set)
            with self.lock:
                job.run_id = run_id
                job.status = "cancelled" if job.cancel_event.is_set() else "complete"
                job.progress = 100 if job.status == "complete" else job.progress
        except InterruptedError:
            with self.lock:
                job.status = "cancelled"
        except Exception as error:
            with self.lock:
                job.status, job.error = "failed", f"{type(error).__name__}: {error}"
        finally:
            with self.lock:
                self._persist(job)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                return job.as_dict()
        path = self.directory / f"{job_id}.json"
        if not path.is_file():
            raise KeyError(job_id)
        return self._normalized(__import__("json").loads(path.read_text(encoding="utf-8")))

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status not in {"queued", "running"}:
                raise ValueError("only queued or running jobs can be cancelled")
            job.cancel_event.set()
            job.status = "cancelling"
            self._persist(job)
            return job.as_dict()

    def shutdown(self) -> None:
        for job in self.jobs.values():
            job.cancel_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
