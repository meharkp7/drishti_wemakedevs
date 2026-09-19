

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Protocol
from uuid import UUID

from backend.core.job_state import AnalysisJob
from backend.schemas.job import AnalysisJobSchema, JobSchemaError


class JobRepositoryError(RuntimeError):
    """Base exception for job repository failures."""


class JobNotFoundError(JobRepositoryError):
    """Raised when a requested job does not exist."""


class JobConcurrencyError(JobRepositoryError):
    """Raised when optimistic concurrency validation fails."""


class JobPersistenceError(JobRepositoryError):
    """Raised when a job cannot be persisted or reconstructed."""


class JobRepository(Protocol):
    """Persistence contract used by the DRISHTI job service."""

    def create(self, job: AnalysisJob) -> None: ...
    def get(self, job_id: UUID) -> AnalysisJob: ...

    def save(
        self,
        job: AnalysisJob,
        *,
        expected_revision: int | None = None,
    ) -> None: ...

    def exists(self, job_id: UUID) -> bool: ...


class LocalFileJobRepository:
    """JSON-backed repository with atomic writes and process-safe locking.

    Each job is stored as ``<root>/<job-id>.json`` and has a companion
    ``<root>/<job-id>.lock`` file.  The lock serializes the complete
    read/compare/write critical section used by optimistic concurrency.
    ``os.replace`` then atomically publishes the new JSON document.

    This implementation is appropriate for a local/single-host worker
    deployment. A multi-host deployment should use a transactional database
    implementation behind the same ``JobRepository`` protocol.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)
        try:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            if not self._root_dir.is_dir():
                raise OSError("repository root is not a directory")
        except OSError as exc:
            raise JobPersistenceError(
                f"Unable to initialize job repository: {self._root_dir}"
            ) from exc

    def create(self, job: AnalysisJob) -> None:
        """Create a job exactly once, rejecting duplicate IDs atomically."""
        path = self._path_for(job.job_id)
        with self._locked(job.job_id):
            if path.exists():
                raise JobPersistenceError(f"Job already exists: {job.job_id}")
            self._atomic_write(job)

    def get(self, job_id: UUID) -> AnalysisJob:
        """Load and fully validate a persisted job."""
        self._path_for(job_id)
        path = self._path_for(job_id)
        with self._locked(job_id, exclusive=False):
            if not path.is_file():
                raise JobNotFoundError(f"Job not found: {job_id}")
            return self._read_job(path, job_id)

    def save(
        self,
        job: AnalysisJob,
        *,
        expected_revision: int | None = None,
    ) -> None:
        """Persist an existing job, atomically enforcing revision CAS.

        When ``expected_revision`` is supplied, the current persisted revision
        and proposed revision are checked while holding the per-job process
        lock. This closes the read/compare/write race present in a plain
        optimistic check.

        Omitting ``expected_revision`` retains the repository protocol's
        legacy unconditional-save behavior. Production job-service writes
        should always provide it.
        """
        path = self._path_for(job.job_id)
        with self._locked(job.job_id):
            if not path.is_file():
                raise JobNotFoundError(
                    f"Cannot save missing job: {job.job_id}"
                )

            if expected_revision is not None:
                current = self._read_job(path, job.job_id)
                if current.revision != expected_revision:
                    raise JobConcurrencyError(
                        f"Job {job.job_id} has revision {current.revision}; "
                        f"expected {expected_revision}."
                    )
                if job.revision != expected_revision + 1:
                    raise JobConcurrencyError(
                        f"Invalid revision transition for job {job.job_id}: "
                        f"expected new revision {expected_revision + 1}, "
                        f"got {job.revision}."
                    )

            self._atomic_write(job)

    def exists(self, job_id: UUID) -> bool:
        """Return whether the canonical job record exists."""
        return self._path_for(job_id).is_file()

    def _path_for(self, job_id: UUID) -> Path:
        if not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID.")
        return self._root_dir / f"{job_id}.json"

    def _lock_path_for(self, job_id: UUID) -> Path:
        return self._root_dir / f".{job_id}.lock"

    @contextmanager
    def _locked(
        self,
        job_id: UUID,
        *,
        exclusive: bool = True,
    ) -> Iterator[None]:
        """Acquire a per-job advisory OS lock."""
        lock_path = self._lock_path_for(job_id)
        try:
            with lock_path.open("a+b") as handle:
                self._acquire_lock(handle, exclusive=exclusive)
                try:
                    yield
                finally:
                    self._release_lock(handle)
        except JobPersistenceError:
            raise
        except OSError as exc:
            raise JobPersistenceError(
                f"Unable to lock job {job_id}"
            ) from exc

    @staticmethod
    def _acquire_lock(handle, *, exclusive: bool) -> None:
        if os.name == "nt":
            import msvcrt

            # msvcrt.locking operates on a byte range. The lock file is
            # guaranteed to contain at least one byte before locking.
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            mode = msvcrt.LK_LOCK
            msvcrt.locking(handle.fileno(), mode, 1)
            return

        import fcntl

        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), operation)

    @staticmethod
    def _release_lock(handle) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_job(self, path: Path, job_id: UUID) -> AnalysisJob:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise JobPersistenceError(
                f"Unable to read persisted job: {job_id}"
            ) from exc

        try:
            job = AnalysisJobSchema.from_dict(payload).to_job()
        except (JobSchemaError, ValueError, TypeError) as exc:
            raise JobPersistenceError(
                f"Persisted job is invalid: {job_id}"
            ) from exc

        if job.job_id != job_id:
            raise JobPersistenceError(
                f"Persisted job ID does not match requested ID: {job_id}"
            )
        return job

    def _atomic_write(self, job: AnalysisJob) -> None:
        """Validate, fsync, atomically replace, then fsync the directory."""
        try:
            payload = AnalysisJobSchema.from_job(job).to_json_dict()
        except (JobSchemaError, ValueError, TypeError) as exc:
            raise JobPersistenceError(
                f"Unable to serialize job: {job.job_id}"
            ) from exc

        target = self._path_for(job.job_id)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._root_dir,
                prefix=f".{job.job_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_path, target)
            temporary_path = None
            self._fsync_directory()
        except (OSError, TypeError, ValueError) as exc:
            raise JobPersistenceError(
                f"Unable to persist job: {job.job_id}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _fsync_directory(self) -> None:
        """Persist the directory entry update where the OS supports it."""
        if os.name == "nt":
            return
        directory_fd: int | None = None
        try:
            directory_fd = os.open(self._root_dir, os.O_RDONLY)
            os.fsync(directory_fd)
        except OSError as exc:
            raise JobPersistenceError(
                f"Unable to durably persist repository directory: {self._root_dir}"
            ) from exc
        finally:
            if directory_fd is not None:
                os.close(directory_fd)