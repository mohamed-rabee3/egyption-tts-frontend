"""Runtime configuration, auto-tuned to the CPU budget of the machine (or container).

Every value can be overridden with an EGY_* environment variable, read from the
process environment or from a .env file next to the repo root.

Why the defaults are what they are (measured on a 6-core Xeon E5-2683 v4, fp32):

    ort_threads  concurrency   throughput   p50
          1           6          42.8/s    120 ms
          2           6          47.4/s    109 ms   <- default: near-peak, lowest p50
          2          12          50.0/s    240 ms
          3           1          21.6/s     47 ms   <- lowest single-request latency
          6           1          21.3/s     46 ms   (no gain: 2 NUMA nodes of 3 cores)

A single ONNX Runtime session is shared by every worker thread: `run()` releases the
GIL and accounts for 97% of the work, so threads scale almost linearly while the
684 MB model stays in memory once. Hence one process, many threads -- NOT several
uvicorn workers, which would copy the model per worker and oversubscribe the cores.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dotenv() -> None:
    """Load the repo's .env without overriding variables already set."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def cpu_budget() -> int:
    """Cores this process may actually use: CPU affinity capped by the cgroup quota.

    os.cpu_count() reports the host's cores and so over-reports inside a container
    with `--cpus`; the cgroup quota is what the scheduler will really give us.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    quota = None
    try:  # cgroup v2
        quota_s, period_s = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_s != "max":
            quota = float(quota_s) / float(period_s)
    except Exception:
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0 and p > 0:
                quota = q / p
        except Exception:
            quota = None
    if quota and quota >= 1:
        n = min(n, int(quota))
    return max(1, n)


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v in (None, "") else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- model ---
    model_dir: str = ""
    quantized: bool = False       # EGY_INT8=1 -> int8 encoder: ~2x faster, slightly worse
    ort_threads: int = 0          # per-session intra-op threads (0 = auto)
    # --- policy thresholds (server-wide; see README "How it works") ---
    rare_uses: int = 20
    deviate_ambiguous: float = 0.7
    deviate_other: float = 0.95
    cache: int = 4096             # sentences memoized (LRU, thread-safe)
    # --- concurrency ---
    workers: int = 0              # worker threads sharing the session (0 = auto)
    queue: int = 0                # extra requests allowed to wait (0 = auto: 2x workers)
    queue_timeout: float = 10.0   # seconds a request waits for a slot before 503
    # --- limits ---
    max_chars: int = 10_000
    max_batch: int = 256
    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8080
    cors: str = ""                # comma-separated origins, "*" for any
    cores: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        _dotenv()
        cores = cpu_budget()
        # 2 intra-op threads per session is the throughput sweet spot from 4 cores up;
        # below that, one thread per request parallelizes better than splitting a run.
        ort_threads = _int("EGY_ORT_THREADS", 2 if cores >= 4 else 1)
        workers = _int("EGY_WORKERS", max(1, cores))
        return cls(
            model_dir=os.environ.get("EGY_MODEL_DIR", str(ROOT / "assets" / "diacritizer")),
            quantized=_bool("EGY_INT8", False),
            ort_threads=ort_threads,
            rare_uses=_int("EGY_RARE_USES", 20),
            deviate_ambiguous=_float("EGY_DEVIATE_AMBIGUOUS", 0.7),
            deviate_other=_float("EGY_DEVIATE_OTHER", 0.95),
            cache=_int("EGY_CACHE", 4096),
            workers=workers,
            queue=_int("EGY_QUEUE", workers * 2),
            queue_timeout=_float("EGY_QUEUE_TIMEOUT", 10.0),
            max_chars=_int("EGY_MAX_CHARS", 10_000),
            max_batch=_int("EGY_MAX_BATCH", 256),
            host=os.environ.get("EGY_HOST", "0.0.0.0"),
            port=_int("EGY_PORT", 8080),
            cors=os.environ.get("EGY_CORS", ""),
            cores=cores,
        )

    @property
    def capacity(self) -> int:
        """Requests in the system (running + waiting) before we shed load."""
        return self.workers + self.queue

    def public(self) -> dict:
        d = asdict(self)
        d["capacity"] = self.capacity
        d["model"] = "int8" if self.quantized else "fp32"
        return d
