"""HTTP API for the Egyptian pronunciation frontend: text in, text for the TTS out.

    python -m api                      # or: uvicorn api.main:app --port 8080
    curl -s localhost:8080/diacritize -H 'content-type: application/json' \
         -d '{"text":"البلكونة اتملت حمام!"}' | jq

Shape of the service (see api/config.py for the measurements behind the defaults):
one process holds one Frontend -- the 684 MB ONNX session and the 517k-word lexicon --
and a fixed pool of worker threads calls it concurrently. ONNX Runtime releases the GIL
inside run(), which is 97% of a request, so N threads give ~N times the throughput
without a second copy of the model. Load above capacity is shed with 503 rather than
queued without bound, so p95 stays near p50 under overload.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Lock
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings

log = logging.getLogger("egy_api")

# ---------------------------------------------------------------- request / response


class DiacritizeRequest(BaseModel):
    text: str = Field(..., description="Egyptian Arabic text, one sentence or a short paragraph.",
                      examples=["البلكونة اتملت حمام!"])
    decisions: bool = Field(True, description="Include the rule applied to each Arabic word.")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., description="Texts to process; run in parallel across the pool.",
                             examples=[["البلكونة اتملت حمام!", "الغسالة صفرت والفني جاي يركب التكييف."]])
    decisions: bool = True


class DecisionOut(BaseModel):
    word: str          # as it reached the frontend (normalized)
    predicted: str     # the diacritizer's marking
    output: str        # what the TTS gets
    rule: Literal["caller", "rare", "ambiguous", "differs", "plain", "unsure", "other"]
    uses: int          # occurrences in the TTS training text
    default: str | None  # the training text's most used reading
    confidence: float  # lowest CRF marginal over the word's letters but the last


class DiacritizeResponse(BaseModel):
    text: str = Field(..., description="Text to hand to the TTS.")
    input: str
    decisions: list[DecisionOut] | None = None
    unsure: list[str] = Field(default_factory=list,
                              description="Words left plain because the diacritizer was not confident; "
                                          "an LLM can be asked about these.")
    marked: int = Field(0, description="Words that received marks.")
    cached: bool = False
    ms: float = 0.0


class BatchResponse(BaseModel):
    results: list[DiacritizeResponse]
    ms: float


# ---------------------------------------------------------------- the engine


class Engine:
    """The shared Frontend plus the thread pool and the load-shedding gate."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.fe: Any = None
        self.pool: ThreadPoolExecutor | None = None
        self.gate: asyncio.Semaphore | None = None
        self.ready = False
        self._lock = Lock()
        self.started = time.time()
        self.served = self.failed = self.shed = 0
        self.busy = 0
        self.peak_busy = 0
        self.total_ms = 0.0

    def load(self) -> None:
        from egy_frontend import Frontend

        # ONNX Runtime prints one "GPU device discovery failed: /sys/class/drm" warning
        # while it is imported, before any of our code can set a log level. Harmless:
        # this runtime is CPU-only by design.
        c = self.cfg
        t0 = time.perf_counter()
        self.fe = Frontend(
            model_dir=c.model_dir, quantized=c.quantized, threads=c.ort_threads,
            rare_uses=c.rare_uses, deviate_ambiguous=c.deviate_ambiguous,
            deviate_other=c.deviate_other, cache=c.cache,
        )
        # Warm up: the first call pays for ONNX Runtime's arena and the lexicon's pages.
        self.fe("البلكونة اتملت حمام والغسالة صفرت.")
        if c.cache:
            self.fe.analyze.cache_clear()
        self.pool = ThreadPoolExecutor(max_workers=c.workers, thread_name_prefix="egy")
        self.gate = asyncio.Semaphore(c.capacity)
        self.ready = True
        log.info("model %s loaded in %.1fs | %d worker threads x %d ort threads on %d cores "
                 "| capacity %d", "int8" if c.quantized else "fp32", time.perf_counter() - t0,
                 c.workers, c.ort_threads, c.cores, c.capacity)

    def shutdown(self) -> None:
        self.ready = False
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def _analyze(self, text: str) -> tuple[str, tuple, bool]:
        """Runs on a worker thread. Returns (text, decisions, was_cached)."""
        hits = self.fe.analyze.cache_info().hits if self.cfg.cache else 0
        out, decisions = self.fe.analyze(text)
        cached = bool(self.cfg.cache) and self.fe.analyze.cache_info().hits > hits
        return out, decisions, cached

    async def run(self, text: str, want_decisions: bool) -> DiacritizeResponse:
        """Acquire a slot, run on the pool, shed load if the system is already full."""
        if self.gate is None or not self.ready:
            raise HTTPException(503, "model is still loading")
        try:
            await asyncio.wait_for(self.gate.acquire(), timeout=self.cfg.queue_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            with self._lock:
                self.shed += 1
            raise HTTPException(503, f"server at capacity ({self.cfg.capacity} in flight); retry shortly",
                                headers={"Retry-After": "1"}) from None
        t0 = time.perf_counter()
        with self._lock:
            self.busy += 1
            self.peak_busy = max(self.peak_busy, self.busy)
        try:
            loop = asyncio.get_running_loop()
            out, decisions, cached = await loop.run_in_executor(self.pool, self._analyze, text)
        except Exception as exc:
            with self._lock:
                self.failed += 1
            log.exception("analyze failed")
            raise HTTPException(500, f"analyze failed: {type(exc).__name__}: {exc}") from exc
        finally:
            with self._lock:
                self.busy -= 1
            self.gate.release()
        ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            self.served += 1
            self.total_ms += ms
        return DiacritizeResponse(
            text=out, input=text,
            decisions=[DecisionOut(**asdict(d)) for d in decisions] if want_decisions else None,
            unsure=[d.word for d in decisions if d.rule == "unsure"],
            marked=sum(1 for d in decisions if d.rule not in ("plain", "unsure")),
            cached=cached, ms=round(ms, 1),
        )

    def stats(self) -> dict:
        with self._lock:
            served, failed, shed = self.served, self.failed, self.shed
            busy, peak, total_ms = self.busy, self.peak_busy, self.total_ms
        up = time.time() - self.started
        ci = self.fe.analyze.cache_info() if (self.fe and self.cfg.cache) else None
        return {
            "ready": self.ready, "uptime_s": round(up, 1),
            "served": served, "failed": failed, "shed": shed,
            "in_flight": busy, "peak_in_flight": peak, "capacity": self.cfg.capacity,
            "mean_ms": round(total_ms / served, 1) if served else None,
            "throughput_per_s": round(served / up, 2) if up > 0 else None,
            "cache": ({"hits": ci.hits, "misses": ci.misses, "size": ci.currsize,
                       "max": ci.maxsize} if ci else None),
        }


engine: Engine | None = None


def get_engine() -> Engine:
    if engine is None or not engine.ready:
        raise HTTPException(503, "model is still loading")
    return engine


EngineDep = Annotated[Engine, Depends(get_engine)]


# ---------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    cfg: Settings = app.state.settings
    engine = Engine(cfg)
    # Loading is ~8s of blocking CPU; keep the event loop free so /health answers.
    await asyncio.get_running_loop().run_in_executor(None, engine.load)
    app.state.engine = engine
    try:
        yield
    finally:
        engine.shutdown()


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings.from_env()
    app = FastAPI(
        title="Egyptian TTS pronunciation frontend",
        version="1.0.0",
        summary="Adds diacritics to the words the Egyptian TTS would otherwise misread.",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    if cfg.cors:
        origins = ["*"] if cfg.cors.strip() == "*" else [o.strip() for o in cfg.cors.split(",") if o.strip()]
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    def _check(text: str) -> str:
        if not text or not text.strip():
            raise HTTPException(422, "text must not be empty")
        if len(text) > cfg.max_chars:
            raise HTTPException(413, f"text is {len(text)} characters; the limit is {cfg.max_chars}")
        return text

    @app.post("/diacritize", response_model=DiacritizeResponse, summary="Prepare one text for the TTS")
    async def diacritize(req: DiacritizeRequest, eng: EngineDep) -> DiacritizeResponse:
        return await eng.run(_check(req.text), req.decisions)

    @app.post("/diacritize/batch", response_model=BatchResponse, summary="Prepare many texts in parallel")
    async def diacritize_batch(req: BatchRequest, eng: EngineDep) -> BatchResponse:
        if not req.texts:
            raise HTTPException(422, "texts must not be empty")
        if len(req.texts) > cfg.max_batch:
            raise HTTPException(413, f"{len(req.texts)} texts; the limit is {cfg.max_batch}")
        for t in req.texts:
            _check(t)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(eng.run(t, req.decisions) for t in req.texts))
        return BatchResponse(results=list(results), ms=round((time.perf_counter() - t0) * 1000, 1))

    @app.get("/health", summary="Liveness: answers while the model is still loading")
    async def health() -> dict:
        return {"status": "healthy", "ready": bool(engine and engine.ready)}

    @app.get("/ready", summary="Readiness: 503 until the model is warm")
    async def ready() -> dict:
        if not (engine and engine.ready):
            raise HTTPException(503, "model is still loading")
        return {"status": "ready"}

    @app.get("/stats", summary="Counters, latency and cache")
    async def stats() -> dict:
        if engine is None:
            raise HTTPException(503, "model is still loading")
        return engine.stats()

    @app.get("/config", summary="Resolved configuration and the CPU budget it was tuned to")
    async def config() -> dict:
        return cfg.public()

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({
            "service": "Egyptian TTS pronunciation frontend",
            "docs": "/docs",
            "endpoints": ["POST /diacritize", "POST /diacritize/batch",
                          "GET /health", "GET /ready", "GET /stats", "GET /config"],
        })

    return app


app = create_app()


def main() -> None:
    import uvicorn

    cfg: Settings = app.state.settings
    logging.basicConfig(level=os.environ.get("EGY_LOG", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # One process on purpose: the model is shared across threads inside it. More uvicorn
    # workers would each load another 684 MB copy and oversubscribe the same cores.
    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1, access_log=False,
                timeout_keep_alive=65)


if __name__ == "__main__":
    main()
