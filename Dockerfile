# Egyptian TTS pronunciation frontend: HTTP API (api/main.py) on port 9900.
#
#   docker build -t electropi/tashkeel_egy .
#   docker run -p 9900:9900 electropi/tashkeel_egy
#
# The exported diacritizer (assets/diacritizer, from the private HF repo via
# build/fetch_model.py) is copied into the image, so the container needs no token and
# no network at start. The image is therefore private: it contains licensed weights.
FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    EGY_HOST=0.0.0.0 \
    EGY_PORT=9900 \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

RUN useradd --uid 10001 --no-create-home app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt requirements-api.txt ./
RUN uv venv $VIRTUAL_ENV && uv pip install --no-cache -r requirements-api.txt

# Model and lexicon (~900 MB): their own layer, rebuilt only when they change.
COPY --chown=app assets/ assets/

COPY --chown=app egy_frontend/ egy_frontend/
COPY --chown=app api/ api/

USER app

EXPOSE 9900

# /ready answers 503 until the model is loaded and warm (~8 s).
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:9900/ready', timeout=4).status != 200)"

# One worker on purpose: the model is shared by the worker threads inside it (see
# api/config.py). Threads and ORT threads are auto-tuned to the container's CPU quota,
# so `docker run --cpus N` is respected; override with EGY_WORKERS / EGY_ORT_THREADS.
CMD ["uv", "run", "--no-project", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9900", "--workers", "1"]
