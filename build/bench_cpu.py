"""CPU latency of the frontend (the whole call: normalize, diacritize, policy,
canonicalize), one sentence at a time as a voice agent sends them, on the 200 showcase
texts' sentences (or --texts, one per line). Cache off.

    .venv/bin/python build/bench_cpu.py --model-dir assets/diacritizer --threads 1 2 4 8
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from egy_frontend import Frontend  # noqa: E402

SHOWCASE = Path("/tmp/claude-0/-workspace/e3aa8d5b-f9ff-4828-a73b-3df047f60a5b/scratchpad/showcase/items.jsonl")


def sentences(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        texts = [json.loads(l)["text"] for l in path.open(encoding="utf-8")]
        return [s for t in texts for s in re.split(r"(?<=[.!؟?])\s+", t) if s.strip()]
    return [l.strip() for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "assets" / "diacritizer"))
    ap.add_argument("--texts", type=Path, default=SHOWCASE)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    texts = sentences(a.texts)
    out = {"sentences": len(texts), "mean_chars": round(sum(map(len, texts)) / len(texts), 1)}
    for q in (False, True):
        for th in a.threads:
            fe = Frontend(model_dir=a.model_dir, quantized=q, threads=th, cache=0)
            for t in texts[:5]:
                fe(t)
            ms = []
            for t in texts:
                s = time.perf_counter()
                fe(t)
                ms.append(1000 * (time.perf_counter() - s))
            ms.sort()
            key = f"{'int8' if q else 'fp32'}_threads{th}"
            out[key] = {"median_ms": round(statistics.median(ms), 1), "p95_ms": round(ms[int(0.95 * len(ms))], 1),
                        "max_ms": round(ms[-1], 1)}
            print(key, out[key], flush=True)
    print(json.dumps(out))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
