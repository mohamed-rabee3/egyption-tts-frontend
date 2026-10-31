"""python -m egy_frontend [--decisions] [--int8] "text" ...   (or lines on stdin)"""
import argparse
import sys
import time

from . import Frontend

ap = argparse.ArgumentParser(prog="python -m egy_frontend")
ap.add_argument("text", nargs="*")
ap.add_argument("--decisions", action="store_true", help="print the rule applied to each word")
ap.add_argument("--int8", action="store_true", help="int8 encoder (faster, slightly less accurate)")
ap.add_argument("--threads", type=int)
ap.add_argument("--model-dir")
a = ap.parse_args()
kw = {"model_dir": a.model_dir} if a.model_dir else {}
fe = Frontend(quantized=a.int8, threads=a.threads, **kw)
for line in a.text or (l.rstrip("\n") for l in sys.stdin):
    t0 = time.perf_counter()
    out, decisions = fe.analyze(line)
    ms = (time.perf_counter() - t0) * 1000
    print(out)
    if a.decisions:
        for d in decisions:
            if d.rule != "plain":
                print(f"   {d.rule:9s} {d.word} -> {d.output}   (uses {d.uses}, default {d.default}, conf {d.confidence:.2f})")
        print(f"   [{ms:.0f} ms]")
