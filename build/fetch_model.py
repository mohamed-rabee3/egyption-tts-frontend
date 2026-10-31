"""Downloads the exported diacritizer (the onnx/ folder of the private Hub repo) into
assets/diacritizer. Needs huggingface_hub and a token with access (HF_TOKEN or a login).

    python build/fetch_model.py [--repo Rabe3/Tashkeel-v4-Egyptian] [--revision main]
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Rabe3/Tashkeel-v4-Egyptian")
ap.add_argument("--revision", default="main")
ap.add_argument("--int8-only", action="store_true", help="skip the 684 MB fp32 model")
a = ap.parse_args()
skip = ["onnx/model.fp32.onnx"] if a.int8_only else []
src = Path(snapshot_download(a.repo, revision=a.revision, allow_patterns=["onnx/*"], ignore_patterns=skip)) / "onnx"
out = ROOT / "assets" / "diacritizer"
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    shutil.copyfile(f, out / f.name)
    print(f"{f.stat().st_size / 1e6:8.1f} MB  {out / f.name}")
