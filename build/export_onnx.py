"""Export a Tashkeel-v4 checkpoint to the CPU runtime (egy_frontend.diacritizer):

    <out>/model.fp32.onnx  network: tokens + character features -> CRF emissions
    <out>/model.int8.onnx  the same, encoder weights in int8 (dynamic quantization)
    <out>/crf.npz          CRF start / end / transition scores (Viterbi runs in numpy)
    <out>/tokenizer.json   the MARBERTv2 tokenizer (run with the `tokenizers` library)

One sentence per call: the publisher's BiLSTM runs over padding, so a batch of one is
the exact (packed) forward; the export is checked against the torch model.

    .venv/bin/python build/export_onnx.py --model /data/tashkeel/runs/egy_v1/best --out assets/diacritizer
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from common_v4 import V4, forward, load  # noqa: E402

INPUTS = ["input_ids", "attention_mask", "char_to_token", "char_ids", "word_pos_ids", "morpheme_type_ids", "is_irab_pos"]


class Emissions(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, char_to_token, char_ids, word_pos_ids, morpheme_type_ids, is_irab_pos):
        m = self.m
        th = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        feats = torch.gather(th, 1, char_to_token.unsqueeze(-1).expand(-1, -1, th.size(-1)))
        x = torch.cat([feats, m.char_embed(char_ids), m.word_pos_embed(word_pos_ids), m.morpheme_embed(morpheme_type_ids)], -1)
        out, _ = m.lstm(x)
        h = m.layer_norm(out + m.proj_residual(x))
        stem, irab = m.stem_head(h), m.irab_head(h)
        return torch.where(is_irab_pos.unsqueeze(-1), 0.3 * stem + 0.7 * irab, stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "diacritizer")
    a = ap.parse_args()
    model, tok, mod = load(a.model, "cpu")
    model.eval()
    v4 = V4(mod)
    a.out.mkdir(parents=True, exist_ok=True)
    text = "الغسالة صفرت والبلكونة اتملت حمام، وأنا لسه راجع من الشغل"
    f = v4.features(tok, [text])
    args = tuple(f[k] for k in INPUTS)
    net = Emissions(model).eval()
    with torch.inference_mode():
        ref = forward(model, f, "cpu", amp=False).numpy()
        got = net(*args).numpy()
    assert np.abs(ref - got).max() < 1e-4, np.abs(ref - got).max()
    torch.onnx.export(
        net, args, a.out / "model.fp32.onnx", input_names=INPUTS, output_names=["emissions"], opset_version=17,
        dynamic_axes={**{k: {1: "tokens"} for k in INPUTS[:2]}, **{k: {1: "chars"} for k in INPUTS[2:]}, "emissions": {1: "chars"}},
        dynamo=False,
    )
    # int8 weights for the MARBERT encoder only (its matmuls and embedding tables), per
    # channel: quantizing the BiLSTM and heads too costs ~1.5 DER points, this ~0.3.
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    rest = [n.name for n in onnx.load(a.out / "model.fp32.onnx").graph.node if n.op_type == "MatMul" and "encoder" not in n.name]
    quantize_dynamic(a.out / "model.fp32.onnx", a.out / "model.int8.onnx", weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"], per_channel=True, nodes_to_exclude=rest)
    crf = model.crf
    np.savez(a.out / "crf.npz", start=crf.start_transitions.detach().numpy(), end=crf.end_transitions.detach().numpy(),
             transitions=crf.transitions.detach().numpy(), classes=np.array(mod.DIACRITICS_LIST))
    shutil.copyfile(Path(a.model) / "tokenizer.json", a.out / "tokenizer.json")
    (a.out / "SOURCE").write_text(str(Path(a.model).resolve()) + "\n")
    for p in sorted(a.out.iterdir()):
        print(f"{p.stat().st_size / 1e6:8.1f} MB  {p.name}")


if __name__ == "__main__":
    main()
