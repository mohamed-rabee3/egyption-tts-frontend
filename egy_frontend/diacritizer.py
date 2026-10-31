"""CPU diacritizer: the Egyptian Tashkeel-v4 network as ONNX (build/export_onnx.py), with
the publisher's features and CRF decoding reimplemented in numpy. Needs onnxruntime,
tokenizers and numpy only.

Each call diacritizes one sentence (the exported network runs one at a time, which is
also its exact forward); text longer than MAX_CHARS is cut at word boundaries. Besides
the marked text it returns, per character, the CRF marginal probability of the chosen
mark: the policy's confidence.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_CHARS = 254
LETTERS = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")
PROCLITICS = ["وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل", "س"]
ENCLITICS = ["هما", "كما", "هم", "هن", "كم", "كن", "نا", "ها", "وا", "ون", "ين", "ان", "ات", "ه", "ك", "ي"]
PROCLITIC, STEM, ENCLITIC = 1, 2, 3


def segment(word: str) -> list[tuple[int, int, int]]:
    """The publisher's morpheme segmentation (modeling_tashkeel.segment_word_morphemes)."""
    n = len(word)
    if n <= 3 or not any(c in LETTERS for c in word):
        return [(0, n, STEM)]
    p = next((len(x) for x in PROCLITICS if word.startswith(x) and n - len(x) >= 2), 0)
    rest = word[p:]
    e = next((len(x) for x in ENCLITICS if rest.endswith(x) and len(rest) - len(x) >= 2), 0)
    spans = [(0, p, PROCLITIC)] if p else []
    spans.append((p, n - e, STEM))
    if e:
        spans.append((n - e, n, ENCLITIC))
    return spans


@dataclass
class Result:
    text: str                 # the input with marks added
    marks: list[str]          # per input character, the mark added ("" for none)
    confidence: np.ndarray    # per input character, P(that mark) under the CRF


class Diacritizer:
    def __init__(self, model_dir: str | os.PathLike, quantized: bool = True, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(512)
        crf = np.load(d / "crf.npz")
        self.start, self.end, self.trans = crf["start"], crf["end"], crf["transitions"]
        self.classes = [str(c) for c in crf["classes"]]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        path = d / ("model.int8.onnx" if quantized else "model.fp32.onnx")
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def features(self, text: str) -> dict[str, np.ndarray]:
        enc = self.tok.encode(text)
        offsets, n_tok = enc.offsets, len(enc.ids)
        wpos, morph, irab = [], [], []
        for w in text.split(" "):
            types, ends = [0] * len(w), [False] * len(w)
            for s, e, t in segment(w):
                types[s:e] = [t] * (e - s)
                if t == STEM and e > s:
                    ends[e - 1] = True
            for i in range(len(w)):
                wpos.append(min(i, 31))
                morph.append(types[i])
                irab.append(ends[i] or i == len(w) - 1)
            wpos.append(0)
            morph.append(0)
            irab.append(False)
        L = len(text)
        c2t = np.zeros(L, np.int64)
        k = 0
        for c in range(L):
            while k < len(offsets) and offsets[k][1] <= c:
                k += 1
            c2t[c] = min(k, n_tok - 1)
        b = lambda x, dt=np.int64: np.asarray(x, dt)[None]  # noqa: E731
        return {"input_ids": b(enc.ids), "attention_mask": b(enc.attention_mask), "char_to_token": c2t[None],
                "char_ids": b([min(ord(c), 255) for c in text]), "word_pos_ids": b(wpos[:L]),
                "morpheme_type_ids": b(morph[:L]), "is_irab_pos": b(irab[:L], bool)}

    def _decode(self, em: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Viterbi path and the CRF marginal of each chosen tag (forward-backward)."""
        L, K = em.shape
        score, back = self.start + em[0], np.zeros((L, K), np.int64)
        for t in range(1, L):
            s = score[:, None] + self.trans
            back[t] = s.argmax(0)
            score = s.max(0) + em[t]
        path = np.zeros(L, np.int64)
        path[-1] = int((score + self.end).argmax())
        for t in range(L - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        lse = lambda x, ax: (m := x.max(ax, keepdims=True)) + np.log(np.exp(x - m).sum(ax, keepdims=True))  # noqa: E731
        alpha, beta = np.zeros((L, K)), np.zeros((L, K))
        alpha[0] = self.start + em[0]
        for t in range(1, L):
            alpha[t] = lse(alpha[t - 1][:, None] + self.trans, 0)[0] + em[t]
        beta[-1] = self.end
        for t in range(L - 2, -1, -1):
            beta[t] = lse(self.trans + (em[t + 1] + beta[t + 1])[None], 1)[:, 0]
        logz = lse(alpha[-1] + self.end, 0)[0]
        marg = np.exp(alpha + beta - logz)
        return path, marg[np.arange(L), path]

    def _one(self, text: str) -> tuple[list[str], np.ndarray]:
        em = self.sess.run(None, self.features(text))[0][0].astype(np.float64)
        path, conf = self._decode(em)
        marks = [self.classes[p] if c in LETTERS else "" for c, p in zip(text, path)]
        # Hamza below alef always carries kasra; the publisher adds it after decoding.
        marks = ["ِ" if c == "إ" and not m else m for c, m in zip(text, marks)]
        return marks, conf

    def __call__(self, text: str) -> Result:
        """`text` without marks (a mark in it is taken as a letter the model does not know)."""
        if not text:
            return Result("", [], np.zeros(0))
        marks, conf, start = [], [], 0
        for piece in _pieces(text):
            m, c = self._one(piece)
            marks += m
            conf.append(c)
            start += len(piece)
            if start < len(text):  # the space between pieces
                marks.append("")
                conf.append(np.ones(1))
                start += 1
        out = "".join(c + m for c, m in zip(text, marks))
        return Result(out, marks, np.concatenate(conf))


def _pieces(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    out, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > MAX_CHARS:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    out.append(cur)
    return out
