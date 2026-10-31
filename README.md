# Egyptian pronunciation frontend

Text in, text for the Egyptian TTS (Audio8, Pocket TTS) out, with diacritics added to
the words the TTS would otherwise misread: البلكونة اتملت حمام → البلكونة اتملت حَمَام.
CPU only at run time (onnxruntime, tokenizers, numpy), and self-contained: nothing is
imported from the TTS repos.

The diacritizer weights are not in this repo. Get the exported model (private Hub repo
[Rabe3/Tashkeel-v4-Egyptian](https://huggingface.co/Rabe3/Tashkeel-v4-Egyptian), `onnx/`)
into `assets/diacritizer` first:

```bash
pip install -r requirements.txt huggingface_hub
HF_TOKEN=... python build/fetch_model.py          # --int8-only skips the 684 MB fp32 model
```

```python
from egy_frontend import Frontend
fe = Frontend()                       # fp32; Frontend(quantized=True) for the int8 encoder
fe("الغسالة صفرت والفني جاي يركب التكييف.")
text, decisions = fe.analyze("...")   # plus the rule applied to each word
```

```bash
python -m egy_frontend --decisions "البلكونة اتملت حمام!"
```

## Why the TTS misreads words

The TTS learned to read from the Jais-diacritized transcripts of its training audio
(26M words). For a word written without marks it says the reading it saw most often:
حمام is حَمَّام (bathroom) 781 times and حَمَام (pigeons) 122 times, so it always says
bathroom, whatever the sentence says. A word it saw a handful of times (صفرت: 11) has no
reliable reading at all. Marks fix both: the TTS was trained with marked, partly marked
and plain text, and follows marks when they are there.

## How it works

```
text → normalize_egyptian → diacritizer (every word, in context) → policy → canonicalize_marks → TTS
```

1. **Diacritizer** (`egy_frontend/diacritizer.py`): Tashkeel-v4 (MARBERTv2 + BiLSTM +
   CRF, 172M parameters) fine-tuned on the same Jais transcripts the TTS trained on, so
   it writes marks in the conventions the TTS learned. Exported to ONNX; features and
   CRF decoding in numpy. It also returns, per letter, the CRF marginal probability of
   its mark (the confidence).
2. **Policy** (`Frontend.decide`): no list can hold every ambiguous word, so the
   frontend predicts every word and keeps marks only where they change what the TTS
   says. For each word, using the lexicon of the TTS training text
   (`assets/lexicon.json.gz`: uses and readings of all 517k words):
   - **caller**: the caller already marked it: kept;
   - **rare**: fewer than 20 uses in training (5.8% of running words): the TTS has no
     reading of its own, marked as predicted;
   - **differs**: the prediction is a different word from the TTS's default reading
     (`readings.conflict`: another shadda or vowel; elision and prefix vowels do not
     count): marked if the diacritizer is confident enough (`deviate_ambiguous` for a
     word read two ways in training, the stricter `deviate_other` for any other word);
     below that, **unsure**, left plain (an LLM can be asked about these);
   - **ambiguous**: a word read two ways in training, predicted with its default
     reading: marked (what the TTS would say anyway; marks on exactly these words are
     the TTS training's hint mode);
   - **plain**: everything else stays plain, so a diacritizer error there cannot hurt.
3. **Canonicalize** (`text.canonicalize_marks`): marks rewritten the way the training
   text writes that word.

Tags ([sighs] ...), Latin words and numbers pass through untouched. Results are cached
per sentence (LRU).

## Layout

| Path | |
|---|---|
| `egy_frontend/` | the runtime: `frontend.py` (policy), `diacritizer.py` (ONNX + CRF), `readings.py` (when two markings are different words), `text.py` (normalization and mark conventions, copied from the TTS repos) |
| `assets/` | `lexicon.json.gz` (phase 1), `diacritizer/` (exported model: `model.fp32.onnx`, `model.int8.onnx`, `crf.npz`, `tokenizer.json`) |
| `build/build_lexicon.py` | phase 1: the lexicon of the TTS training text |
| `build/build_splits.py` | phase 2: diacritizer train / validation / test splits, by video |
| `build/mine_homographs.py`, `build/build_authored.py` | phase 2: the homograph tests (`eval/`) |
| `train/` | phase 3: the trainer, vendored from tashkeel-layer (memory cap, all evaluated checkpoints kept) |
| `build/export_onnx.py` | checkpoint → `assets/diacritizer` |
| `build/fetch_model.py` | the exported model from the Hub → `assets/diacritizer` |
| `build/eval_frontend.py` | homograph scores over a grid of policy thresholds |
| `build/bench_cpu.py` | CPU latency |

## Rebuild

```bash
uv venv .venv && uv pip install -r requirements-train.txt   # see the torch line in it
python build/build_lexicon.py                 # 1 min
python build/build_splits.py                  # 1 min, -> /data/tashkeel/egy_splits
python build/mine_homographs.py && python build/build_authored.py
cd train && python train.py --base-model /data/tashkeel/Tashkeel-v4 --data-dir /data/tashkeel/egy_splits \
    --epochs 2 --evals-per-epoch 4 --mem-frac 0.135 --out /data/tashkeel/runs/egy_v1
python build/export_onnx.py --model /data/tashkeel/runs/egy_v1/best --out assets/diacritizer
python build/eval_frontend.py && python build/bench_cpu.py
```

## Evaluation

- `eval/homographs_authored.jsonl`: 285 sentences written for this test, 50 words in
  95 readings (حمام bathroom / pigeons, السنة year / sunnah, يركب ride / install, بحبك
  to a man / a woman ...), in the TTS repo's homographs.jsonl format, so
  `eval_homographs.py` there scores it end to end through the TTS. Not yet checked by a
  native speaker.
- `eval/homographs_mined.jsonl`: 25,753 occurrences of ambiguous words in held-out
  speech (videos no training split has), with the reading Jais heard in the audio:
  6,768 where it is not the TTS's default reading, 18,985 controls where it is.

## Results (Tashkeel-v4 fine-tuned 2 epochs, `/data/tashkeel/runs/egy_v1`, step 36,152)

Diacritizer, error on marked internal letters (`train/` evaluator):

| | Base Tashkeel-v4 | Fine-tuned |
|---|---:|---:|
| validation (12,000 chunks) | 15.41% | 2.99% |
| test (18,297 chunks, unseen videos) | 15.54% | **2.90%** |
| TTS held-out set (737 chunks) | 15.89% | 2.90% |

Frontend on the homograph tests (`build/eval_frontend.py`, `eval/results_fp32.json`).
Authored: share of the 285 sentences where what the TTS reads is the right word.
Mined: of the 6,931 occurrences whose reading is not the TTS default, the share
fixed; of the 18,882 controls, the share broken. Net: per 1,000 running ambiguous words
(21% of which are not the default reading in held-out speech).

| | Authored right | Mined fixed | Mined broken | Net / 1,000 |
|---|---:|---:|---:|---:|
| plain TTS | 50.2% | 0 | 0 | 0 |
| frontend, `deviate_ambiguous` 0.5 | 86.0% | 52.1% | 4.9% | +70 |
| **frontend, 0.7 (default)** | **82.5%** | 38.4% | **1.8%** | +66 |
| frontend, 0.9 | 74.4% | 24.9% | 0.8% | +46 |
| int8 encoder, 0.7 | 77.9% | 35.6% | 2.2% | +57 |

The default (0.7) gives up a little of the gain of 0.5 for less than half the
breakage: a word the TTS read right and now reads wrong is the worst outcome.

The diacritizer alone, on the word, agrees with the reference 94% of the time on default
readings but only 59% (vowel) and 45% (shadda) on rare ones: it learned the same skew as
the TTS from the same transcripts. Words whose rare reading is very rare in training
still fail: البلكونة اتملت حمام (pigeons, predicted bathroom), الفني هيركب التكييف
(install, predicted ride, at confidence 1.0), الغسالة صفرت. Fixing these needs training
text where the rare readings are frequent (see Next steps).

CPU latency, whole call per sentence (showcase sentences, `build/bench_cpu.py`):

| | 1 thread | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| fp32 (684 MB) | 35 ms | 23 ms | 14 ms | 17 ms |
| int8 encoder (198 MB) | 20 ms | 14 ms | 13 ms | 14 ms |

## Next steps

- Rare readings: correctly marked sentences for the rare sense of the most skewed
  homographs (authored or LLM-generated), oversampled in a short continued fine-tune.
- LLM fallback for "unsure" words (Decision.unsure).
- A native speaker's pass over `eval/authored_homographs.py`; end-to-end scoring through
  the TTS with `eval_homographs.py` (TTS repo) on `eval/homographs_authored.jsonl`.

## Licence

Tashkeel-v4 and models fine-tuned from it are under the Tashkeel licence (separate
licence held). The training text is the TTS training transcripts.
