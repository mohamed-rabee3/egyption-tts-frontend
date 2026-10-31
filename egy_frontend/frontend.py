"""The pronunciation frontend: Egyptian text in, text for the TTS out, with marks on the
words the TTS would otherwise misread.

    from egy_frontend import Frontend
    fe = Frontend()                      # assets/diacritizer + assets/lexicon.json.gz
    fe("البلكونة اتملت حمام!")            # -> "البلكونة اتملت حَمَام!"

Per sentence: normalize_egyptian; the diacritizer predicts marks for every word in
context; the policy (`decide`) keeps a word's marks only where they help. What the TTS
says for a plain word is its default reading (the training text's most used one), so:

- caller: the caller marked the word; the caller's marks are kept;
- rare: the training text has the word fewer than `rare_uses` times, so the TTS has no
  reliable reading of its own: marked as predicted;
- differs: the prediction is a different word from the default (readings.DIFFERENT:
  another shadda or vowel): the TTS would misread the word. Marked only if the
  diacritizer is confident (lowest CRF marginal over the word's letters but the last):
  >= `deviate_ambiguous` if the prediction is a reading the training text has (>= 2
  uses: شَغَّل beside the default شُغْل, however rare), >= `deviate_other` if it is not,
  which is more often a diacritizer error. Below the threshold: "unsure", left plain
  (where a caller can ask an LLM);
- ambiguous: a hint-lexicon word predicted with its default reading: marked (no risk, it
  is what the TTS would say; marks on these words are the training's hint mode);
- plain: everything else.

Marked words are then rewritten in the training text's conventions (canonicalize_marks).
"""
import gzip
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .diacritizer import Diacritizer
from .readings import DIFFERENT, conflict
from .text import (ARABIC_WORD, TAGS, MarkLexicon, canonicalize_marks, normalize_egyptian, split_edges,
                   strip_marks)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
_TAG = re.compile("|".join(re.escape(t) for t in TAGS))


@dataclass
class Decision:
    word: str            # as it reached the frontend (normalized)
    predicted: str       # the diacritizer's marking
    output: str          # what the TTS gets
    rule: str            # caller / rare / ambiguous / differs / plain / unsure / other
    uses: int            # occurrences in the TTS training text
    default: str | None  # the training text's most used reading
    confidence: float

    @property
    def unsure(self) -> bool:
        return self.rule == "unsure"


class Frontend:
    def __init__(self, model_dir=ASSETS / "diacritizer", lexicon=ASSETS / "lexicon.json.gz", quantized: bool = False,
                 threads: int | None = None, rare_uses: int = 20, deviate_ambiguous: float = 0.7,
                 deviate_other: float = 0.95, cache: int = 4096):
        with gzip.open(lexicon, "rt", encoding="utf-8") as fh:
            lex = json.load(fh)
        self.uses: dict[str, int] = lex["uses"]
        self.default = {k: v[0][2] for k, v in lex["readings"].items() if v}
        # readings attested >= 2 times, but the default: the alternatives a prediction may pick
        self.alternatives = {k: [f for _, n, f in v[1:] if n >= 2] for k, v in lex["readings"].items() if len(v) > 1}
        self.ambiguous = frozenset(lex["ambiguous"])
        self.marks = MarkLexicon({k: [(f, n) for f, n in v] for k, v in lex["forms"].items()},
                                 frozenset(lex["known"]), self.ambiguous)
        self.diacritizer = Diacritizer(model_dir, quantized=quantized, threads=threads)
        self.rare_uses, self.deviate_ambiguous, self.deviate_other = rare_uses, deviate_ambiguous, deviate_other
        self.analyze = lru_cache(maxsize=cache)(self._analyze) if cache else self._analyze

    def __call__(self, text: str) -> str:
        return self.analyze(text)[0]

    def decide(self, core: str, pred: str, conf: float, **thresholds) -> str:
        """The rule for one word (thresholds default to the instance's)."""
        rare_uses = thresholds.get("rare_uses", self.rare_uses)
        if core != strip_marks(core):
            return "caller"
        if self.uses.get(core, 0) < rare_uses:
            return "rare"
        d = self.default.get(core)
        if d and conflict(pred, d) in DIFFERENT:
            attested = any(conflict(pred, f) not in DIFFERENT for f in self.alternatives.get(core, ()))
            need = thresholds.get("deviate_ambiguous", self.deviate_ambiguous) if attested else \
                thresholds.get("deviate_other", self.deviate_other)
            return "differs" if conf >= need else "unsure"
        return "ambiguous" if core in self.ambiguous else "plain"

    def _analyze(self, text: str) -> tuple[str, tuple[Decision, ...]]:
        """(text for the TTS, the decision on each Arabic word)."""
        norm = normalize_egyptian(text)
        tokens = norm.split(" ") if norm else []
        model_idx = [i for i, t in enumerate(tokens) if not _TAG.fullmatch(t) and strip_marks(t)]
        res = self.diacritizer(" ".join(strip_marks(tokens[i]) for i in model_idx))
        pred_tokens = res.text.split(" ") if model_idx else []
        assert len(pred_tokens) == len(model_idx), (norm, res.text)
        # per model token, its slice of the per-character confidences
        spans, pos = [], 0
        for i in model_idx:
            n = len(strip_marks(tokens[i]))
            spans.append((pos, pos + n))
            pos += n + 1
        out, decisions = list(tokens), []
        for k, i in enumerate(model_idx):
            lead, core, trail = split_edges(tokens[i])
            if not core or not ARABIC_WORD.match(core):
                continue
            pred = split_edges(pred_tokens[k])[1]
            s, e = spans[k]
            s += len(lead)
            conf = res.confidence[s:s + max(1, len(strip_marks(core)) - 1)]
            c = float(conf.min()) if len(conf) else 1.0
            rule = self.decide(core, pred, c)
            word = core if rule == "caller" else pred if rule in ("rare", "ambiguous", "differs") else strip_marks(core)
            out[i] = lead + word + trail
            sk = strip_marks(core)
            decisions.append(Decision(core, pred, word, rule, self.uses.get(sk, 0), self.default.get(sk), round(c, 3)))
        final = canonicalize_marks(normalize_egyptian(" ".join(out)), self.marks)
        # report the canonical forms
        canon = {i: split_edges(t)[1] for i, t in enumerate(final.split(" "))} if len(final.split(" ")) == len(out) else {}
        if canon:
            j = 0
            for k, i in enumerate(model_idx):
                if j < len(decisions) and decisions[j].word == split_edges(tokens[i])[1]:
                    decisions[j].output = canon.get(i, decisions[j].output)
                    j += 1
        return final, tuple(decisions)
