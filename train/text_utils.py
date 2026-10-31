"""Diacritic units, stripping and weak-label encoding shared by data preparation, training and evaluation.

The 15-class mark table is the Shakkelha class set (arabic-vocalization, RNN_REV_CLASSES_MAPPING[0..14]),
inlined so this repo does not need the vendored Shakkelha code. Class ids are only used by the evaluator."""
import re, unicodedata

FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN = (chr(c) for c in range(0x064B, 0x0653))
MARKS = {FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN, 'ٰ'}
ARABIC = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي')
CLASSES = ['', FATHA, FATHATAN, DAMMA, DAMMATAN, KASRA, KASRATAN, SUKUN, SHADDA,
           SHADDA+FATHA, SHADDA+FATHATAN, SHADDA+DAMMA, SHADDA+DAMMATAN, SHADDA+KASRA, SHADDA+KASRATAN]
LABELS = {''.join(sorted(v)): k for k, v in enumerate(CLASSES)}

def units(text):
    """NFC text -> [[base_char, marks], ...]; marks are attached to the preceding character."""
    out=[]
    for c in unicodedata.normalize('NFC', text):
        if c in MARKS:
            if out: out[-1][1]+=c
        else: out.append([c,''])
    return out

def strip(text): return ''.join(c for c,_ in units(text))

def encode_target(text):
    """Per bare character: (class id or -100, is non-final letter, has an explicit mark).
    Letters of a word with no marks at all are -100 (unknown); unmarked letters inside a marked word are class 0."""
    u=units(text); bare=''.join(c for c,_ in u)
    marked=set()
    for m in re.finditer(r'[ء-ي]+', bare):
        if any(u[i][1] for i in range(m.start(),m.end())):
            marked.update(range(m.start(),m.end()))
    y=[]; internal=[]; annotated=[]
    for i,(c,d) in enumerate(u):
        label=LABELS.get(''.join(sorted(set(d))),-100)
        y.append(label if c in ARABIC and i in marked else -100)
        internal.append(i+1<len(u) and u[i+1][0] in ARABIC)
        annotated.append(bool(d))
    return y, internal, annotated
