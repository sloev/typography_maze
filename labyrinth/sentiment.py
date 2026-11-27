"""AFINN-165 sentiment (Nielsen 2011; lexicon under ODbL, fetched into cache/).

Names are not feelings. A word is not scored when it is capitalised and does not
start a sentence or a quotation: "Miss Bürstner", "Block", "Grace" are names or
titles. A capitalised word counts as a start when it is the first word, or follows
an opening quotation mark or bracket, or follows . ! ? : (so '"Yes," said K.' and
'he said, "Please ...' are scored). Lower-case "miss", "block" and "grace" are scored.
"""
import os
import re
import urllib.request

from .text import CACHE

URL = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"
_lex = None
_WORD = re.compile(r"[A-Za-zÀ-ÿ'’]+")
_OPENERS = set("\"“‘'([")
_ENDERS = set(".!?:")


def lexicon():
    global _lex
    if _lex is None:
        path = os.path.join(CACHE, "AFINN-en-165.txt")
        if not os.path.exists(path):
            os.makedirs(CACHE, exist_ok=True)
            urllib.request.urlretrieve(URL, path)
        _lex = {}
        for line in open(path, encoding="utf-8"):
            w, s = line.rstrip("\n").rsplit("\t", 1)
            _lex[w] = int(s)
    return _lex


def key(w):
    return w.lower().replace("’", "'").strip("'")


def _tokens(text):
    """(token, core word, is-a-start) for every whitespace token."""
    out, prev_end = [], True
    for tok in text.split():
        m = _WORD.search(tok)
        w = m.group(0).strip("'’") if m else ""
        opened = bool(m) and any(c in _OPENERS for c in tok[:m.start()])
        out.append((tok, w, prev_end or opened))
        prev_end = tok.rstrip("\"”’')]")[-1:] in _ENDERS
    return out


def book_names(texts, share=0.5, min_count=3):
    """Lexicon words used as names in this book: capitalised away from a sentence or
    quotation start in at least `share` of their occurrences (e.g. Miss, Block).
    These are not scored anywhere in the book, not even at a sentence start."""
    lx = lexicon()
    total, mid_cap = {}, {}
    for t in texts:
        for _, w, start in _tokens(t):
            k = key(w)
            if not w or k not in lx:
                continue
            total[k] = total.get(k, 0) + 1
            if w[:1].isupper() and not start and w != "I":
                mid_cap[k] = mid_cap.get(k, 0) + 1
    return frozenset(k for k, n in mid_cap.items() if n >= min_count and n / total[k] >= share)


def token_valences(text, names=frozenset()):
    """Valence of every whitespace token of a sentence, in AFINN points (names give 0).

    Returns (tokens, values), with one entry per whitespace-separated token.
    """
    lx = lexicon()
    toks = text.split()
    vals = []
    prev_end = True              # the sentence start counts as a start
    for tok in toks:
        m = _WORD.search(tok)
        if not m:
            vals.append(0)
        else:
            w = m.group(0).strip("'’")
            opened = any(c in _OPENERS for c in tok[:m.start()])
            start = prev_end or opened
            name = (w[:1].isupper() and not start and w != "I" and not w.startswith(("I'", "I’"))) \
                or key(w) in names
            vals.append(0 if (name or not w) else lx.get(key(w), 0))
        prev_end = tok.rstrip("\"”’')]")[-1:] in _ENDERS
    return toks, vals


def score_text(text, names=frozenset()):
    """A sentence's AFINN sum, names excluded."""
    return sum(token_valences(text, names)[1])


def turn(sc, dead_zone=0):
    """+1 = left (positive sentence), -1 = right (negative), 0 = straight."""
    if sc > dead_zone:
        return 1
    if sc < -dead_zone:
        return -1
    return 0
