"""AFINN-165 sentence sentiment (Nielsen 2011; lexicon under ODbL, fetched into cache/)."""
import os
import re
import urllib.request
from .text import CACHE

URL = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"
_lex = None


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


def score(words):
    lx = lexicon()
    return sum(lx.get(w.lower().replace("’", "'"), 0) for w in words)


def turn(sc, dead_zone=0):
    """+1 = left (positive sentence), -1 = right (negative), 0 = straight."""
    if sc > dead_zone:
        return 1
    if sc < -dead_zone:
        return -1
    return 0
