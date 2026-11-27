"""Fetch, clean and segment a Project Gutenberg text.

Sentence segmentation is rule-based and deterministic, so that every number in
the book can be reproduced exactly:
  * paragraph breaks always end a sentence;
  * inside a paragraph a sentence ends at . ! ? (optionally followed by closing
    quotes or brackets) when the next token starts with an upper-case letter,
    a digit or an opening quote;
  * no break after honorifics (Mr. Mrs. Dr. ...) or single-letter initials
    ("K.", "J. K."), which are frequent in Kafka.
"""
import os
import re
import urllib.request
from dataclasses import dataclass, field

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
MIRRORS = ["https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
           "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"]
ABBREV = {"mr", "mrs", "ms", "dr", "st", "mt", "messrs", "prof", "rev", "hon", "capt", "col",
          "gen", "lt", "sgt", "jr", "sr", "vs", "etc", "no", "vol", "ch", "fig", "herr", "frau", "fräulein"}


def fetch(gid):
    os.makedirs(os.path.join(CACHE, "texts"), exist_ok=True)
    path = os.path.join(CACHE, "texts", f"{gid}.txt")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return open(path, encoding="utf-8").read()
    last = None
    for m in MIRRORS:
        try:
            req = urllib.request.Request(m.format(id=gid), headers={"User-Agent": "labyrinth-book-builder"})
            data = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
            if "*** START" in data:
                open(path, "w", encoding="utf-8").write(data)
                return data
        except Exception as e:  # try the next mirror
            last = e
    raise RuntimeError(f"could not fetch Gutenberg #{gid}: {last}")


def strip_gutenberg(raw):
    s = raw.find("*** START")
    s = raw.find("\n", s) + 1
    e = raw.find("*** END")
    return raw[s:e].replace("\r\n", "\n")


@dataclass
class Sentence:
    i: int              # index in the book
    text: str
    chapter: int        # 0-based chapter index
    paragraph: int
    char_offset: int    # offset in the cleaned body text
    words: list = field(default_factory=list)


@dataclass
class Book:
    title: str
    author: str
    translator: str
    gid: int
    sentences: list
    chapters: list      # chapter titles, in order
    chapter_starts: list  # sentence index where each chapter starts
    subtitles: list = field(default_factory=list)


_END = re.compile(r'([.!?]+)(["”’\')\]]*)\s+(?=["“‘(\[]?[A-Z0-9])')


def _split_paragraph(p):
    out, start = [], 0
    for m in _END.finditer(p):
        before = p[start:m.start()]
        last = re.search(r"([A-Za-zÄÖÜäöüß]+)$", before)
        tok = last.group(1) if last else ""
        if m.group(1) == "." and (len(tok) == 1 and tok.isupper() or tok.lower() in ABBREV):
            continue
        out.append(p[start:m.end()].strip())
        start = m.end()
    tail = p[start:].strip()
    if tail:
        out.append(tail)
    return out


def load(gid, title, author, translator="", chapter_pattern=r"^\s*(Chapter|CHAPTER)\s+\S+.*$",
         skip_until=None):
    body = strip_gutenberg(fetch(gid))
    if skip_until:  # drop front matter (title page, copyright lines) before the first chapter
        k = re.search(skip_until, body, re.M)
        body = body[k.start():] if k else body
    paras = re.split(r"\n\s*\n", body)
    sentences, chapters, chapter_starts, subtitles = [], [], [], []
    chap, offset, pnum, just_headed = -1, 0, 0, False
    cre = re.compile(chapter_pattern)
    for raw in paras:
        p = re.sub(r"\s+", " ", raw).strip()
        offset = body.find(raw.strip()[:40], offset) if raw.strip() else offset
        if not p:
            continue
        if cre.match(p) and len(p) < 80:
            chap += 1
            chapters.append(p)
            chapter_starts.append(len(sentences))
            subtitles.append("")
            just_headed = True
            continue
        if chap < 0:  # text before the first chapter heading
            continue
        if just_headed and not re.search(r'[.!?"”’]$', p) and len(p) < 200:
            subtitles[-1] = p          # a chapter subtitle, not a sentence
            just_headed = False
            continue
        just_headed = False
        if p in ("End", "THE END", "The End"):
            continue
        for s in _split_paragraph(p):
            words = re.findall(r"[A-Za-zÀ-ÿ'’]+", s)
            if not words:
                continue
            sentences.append(Sentence(len(sentences), s, chap, pnum, max(offset, 0), words))
        pnum += 1
    return Book(title, author, translator, gid, sentences, chapters, chapter_starts, subtitles)
