#!/usr/bin/env python3
"""langtok.py — multi-script tokenizer shared by the bridge + aligner, so any language can become addressable
nodes in the meaning-geoform. The English pivot is always ASCII; the foreign side picks a mode:

  latin   : Latin script incl. diacritics → NFKD-stripped to ASCII ([a-z']+). (Latin, Portuguese, …)
  unicode : keep native Unicode letters, lowercased ([^\\W\\d_]{2,} with re.UNICODE). Cyrillic/Greek/Arabic/
            diacritic-Latin stay as their own forms (Бог, Bůh, Dumnezeu). Whitespace/punct delimited.
  cjk     : character-level — Han/Kana/Thai have no spaces and single chars carry meaning (神=god, 王=king,
            水=water), so each CJK char is a token (no segmenter/dictionary needed = agnostic); embedded
            Latin runs are also captured.

Registry maps a language name → (bible-api translation id, mode). Add a row to support a new language.
"""
import re, unicodedata

TOK_ASCII = re.compile(r"[a-z][a-z']+")
TOK_UNI = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# Han (CJK Unified + Ext-A + compat), Hiragana, Katakana, Thai — single chars are tokens
CJK_CHAR = re.compile(r"[㐀-鿿豈-﫿぀-ヿ฀-๿]")

# name -> (bible-api translation id, tokenizer mode). 'english' is the pivot.
REGISTRY = {
    "latin":      ("clementine", "latin"),
    "portuguese": ("almeida",    "latin"),
    "russian":    ("synodal",    "unicode"),
    "czech":      ("bkr",        "latin"),
    "romanian":   ("rccv",       "latin"),
    "chinese":    ("cuv",        "cjk"),
    "cherokee":   ("cherokee",   "unicode"),
    "german":     ("",           "latin"),
    "french":     ("",           "latin"),
    "spanish":    ("",           "latin"),
    "arabic":     ("",           "unicode"),
    "hindi":      ("",           "unicode"),
    "japanese":   ("",           "cjk"),
    "greek":      ("",           "unicode"),
    "italian":    ("",           "latin"),
    "dutch":      ("",           "latin"),
    "polish":     ("",           "latin"),
    "hebrew":     ("",           "unicode"),
    "vietnamese": ("",           "latin"),
    "indonesian": ("",           "latin"),
    "turkish":    ("",           "latin"),
    "swahili":    ("",           "latin"),
    "hungarian":  ("",           "latin"),
    "tagalog":    ("",           "latin"),
}

# BibleNLP/ebible corpus: verse-per-line files (line N = same verse via metadata/vref.txt). One download per
# language, no rate limit. name -> corpus file id (under corpus/<id>.txt). English pivot = ENG_CORPUS.
ENG_CORPUS = "eng-engDRA"      # Douay-Rheims (English translation of the Vulgate; clean alignment)
CORPUS = {
    "russian": "rus-russyn", "chinese": "cmn-cmnfeb", "latin": "lat-latVUC", "czech": "ces-ces1613",
    "german": "deu-deu1912", "french": "fra-fraLSG", "spanish": "spa-spaRV1909", "arabic": "arb-arb_vd",
    "hindi": "hin-hin2017", "japanese": "jpn-jpn1965", "portuguese": "por-porblt",
    "italian": "ita-ita1885", "dutch": "nld-nld1939", "polish": "pol-polubg", "greek": "grc-grc_tisch",
    "hebrew": "heb-heb", "vietnamese": "vie-vie1934", "indonesian": "ind-ind", "turkish": "tur-turytc",
    "swahili": "swh-swh1850", "hungarian": "hun-hun", "tagalog": "tgl-tglulb",
}


def deascii(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def tokenize(text, mode):
    if mode == "latin":
        return TOK_ASCII.findall(deascii(text))
    if mode == "unicode":
        return TOK_UNI.findall(text.lower())
    if mode == "cjk":
        return CJK_CHAR.findall(text) + TOK_ASCII.findall(deascii(text))
    raise ValueError(f"unknown tokenizer mode: {mode}")


def tokenize_en(text):
    return TOK_ASCII.findall(text.lower())


def mode_for(name):
    return REGISTRY.get(name, (None, "unicode"))[1]


def trans_for(name):
    return REGISTRY.get(name, (None, None))[0]
