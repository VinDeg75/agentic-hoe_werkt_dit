"""
tokenizer-demo.py  --  illustratie bij §3.1 (Tokenisatie)

Dit script toont CONCREET wat er in de tokenizer gebeurt:
  1. Een mini-BPE (Byte-Pair Encoding) die we vanaf nul trainen op een klein
     corpus, zodat je het "merge"-proces letterlijk ziet.
  2. (Optioneel) de echte `tiktoken`-tokenizer van OpenAI, om te tonen dat
     "agentic" vaak als MEERDERE subwoorden getokeniseerd wordt.

De code is pedagogisch: ze toont het PRINCIPE correct, maar is geen
productie-tokenizer (geen vocabulaire-clipping, geen special tokens, enz.).
Vereenvoudigingen staan telkens in commentaar.
"""


# Uitbreidingen op dit script (voorheen TODO's):
#   * §0 hieronder: controlkarakters wegfilteren zoals een echte tokenizer.
#   * __main__ (onder): corpus met de VOLLEDIGE ASCII-tabel (0..127).

# ----------------------------------------------------------------------------
# 0) Voorbehandeling: control characters (zoals een echte tokenizer)
# ----------------------------------------------------------------------------
# Echte tokenizers krijgen vaak "vieze" input: naast zichtbare tekens zitten er
# ook Control Characters in (ASCII 0x00-0x1F en 0x7F) - NUL-bytes, newlines,
# tabs, carriage returns, enz. die uit bestanden, netwerk of copy/paste komen.
# BPE zou die als gewone symbolen tellen en het vocabulaire vervuilen. Net als
# echte tokenizers normaliseren we de input: control characters worden verwijderd
# (of vervangen door een spatie) voordat er getokeniseerd wordt.
CONTROL_CHARS = set(chr(c) for c in list(range(0, 32)) + [127])


def strip_control_chars(text, replace_with=" "):
    """Verwijder ASCII-controlkarakters uit `text`.

    In plaats van ze als aparte tokens te leren, vervangen we ze door
    `replace_with` (standaard een spatie) zodat woorden niet aan elkaar
    plakken, of - bij een lege vervanger - volledig verwijderd.
    """
    return "".join(
        ch if ch not in CONTROL_CHARS else replace_with for ch in text
    )


# ----------------------------------------------------------------------------
# 1) Mini-BPE vanaf nul (geen externe dependencies)
# ----------------------------------------------------------------------------
from collections import Counter
import re


def get_pairs(word):
    """Alle naastgelegen symbol-paren in een 'word' (lijst van symbols)."""
    return set(zip(word[:-1], word[1:]))


def train_bpe(corpus, vocab_size=8):
    """Train een mini-BPE: merge steeds het frequentste symbol-paar.

    Vereenvoudiging: we tellen woorden als losse strings en voegen een
    expliciet woord-eind-token '</w>' toe, zodat 'agent' en 'agents' verschillend
    afgesloten worden (net als echte tokenizers). Control characters worden
    vóór training weggefilterd (zie §0).
    """
    corpus = strip_control_chars(corpus)
    words = re.findall(r"\w+", corpus.lower())
    # start: elk woord opgesplitst in karakters + een woord-eind-symbool
    tokens = [list(w) + ["</w>"] for w in words]
    merges = []

    while len(merges) < vocab_size:
        counts = Counter()
        for t in tokens:
            counts.update(get_pairs(t))
        if not counts:
            break
        best = max(counts, key=counts.get)      # frequentste paar
        merges.append(best)

        a, b = best
        new_tokens = []
        for t in tokens:
            nt, i = [], 0
            while i < len(t):
                # smelt (a, b) samen tot één symbool waar mogelijk
                if i < len(t) - 1 and t[i] == a and t[i + 1] == b:
                    nt.append(a + b)
                    i += 2
                else:
                    nt.append(t[i])
                    i += 1
            new_tokens.append(nt)
        tokens = new_tokens

    return merges


def apply_bpe(text, merges):
    """Pas geleerde merges toe op nieuwe tekst (encodering).

    Control characters worden eerst weggefilterd (zie §0), zodat dezelfde
    normalisatie geldt als bij training.
    """
    text = strip_control_chars(text)
    words = re.findall(r"\w+", text.lower())
    out = []
    for w in words:
        symbols = list(w) + ["</w>"]
        for a, b in merges:
            # herhaal merge totdat geen (a,b) meer naast elkaar staat
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        out.extend(symbols)
    return out


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Uitbreiding TODO #2: meer trainingsdata + VOLLEDIGE ASCII-dekking.
    # We bouwen een 'vieze' corpus-string die de HELE ASCII-tabel bevat
    # (0..127), inclusief control characters zoals die in echte data zitten
    # (NUL-bytes, newlines, carriage returns, ...). Zo ziet de BPE het
    # volledige tekenbereik en leert hij elk ASCII-teken als basistoken.
    # -------------------------------------------------------------------
    raw_ascii = "".join(chr(c) for c in range(0, 128))
    dirty_corpus = (
        "agents agentic agent wise agency token learn merge "
        "the quick brown fox jumps over the lazy dog 1234567890 "
        + raw_ascii
    )

    # Uitbreiding TODO #1: control characters wegfilteren vóór training
    # (net als een echte tokenizer normaliseert). Tel hoeveel we verwijderen.
    n_control = sum(1 for ch in dirty_corpus if ch in CONTROL_CHARS)
    corpus = strip_control_chars(dirty_corpus)
    print(f"Voorbehandeling: {n_control} controlkarakters verwijderd uit corpus")

    merges = train_bpe(corpus, vocab_size=8)
    print("\nGeleerde merges (paar -> samengesmolten symbool):")
    for m in merges:
        print("  ", m)

    encoded = apply_bpe("agentic agents", merges)
    print("\n'agentic agents'  ->", encoded)
    print(f"  aantal subwoorden: {len(encoded)}")

    # -------------------------------------------------------------------
    # Visuele demonstratie van control-character-normalisatie op ruwe input
    # -------------------------------------------------------------------
    sample = "hello\x00world\r\nagentic"
    print("\nVoorbeeld normalisatie van ruwe gebruikersinput:")
    print("  ruw   :", repr(sample))
    print("  schoon:", repr(strip_control_chars(sample)))

    # ------------------------------------------------------------------------
    # 2) Echte tokenizer (optioneel) -- toont variatie in token-aantal per model
    # ------------------------------------------------------------------------
    # pip install tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")   # GPT-3.5/4 vocab
        text = "Agentic AI werkt met tokens."
        ids = enc.encode(text)
        print("\ntiktoken (cl100k_base):")
        print("  token-IDs :", ids)
        print("  decode    :", [enc.decode([i]) for i in ids])
        print(f"  {len(ids)} tokens voor {len(text)} karakters "
              f"(~{len(text) / len(ids):.1f} kar/token)")
        print("  -> 'agentic' wordt vaak NIET als 1 woord getokeniseerd;")
        print("     dat beïnvloedt de kosten (zie §15 token economics).")
    except ImportError:
        print("\n(tiktoken niet geïnstalleerd -> stap 2 overgeslagen;"
              " `pip install tiktoken` om het te zien)")
