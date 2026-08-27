"""
Demo: §21 Lite-classificatie & routing (optimalisaties).

Toont concreet hoe je de *categorie* van een vraag bepaalt zONDER een LLM:
 - 21.1 keyword / regex routering (snel, deterministisch)
 - 21.3 semantische routering via een lichte hash-embedding (pure stdlib)
 - 21.4 een gecombineerde router (keyword eerst, embedder als fallback)

Optioneel: een TF-IDF + LogisticRegression classifier wanneer scikit-learn
geinstalleerd is (zie onderaan, gemarkeerd met `# pip install scikit-learn`).

Draait volledig met de Python-standaardbibliotheek.
Zie docs/17-lite-classification.md voor de volledige uitleg.
"""

import hashlib
import math
import re


# --- 21.1 Keyword / regex routering -----------------------------------------
KEYWORD_RULES = {
    "financieel": r"\b(factuur|omzet|btw|betaling|q[1-4])\b",
    "support":    r"\b(crash|error|wachtwoord|login|bug)\b",
}


def route_keyword(text: str) -> str:
    for category, pattern in KEYWORD_RULES.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return "overig"


# --- 21.3 Lokale embedder + cosine (semantisch, zonder LLM) ------------------
# Pedagogische embedder op basis van een hash-seed: deterministisch, geen numpy.
def embed(text: str, dim: int = 64):
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**31)
    rng = seed
    vec = []
    for _ in range(dim):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF   # LCG, reproduceerbaar
        vec.append(rng / 0x7FFFFFFF * 2 - 1)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


EXAMPLES = {
    "financieel": embed("omzet en facturen van het kwartaal"),
    "support":    embed("problemen met inloggen en crashes"),
}


def route_embed(query: str) -> str:
    q = embed(query)
    return max(EXAMPLES, key=lambda c: cosine(q, EXAMPLES[c]))


# --- 21.4 Gecombineerde router -----------------------------------------------
def route(text: str) -> str:
    kw = route_keyword(text)
    if kw != "overig":
        return kw                 # snelle, zekere match
    return route_embed(text)      # fallback naar semantische routering


if __name__ == "__main__":
    samples = [
        "de factuur voor Q2 ontbreekt",
        "de app crasht bij login",
        "ik wil mijn abonnement opzeggen",   # geen keyword -> embedder
    ]
    for s in samples:
        print(f"{route(s):12s} <- {s!r}")

    # Optioneel: TF-IDF + LogisticRegression (vereist scikit-learn).
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        train = [
            ("Wat is de omzet in Q2?", "financieel"),
            ("Toon het factuur van klant X", "financieel"),
            ("Hoe reset ik mijn wachtwoord?", "support"),
            ("De app crasht bij login", "support"),
        ]
        X = [t for t, _ in train]
        y = [c for _, c in train]
        vec = TfidfVectorizer().fit(X)
        clf = LogisticRegression().fit(vec.transform(X), y)
        pred = clf.predict(vec.transform(["factuur voor Belgie ontbreekt"]))[0]
        print(f"\nTF-IDF classifier -> {pred}")
    except ImportError:
        print("\n(scikit-learn niet geinstalleerd; TF-IDF-branch overgeslagen)")
