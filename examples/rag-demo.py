"""
rag-demo.py  --  illustratie bij §8 (RAG — Retrieval Augmented Generation)

Toont de KERNFASEN van RAG (§8.2) zonder externe dependencies:

  INDEXEERFASE (offline):  document -> chunk -> embed -> store in vector-DB
  RETRIEVAL (online):      vraag -> embed -> similarity top-k -> augmentatie

Embedden doen we hier met een **deterministische hash-based** vector, zodat de
cosine-wiskunde echt en reproduceerbaar is. In productie vervang je
`_embed()` door een echt embedding-model (sentence-transformers / OpenAI
text-embedding-3-small, §8.4) — en de vector-DB door FAISS/Chroma/Qdrant (§8.5).
Belangrijk: index én query gebruiken **hetzelfde** embed-model (§8.2).

Vereenvoudiging: de hash-embedder heeft géén taalkundig begrip; twee
verschillende zinnen krijgen hier willekeurige vectoren. Het toont enkel het
MECHANISME (chunk -> embed -> top-k cosine).
"""

# pip install numpy   (wordt gebruikt voor de dot-product in retrieval)
import numpy as np


# ----------------------------------------------------------------------------
# Embedder (placeholder) —归一leer zodat cosine == dot product
# ----------------------------------------------------------------------------
def _embed(text: str, dim: int = 32) -> np.ndarray:
    """Deterministische, genormaliseerde vector (GEEN semantisch begrip)."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------
# Chunking (§8.3) — recursief op paragrafen/zinnen, met overlap
# ----------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces = []
    for p in paragraphs:
        pieces.extend([s + " " for s in p.split(". ")])
    chunks, current = [], ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + piece
    if current:
        chunks.append(current.strip())
    return chunks


# ----------------------------------------------------------------------------
# Indexeerfase (offline) — §8.2 stappen 1-4
# ----------------------------------------------------------------------------
def build_index(docs: list, chunker=chunk_text) -> list:
    store = []
    for doc in docs:
        for chunk in chunker(doc):
            store.append({"text": chunk, "vec": _embed(chunk)})
    return store


# ----------------------------------------------------------------------------
# Retrieval (online) — §8.2 stappen 5-8
# ----------------------------------------------------------------------------
def retrieve(store: list, question: str, top_k: int = 2) -> list:
    q = _embed(question)
    scored = [(float(np.dot(r["vec"], q)), r["text"]) for r in store]
    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]


def augment(question: str, chunks: list) -> str:
    context = "\n---\n".join(chunks)
    return f"Context:\n{context}\n\nVraag: {question}"


if __name__ == "__main__":
    docs = [
        "Onze Q2-omzet in Belgie was sterk. De groei kwam vooral uit software.",
        "Het partnerkanaal in Nederland leverde een stabiele bijdrage.",
        "In Frankrijk daalde de verkoop door vertraging in de logistiek.",
    ]
    store = build_index(docs)
    print(f"Geindexeerde chunks: {len(store)}")

    question = "Hoe was de omzet in Belgie?"
    chunks = retrieve(store, question, top_k=2)
    print("\nTop-k opgehaalde chunks:")
    for c in chunks:
        print("  -", c)

    prompt = augment(question, chunks)
    print("\nGeaugmenteerde prompt (klaar voor de LLM, §4/§5):")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("\nZie 06-rag.md (§8) voor re-ranking (§8.6), hybrid search en",
          "evaluatiemetrics (§8.7).")
