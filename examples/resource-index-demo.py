"""
Resource Index demo (AGENTS.md §19 / docs/15-project-file-index.md)

Shows how an agent builds a *structured* catalog of a repo (file paths, types
and shallow content metadata) WITHOUT embeddings, and how it decides between
the Resource Index and RAG:

  - "bestaat X? / welke functie / welke sectie?" -> Resource Index (structureel)
  - "wat staat erin over X?"                    -> RAG (§8, semantisch)

This mirrors the approach in docs/15-project-file-index.md:
  - Python symbols via the stdlib `ast` module (deterministisch, geen embeddings)
  - Markdown headings via regex (H1-H3)
  - one JSON manifest with one entry per file
  - the rag_indexed rule: knowledge-base docs get NO headings in the index

Pure standard library so it runs without installs:
    python examples/resource-index-demo.py
"""

import ast
import json
import os
import re

# ---------------------------------------------------------------------------
# Extraction primitives (mirror docs/15 section 19.1)
# ---------------------------------------------------------------------------

CODE_EXT = {".py": "python", ".ts": "typescript", ".rs": "rust", ".go": "go"}
TEXT_EXT = {".md", ".txt", ".rst"}
DATA_EXT = {".json", ".yaml", ".toml"}


def extract_python_symbols(source: str) -> dict:
    """Symbolen uit Python met `ast` (deterministisch, tree-sitter-achtig)."""
    tree = ast.parse(source)
    functions = sorted({n.name for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)})
    classes = sorted({n.name for n in ast.walk(tree)
                      if isinstance(n, ast.ClassDef)})
    # globale constanten: velden zoals MAX_TOKENS, DEFAULT_MODEL (hoofdletters)
    globals_ = sorted({n.targets[0].id for n in ast.walk(tree)
                       if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id.isupper()})
    return {"functions": functions, "classes": classes, "globals": globals_}


def extract_headings(source: str, max_level: int = 3) -> list[str]:
    """Koppen H1-H3 uit Markdown (zelfde regel als docs/15 19.2)."""
    heads = []
    for line in source.splitlines():
        s = line.lstrip()
        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            if 1 <= level <= max_level:
                heads.append(s[level:].strip())
    return heads


# ---------------------------------------------------------------------------
# Repo indexer (mirror docs/15 section 19.3)
# ---------------------------------------------------------------------------

def index_repo(root: str, rag_indexed_paths: set[str] | None = None) -> dict:
    """Doorloop de repo en bouw één manifest met één entry per bestand."""
    rag_indexed_paths = rag_indexed_paths or set()
    files = []
    for dirpath, _, names in os.walk(root):
        # skip niet-relevante mappen
        if any(skip in dirpath for skip in ("/.git", "/.venv", "/node_modules")):
            continue
        for name in names:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext in CODE_EXT:
                    src = open(path, encoding="utf-8").read()
                    entry = {"path": rel, "type": "code",
                             "language": CODE_EXT[ext],
                             "symbols": extract_python_symbols(src)}
                elif ext in TEXT_EXT:
                    src = open(path, encoding="utf-8").read()
                    if rel in rag_indexed_paths:
                        # knowledge-base doc: wél via RAG, géén koppen in index
                        entry = {"path": rel, "type": "text",
                                 "rag_indexed": True,
                                 "note": "RAG-knowledge base; hoofdstukken NIET in resource index"}
                    else:
                        entry = {"path": rel, "type": "text",
                                 "headings": extract_headings(src),
                                 "rag_indexed": False}
                elif ext in DATA_EXT:
                    entry = {"path": rel, "type": "data/config",
                             "note": "top-level sleutels via JSON/YAML-parser"}
                else:
                    entry = {"path": rel, "type": "overig"}
                files.append(entry)
            except (OSError, SyntaxError):
                # bestand niet leesbaar / geen geldige Python AST -> overslaan
                continue
    return {"files": files}


# ---------------------------------------------------------------------------
# Routing decision: Resource Index vs. RAG (the point the user emphasized)
# ---------------------------------------------------------------------------

def route(query: str, manifest: dict, rag_indexed_paths: set[str]) -> str:
    """
    Decideer of een vraag structureel (Resource Index) of semantisch (RAG) is.

    Heuristiek (zie §21 voor lichte classificatie zonder LLM):
      - "bestaat / waar ligt / welke functie / welke sectie" -> Resource Index
      - "wat staat erin / hoe werkt X / leg uit"             -> RAG
    """
    q = query.lower()
    structural_kw = ("bestaat", "waar ligt", "welke functie",
                     "welke sectie", "bestand", "welke klasse")
    semantic_kw = ("wat staat erin", "hoe werkt", "leg uit",
                   "uitleg over", "waarom")
    if any(k in q for k in structural_kw):
        return "Resource Index"
    if any(k in q for k in semantic_kw):
        return "RAG (§8)"
    # default: eerst Resource Index (goedkoop, deterministisch)
    return "Resource Index (default)"


def find_symbol_or_section(query: str, manifest: dict) -> list[str]:
    """Concrete lookup in de Resource Index (lexicaal, geen embeddings)."""
    q = query.lower()
    hits = []
    for f in manifest["files"]:
        if f["type"] == "code" and "symbols" in f:
            for sym in (f["symbols"]["functions"]
                        + f["symbols"]["classes"]
                        + f["symbols"]["globals"]):
                if q in sym.lower():
                    hits.append(f"{f['path']}::{sym}")
        elif f["type"] == "text" and "headings" in f:
            for h in f["headings"]:
                if q in h.lower():
                    hits.append(f"{f['path']}  [sectie] {h}")
    return hits


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Voorbeeld: stel dat knowledge/handbook.md deel is van de RAG-knowledge base
    rag_indexed_paths = {os.path.join("knowledge", "handbook.md")}

    manifest = index_repo(root, rag_indexed_paths)

    print(f"# Resource Index gebouwd voor: {root}")
    print(f"# Aantal bestanden in index: {len(manifest['files'])}")
    by_type = {}
    for f in manifest["files"]:
        by_type[f["type"]] = by_type.get(f["type"], 0) + 1
    print(f"# Types: {by_type}\n")

    print("## Voorbeeld-entries (eerste 3):")
    print(json.dumps(manifest["files"][:3], indent=2, ensure_ascii=False))

    print("\n## Routerings-beslissing (Resource Index vs RAG):")
    for vraag in [
        "Welke functie heet run_agent?",
        "Wat staat erin over tokenisatie?",
        "Waar ligt het hoofdstuk over MCP?",
        "Bestaat er een bestand over memory?",
    ]:
        print(f"  - {vraag!r:45} -> {route(vraag, manifest, rag_indexed_paths)}")

    print("\n## Lexicale lookup in de Resource Index (zonder embeddings):")
    for zoek in ["agent", "token", "mcp"]:
        hits = find_symbol_or_section(zoek, manifest)
        print(f"  zoekterm {zoek!r}: {len(hits)} hit(s)")
        for h in hits[:3]:
            print(f"      {h}")


if __name__ == "__main__":
    main()
