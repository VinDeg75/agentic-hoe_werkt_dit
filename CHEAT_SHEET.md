# Cheat Sheet — Hoe je een Agentic AI-tool bouwt

**Snelzoek-blad** voor dit hele project. Voor de diepte per onderwerp: zie
`AGENTS.md` (fundament, §1–§21) en de uitwerking in `docs/NN-*.md`. Voor
werkende code: `examples/*.py`.

> Kerno­nderscheid dat alles stuurt: **algemene LM** (§3) ≠ **chat** (§4) ≠
> **agentic** (§5). De basis (tokenisatie → embedding → Transformer → logits)
> is universeel; chat en agentic zijn *lagen erbovenop*.

---

## 0. De drie lagen op één rij

| Laag | Vraag | Kern |
|------|-------|------|
| **LM (§3)** | Hoe voorspelt een model het volgende token? | tokeniseer → embed + positioneer → N Transformer-lagen → logits → sample |
| **Chat (§4)** | Wat doet de *server* met een plaintext-vraag? | validatie → prompt-assemblage + chat-template → forward pass → sampling → decode/stream |
| **Agentic (§5)** | Hoe *onderneemt* een model acties? | perceive → reason → act (tool) → observe → repeat → respond |

---

## 1. Language Model (algemeen) — §3 → `docs/01-language-model.md`

- **Tokenisatie (3.1):** BPE / WordPiece / SentencePiece zet tekst om in IDs.
  Token ≠ karakter → bepaalt de *kosteenheid* (§15).
- **Embeddings + positie (3.2):** lookup-matrix + **RoPE/ALiBi** geven volgorde
  (Transformer heeft geen tijdsbesef).
- **Decoder-only Transformer (3.3):** per laag: causal self-attention → FFN →
  LayerNorm + residual. **KV-cache** onthoudt eerdere key/value voor snelheid.
- **Architectuur-typen (3.4):** Dense (alle params actief) vs **MoE** (router
  kiest top-k experts → *actieve* params ≪ *totale* params) vs SSM/Mamba vs
  Hybrid.

## 2. LLM in chatmodus — §4 → `docs/02-llm-chatmode.md`

1. **Server krijgt plaintext** → validatie + prompt-assemblage (system + history
   + user) + chat-template (`<|im_start|>` …).
2. **Forward pass** → logits (score per vocab-token).
3. **Sampling (4.3):** greedy / temperature / top-k / top-p / beam.
   **Autoregressief** + KV-cache tot stop-token.
4. **Decoding** → tekst, eventueel **streaming**.

## 3. Agentic loop — §5 → `docs/03-agentic-loop.md`

- De **dirigent**: per iteratie beslist hij welk component (tool, RAG,
  compression, memory) nodig is en voegt het resultaat terug in de context.
- Wanneer welk component? Zie de tabel in §5.1 / `docs/03-agentic-loop.md`.

## 4. Componenten (verdieping per §)

| § | Component | Wanneer ingezet | Docs | Demo |
|---|-----------|-----------------|------|------|
| §6 | **Tools / Function calling** | Model moet *handelen* (API, rekenen, code, DB) | `04-tools.md` | `tool-call-demo.py` |
| §7 | **MCP** | Veel externe systemen *gestandaardiseerd* ontsluiten (tools/resources/prompts) | `05-mcp.md` | `mcp-server-demo.py` |
| §8 | **RAG** | Actuele / privé / gespecialiseerde *kennis* (semantisch) | `06-rag.md` | `rag-demo.py` |
| §9 | **Compression** | Context te groot / te duur / te traag wordt | `07-compression.md` | — |
| §10 | **Memory** | Continuïteit *over sessies* (feiten onthouden) | `08-memory.md` | — |
| §11 | **Planning** | Complexe, multi-stap of foutgevoelige taken | `09-planning.md` | — |
| §12 | **Skills** | Terugkerende SOP's / domeinexpertise als pakket | `10-skills.md` | — |
| §13 | **Self-hosting** | Privacy / volume / offline → eigen inference-server | `11-self-hosting.md` | `quantization-demo.py` |
| §14 | **Anonymizing proxy** | PII maskeren vóór een (cloud-)API | `12-data-privacy.md` | `pii-masking-demo.py` |
| §15 | **Token economics** | Kosten beheersen (per-token) | `13-token-economics.md` | — |
| §16 | **Evaluatie / guardrails** | Veilig & meetbaar laten werken | `14-evaluation.md` | — |
| §19 | **Resource Index** | *Structuur* ontsluiten (welk bestand/functie bestaat) | `15-project-file-index.md` | — |
| §20 | **Agentic & model types** | Coding vs generieke agent; codex-model; hoe model→loop seint | `16-agentic-model-types.md` | — |
| §21 | **Lite-classificatie & routing** | Categorie bepalen *zonder* LLM (lokale libs/modellen) | `17-lite-classification.md` | — |

## 5. RAG vs. Resource Index (de belangrijkste keuze) — §8 vs §19

- **RAG (§8):** "Wat *staat erin* over onderwerp X?" → semantisch, embeddings,
  vector-DB. Gebruik voor inhoud/privé-kennis.
- **Resource Index (§19):** "Welk bestand / welke functie / sectie *bestaat*?"
  → structureel, deterministisch, goedkoop (tree-sitter / heading-extractie).
- **Complementair:** eerst de Resource Index (waar ligt wat), dan RAG of direct
  lezen (wat is de inhoud).

## 6. Wanneer gebruik je wat? (korte beslissers)

- Chat-antwoord volstaat? → enkel §3+§4.
- Model moet iets *doen*? → **Tools (§6)**; veel bronnen → **MCP (§7)**.
- Model mist *kennis*? → **RAG (§8)**; onthouden over sessies → **Memory (§10)**.
- Context explodeert? → **Compression (§9)** + kosten (§15).
- Taak is complex/multi-stap? → **Planning (§11)**; terugkerende werkwijze →
  **Skills (§12)**.
- Data mag niet weg? → **Self-host (§13)** of **Anonymizing proxy (§14)**.
- Agent is code-gericht? → **Coding agentic (§20)** + codex-type model; taak
  classificeren vóór de LLM? → **Lite-classificatie (§21)** (keyword/TF-IDF/embedder).

## 7. Kosten-hefbomen (§15)

Compressie · prompt-caching · kleinere modellen voor sub-agenten · self-host bij
volume · tool-output-limiet · token-budget. Een agent stuurt de *volledige*
context bij elke stap opnieuw → tokens stapelen snel.

## 8. Snel bekijken van de code

```bash
# elk demo is standalone; numpy en tiktoken enkel waar aangegeven
python examples/tokenizer-demo.py        # §3.1  BPE vanaf nul + tiktoken
python examples/chat-pipeline-demo.py    # §4    prompt-assemblage + template
python examples/agentic-loop-demo.py     # §5    perceive→act→observe→respond
python examples/tool-call-demo.py        # §6    function calling cyclus
python examples/mcp-server-demo.py       # §7    JSON-RPC 2.0 stdio-server
python examples/rag-demo.py              # §8    chunk + embed + retrieve
python examples/quantization-demo.py     # §13.4 VRAM per quant-niveau
python examples/pii-masking-demo.py      # §14   detect → mask → unmask
```

---

*Dit blad is het eindproduct (§17 stap 7). Voor elke claim: zie het gekoppelde
`docs/NN-*.md` en de bron in `AGENTS.md` (§-nummer, §1–§21).*
