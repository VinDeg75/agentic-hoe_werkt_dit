# project_state.md — Statusoverzicht Agentic AI documentatie

Dit bestand houdt de voortgang bij van **alle elementen** uit `AGENTS.md`. Het
is bedoeld als levend dashboard: na elke wijziging (nieuw hoofdstuk, nieuw
`docs/`-bestand, nieuwe `examples/`-demo) wordt dit bestand bijgewerkt.

**Conventies**

- Statuswaarden: `✅ voltooid` · `🟡 in uitvoering` · `⚪ niet gestart` ·
  `🔄 herziening`.
- `AGENTS.md` = het fundamenteel overzicht; `docs/` = verdiepende uitwerking
  per hoofdstuk; `examples/` = kleine code-/diagramdemonstraties.
- Een hoofdstuk kan "voltooid" zijn in `AGENTS.md` maar nog "niet gestart" in
  `docs/`.

---

## 1. Algemene projectstatus

| Onderdeel | Status | Opmerking |
|-----------|--------|-----------|
| `AGENTS.md` aangemaakt | ✅ voltooid | 21 hoofdstukken (§1–§21), zie onder |
| Onderscheid algemeen LM / chat / agentic | ✅ voltooid | consequent doorgevoerd in §3 / §4 / §5 |
| `docs/` structuur voorgesteld | ✅ voltooid | 15 bestanden gedefinieerd in §2 |
| `docs/` bestanden uitgewerkt | ✅ voltooid | alle 17 aanwezig (01–15, 16, 17) — zie onder |
| `examples/` demo's | ✅ voltooid | 15 demo's, geschreven; 8 ervan eerder runtime-getest, de rest stdlib (zie §5) |
| Cheat sheet (eindproduct) | ✅ voltooid | `CHEAT_SHEET.md` aan project-root (§17 stap 7) |

**Totale voortgang:** `AGENTS.md` = 100% · `docs/` = 100% (17/17) · `examples/` = 100% (15/15) · Cheat sheet = 100%.

---

## 2. Status per hoofdstuk uit AGENTS.md

De kolom "AGENTS.md" geeft aan of het hoofdstuk (incl. subsecties) afdoende is
uitgewerkt in het fundament. De kolom "docs-bestand" koppelt het aan de
voorgestelde verdieping in `docs/`.

| § | Hoofdstuk | AGENTS.md | docs-bestand | examples | Prioriteit verdieping |
|---|-----------|-----------|--------------|----------|------------------------|
| §1 | Doel van het project | ✅ voltooid | n.v.t. (meta) | n.v.t. | — |
| §2 | Projectstructuur (voorstel) | ✅ voltooid | n.v.t. (meta) | n.v.t. | — |
| §3 | Language Model: algemeen | ✅ voltooid | ✅ `01-language-model.md` | ✅ tokenizer-demo | **hoog** |
| §4 | LLM in chatmodus | ✅ voltooid | ✅ `02-llm-chatmode.md` | ✅ chat-pipeline-demo | hoog |
| §5 | Agentic loop | ✅ voltooid | ✅ `03-agentic-loop.md` | ✅ agentic-loop-demo | hoog |
| §6 | Tools / Function calling | ✅ voltooid | ✅ `04-tools.md` | ✅ tool-call-demo | middel |
| §7 | MCP — Model Context Protocol | ✅ voltooid | ✅ `05-mcp.md` | ✅ mcp-server-demo | middel |
| §8 | RAG | ✅ voltooid | ✅ `06-rag.md` | ✅ rag-demo | middel |
| §9 | Compression | ✅ voltooid | ✅ `07-compression.md` | ✅ compression-demo | laag |
| §10 | Memory | ✅ voltooid | ✅ `08-memory.md` | ✅ memory-demo | laag |
| §11 | Planning | ✅ voltooid | ✅ `09-planning.md` | ✅ planning-demo | laag |
| §12 | Skills | ✅ voltooid | ✅ `10-skills.md` | — (niet nodig) | laag |
| §13 | Zelf model hosten (llama.cpp e.a.) | ✅ voltooid | ✅ `11-self-hosting.md` | ✅ quantization-demo | laag |
| §14 | Anonymizing proxy (PII) | ✅ voltooid | ✅ `12-data-privacy.md` | ✅ pii-masking-demo | laag |
| §15 | Token economics | ✅ voltooid | ✅ `13-token-economics.md` | ✅ token-economics-demo | laag |
| §16 | Aanvullingen (evaluatie/guardrails) | ✅ voltooid | ✅ `14-evaluation.md` | — (niet nodig) | middel |
| §17 | Werkwijze / volgende stappen | ✅ voltooid | n.v.t. (meta) | n.v.t. | — |
| §18 | Conventies | ✅ voltooid | n.v.t. (meta) | n.v.t. | — |
| §19 | Projectbestandsindex (Resource Index) | ✅ voltooid | ✅ `15-project-file-index.md` | ✅ resource-index-demo | laag |
| §20 | Agentic & model types (generiek vs coding) | ✅ voltooid | ✅ `16-agentic-model-types.md` | ✅ agentic-model-types-demo | laag |
| §21 | Lite-classificatie & routing (optimalisaties) | ✅ voltooid | ✅ `17-lite-classification.md` | ✅ lite-classification-demo | laag |

---

## 3. Status per subsectie (detailniveau)

Per hoofdstuk worden de subsecties bijgehouden zodat zichtbaar is wat later nog
verdiept moet worden in `docs/`.

### §3 Language Model (algemeen)
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 3.1 | Tokenisatie (BPE / WordPiece / SentencePiece) | ✅ |
| 3.2 | Embeddings + positionele encoding (RoPE / ALiBi) | ✅ |
| 3.3 | Decoder-only Transformer (attention / FFN / LayerNorm) | ✅ |
| 3.4 | Architectuur-typen (Dense / MoE / SSM / Hybrid) | ✅ |

### §4 LLM in chatmodus
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 4.1 | Server ontvangt plaintext (validatie, prompt-assemblage, chat template) | ✅ |
| 4.2 | Forward pass → logits | ✅ |
| 4.3 | Sampling (greedy/temp/top-k/top-p/beam) + autoregressief + KV-cache | ✅ |
| 4.4 | Decoding / detokenisatie / streaming | ✅ |

### §5 Agentic loop
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 5.1 | Component-tabel (wanneer welk component) | ✅ |
| 5.2 | Concreet voorbeeld (Q2-rapport) | ✅ |

### §6 Tools / Function calling
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 6.1 | Tool-call cyclus (sequence-diagram) | ✅ |
| 6.2 | Tool-schema (JSON-Schema definitie) | ✅ |
| 6.3 | Native function calling vs. ReAct | ✅ |
| 6.4 | Geavanceerde patronen (parallel / chaining / sub-agent) | ✅ |
| 6.5 | Veiligheid & betrouwbaarheid | ✅ |

### §7 MCP
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 7.1 | Architectuur (Host / Client / Server, JSON-RPC) | ✅ |
| 7.2 | Drie primitieven (Tools / Resources / Prompts) | ✅ |
| 7.3 | Waarom het ertoe doet (interoperabiliteit) | ✅ |
| 7.4 | Wanneer te gebruiken | ✅ |

### §8 RAG
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 8.1 | Waarom RAG? | ✅ |
| 8.2 | Volledige pijplijn (index → retrieval) | ✅ |
| 8.3 | Chunking-strategieën | ✅ |
| 8.4 | Embedding-modellen | ✅ |
| 8.5 | Vector-DB & similariteit | ✅ |
| 8.6 | Re-ranking & hybrid search | ✅ |
| 8.7 | Evaluatie van RAG | ✅ |
| 8.8 | Wanneer (niet) te gebruiken | ✅ |

### §9 Compression
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 9.1 | Waarom comprimeren? | ✅ |
| 9.2 | Vormen (samenvatting / prompt / KV-cache / window) | ✅ |
| 9.3 | Trade-offs | ✅ |
| 9.4 | Wanneer toepassen | ✅ |

### §10 Memory
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 10.1 | Kortetermijn (working memory) | ✅ |
| 10.2 | Langetermijn (vector-DB / SQL / file) | ✅ |
| 10.3 | Geheugentypen (episodisch/semantisch/procedureel) | ✅ |
| 10.4 | Write- & read-patronen | ✅ |
| 10.5 | Consolidatie & vergeten | ✅ |

### §11 Planning
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 11.1 | Waarom plannen? | ✅ |
| 11.2 | Strategieën (ReAct / plan-execute / reflexion / ToT) | ✅ |
| 11.3 | Planner vs. Executor | ✅ |
| 11.4 | Evaluatie & error recovery | ✅ |
| 11.5 | Wanneer welke strategie | ✅ |

### §12 Skills
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 12.1 | Wat is een skill? | ✅ |
| 12.2 | Skills vs. Tools vs. MCP | ✅ |
| 12.3 | Lifecycle (discovery / loading / execution) | ✅ |
| 12.4 | Wanneer skills gebruiken | ✅ |

### §13 Zelf model hosten
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 13.1 | Waarom zelf hosten? | ✅ |
| 13.2 | Inference-engines (llama.cpp / vLLM / Ollama e.a.) | ✅ |
| 13.3 | Serving & OpenAI-compatibiliteit | ✅ |
| 13.4 | Quantisatie (GGUF / GPTQ / AWQ) | ✅ |
| 13.5 | Cloud vs. self-host | ✅ |

### §14 Anonymizing proxy (PII)
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 14.1 | Het probleem (data verlaat organisatie) | ✅ |
| 14.2 | Werkwijze (detectie / maskering / un-mask) | ✅ |
| 14.3 | Afwegingen (reversibel / kwaliteit / logging) | ✅ |
| 14.4 | Plaats in de architectuur | ✅ |

### §15 Token economics
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 15.1 | Waaruit bestaan de kosten? | ✅ |
| 15.2 | Waarom het bij agents explodeert | ✅ |
| 15.3 | Hefboomacties (compressie / caching / kleine modellen) | ✅ |
| 15.4 | Cloud vs. self-host (break-even) | ✅ |

### §16 Aanvullingen
| Onderwerp | Status AGENTS.md |
|-----------|------------------|
| Evaluatie & guardrails | ✅ |
| Kosten & latency | ✅ |
| Observability | ✅ |
| Determinisme vs. creativiteit | ✅ |
| Privacy & compliance | ✅ |

### §20 Agentic & model types
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 20.1 | Generieke vs coding agentic | ✅ |
| 20.2 | Codex-type modellen (wat verwachten) | ✅ |
| 20.3 | Hoe model instructies naar loop seint | ✅ |

### §21 Lite-classificatie & routing
| Sub | Onderwerp | Status AGENTS.md |
|-----|-----------|------------------|
| 21.1 | Probleem (categorie zonder geheugenmodel) | ✅ |
| 21.2 | Shortcuts & lokale "modellen" | ✅ |
| 21.3 | Voorbeeld (TF-IDF routing, lokaal) | ✅ |
| 21.4 | Trade-offs | ✅ |

---

## 4. Status per `docs/` bestand (voorgesteld in §2)

| Bestand | Koppel aan | Aangemaakt? | Kluis inhoud |
|---------|-----------|-------------|--------------|
| `docs/01-language-model.md` | §3 | ✅ ja | tokenisatie, embeddings, Transformer, MoE/SSM |
| `docs/02-llm-chatmode.md` | §4 | ✅ ja | server-pipeline, sampling, KV-cache, streaming |
| `docs/03-agentic-loop.md` | §5 | ✅ ja | perceive→reason→act→observe→respond, router + Q2-voorbeeld |
| `docs/04-tools.md` | §6 | ✅ ja | tool-call cyclus, schema, natief vs. ReAct, patronen, veiligheid |
| `docs/05-mcp.md` | §7 | ✅ ja | JSON-RPC 2.0 stdio-server, primitieven, host-aggregatie, vertrouwensgate |
| `docs/06-rag.md` | §8 | ✅ ja | chunking, embedder, vector-DB, re-ranking |
| `docs/07-compression.md` | §9 | ✅ ja | samenvatting, prompt-compressie, KV-cache |
| `docs/08-memory.md` | §10 | ✅ ja | STM/LTM, episodisch/semantisch, consolidatie |
| `docs/09-planning.md` | §11 | ✅ ja | ReAct, plan-execute, reflexion |
| `docs/10-skills.md` | §12 | ✅ ja | Skill-klasse, vs tools/MCP, lifecycle, selectie/best-practice |
| `docs/11-self-hosting.md` | §13 | ✅ ja | llama.cpp, quantisatie, serveren |
| `docs/12-data-privacy.md` | §14 | ✅ ja | PII-lek demo, AnonymizingProxy (detect/mask/unmask), afwegingen, wrapper |
| `docs/13-token-economics.md` | §15 | ✅ ja | TokenBill, agent-loop-explosie, hefbomen, break-even |
| `docs/14-evaluation.md` | §16 | ✅ ja | guardrail/eval, kosten/latency, trace, temperatuur, privacy-gate |
| `docs/15-project-file-index.md` | §19 | ✅ ja | metadataschema, manifest, repo-indexer, router, MCP/Memory-relatie, trade-offs |
| `docs/16-agentic-model-types.md` | §20 | ✅ ja | generieke vs coding agent, codex-type modellen, model→loop signaling |
| `docs/17-lite-classification.md` | §21 | ✅ ja | keyword/TF-IDF/embedder routing, zero-shot/fastText, gecombineerde router |

---

## 5. Status per `examples/` demonstratie

| Demo | Gekoppeld aan | Aangemaakt? | Soort / status |
|------|--------------|-------------|-------|
| `examples/tokenizer-demo.py` | §3.1 | ✅ ja | BPE vanaf nul + tiktoken (runtime-getest) |
| `examples/chat-pipeline-demo.py` | §4 | ✅ ja | prompt-assemblage + ChatML-template (runtime-getest) |
| `examples/agentic-loop-demo.py` | §5 | ✅ ja | perceive→reason→act→observe→respond (runtime-getest) |
| `examples/tool-call-demo.py` | §6 | ✅ ja | function-calling cyclus + JSON-schema (runtime-getest) |
| `examples/mcp-server-demo.py` | §7 | ✅ ja | JSON-RPC 2.0 stdio-server, tools-primitief (stdio-getest) |
| `examples/rag-demo.py` | §8 | ✅ ja | chunk + hash-embed + cosine top-k (runtime-getest, numpy) |
| `examples/quantization-demo.py` | §13.4 | ✅ ja | VRAM per quant-niveau (runtime-getest) |
| `examples/pii-masking-demo.py` | §14 | ✅ ja | detect → mask → unmask (runtime-getest) |
| `examples/agentic-model-types-demo.py` | §20 | ✅ ja | type-router, codex-output-stijl, loop-parsing signaling (stdlib) |
| `examples/lite-classification-demo.py` | §21 | ✅ ja | keyword + hash-embed router, gecombineerd (stdlib; sklearn optioneel) |
| `examples/resource-index-demo.py` | §19 | ✅ ja | ast-symbolen + koppen + manifest + Resource-Index-vs-RAG router (stdlib) |
| `examples/compression-demo.py` | §9 | ✅ ja | history-summary + sliding-window + budget-gestuurde ContextManager (stdlib) |
| `examples/memory-demo.py` | §10 | ✅ ja | STM/LTM + episodisch/semantisch/procedureel + consolidatie/TTL + JSONL-log (stdlib) |
| `examples/planning-demo.py` | §11 | ✅ ja | planner + executor + error recovery (send_email retry na Go) (stdlib) |
| `examples/token-economics-demo.py` | §15 | ✅ ja | naive/caching/compress simulator + kosten (stdlib) |

> Extra demo's toegevoegd voor §13 (quantisatie) en §14 (PII-masking) omdat die
> het meest concreet te demonstreren zijn. §9–§12, §15, §16 zijn verwerkt in de
> in de `docs/`-uitwerking; een aparte code-demo was daar niet strikt nodig, maar
> **§9 (Compression), §10 (Memory) en §19 (Resource Index) hebben nu wél een standalone demo**
> (ast-symbolen + koppen + manifest + Resource-Index-vs-RAG router), omdat de
> gebruiker die ontsluiting expliciet benadrukte. §20 en §21 hebben eveneens een
> standalone `examples/*.py` (router/signaling-loop resp. keyword+embed-router),
> zodat die hoofdstukken concreet runbaar zijn.

---

## 6. Open punten / TODO

- [x] Starten met `docs/01-language-model.md` (voltooid als template).
- [x] `docs/02-llm-chatmode.md` en `docs/06-rag.md` t/m `docs/11-self-hosting.md`
      (06/08/11 aangemaakt in eerdere run; 07/09 in parallelle run) nu
      aanwezig op disk; voortzetting **sequentieel** per doc vanaf `docs/03`.
- [x] `docs/03-agentic-loop.md` (§5) aangemaakt.
- [x] `docs/15-project-file-index.md` (§19) — **LAATSTE docs-bestand**, voltooid.
      `docs/10-skills.md` (§12), `docs/12-data-privacy.md` (§14),
      `docs/13-token-economics.md` (§15), `docs/14-evaluation.md` (§16),
      `docs/15-project-file-index.md` (§19).
- [x] `06/07/08/09/11` (reeds op disk) gecontroleerd op template-conformiteit:
      alle vijf volgen de structuur Functioneel / Technisch (Python-voorbeeld
      per subsectie) / Samenvatting + mermaid-regels — **geen normalisatie nodig**.
- [x] Beslist welke hoofdstukken een `examples/`-demo krijgen: §3, §4, §5, §6,
      §7, §8, §9, §10, §13, §14, §19, §20, §21 (13 demo's; zie §5).
- [x] Cheat sheet (`CHEAT_SHEET.md`, §17 stap 7) aangemaakt aan project-root.
- [ ] Bij elke nummerwijziging in `AGENTS.md` de interne koppelingen hier
      (en de ankers in §2/§5.1) synchroniseren — renummering is fragiel.
- [x] Conventie: `project_state.md` wordt na elke fase bijgewerkt.
- [x] Ideeën uit `ideas.md` verwerkt: chunkers (§8.3), tokenisatie init + merge
      (§3.1), RAG-varianten & recordstructuur (§8.9), agentic & model types (§20),
      lite-classificatie & routing (§21).
- [x] `docs/06-rag.md` aangevuld (was incompleet: §8.5–§8.8 + Samenvatting
      ontbraken) én uitgebreid met chunker-implementaties en §8.9.
- [x] `docs/01-language-model.md` §3.1 uitgebreid met BPE init + merge-walkthrough.
- [x] `docs/16-agentic-model-types.md` (§20) en `docs/17-lite-classification.md`
      (§21) aangemaakt; `CHEAT_SHEET.md` en §2-structuur in `AGENTS.md` bijgewerkt.

- [x] Extra `examples/`-demo's voor de laatste proza-hoofdstukken:
      - [x] §11 Planning — plan-and-execute loop met sub-stappen + error recovery
      - [x] §15 Token economics — TokenBill-simulator van de agent-loop-explosie
- [ ] Meta-uitbreidingen (optioneel, nog niet gestart):
      - [ ] Kwaliteitspas: interne kruisverwijzingen/ankers in `AGENTS.md` en
            `project_state.md` controleren op consistentie
      - [ ] `README.md` als instappunt/overzicht aan project-root toevoegen
      - [ ] `examples/run_all.py` die alle demo's achter elkaar draait
      - [ ] Git-commit(s) per logische groep (conform commit-stijl in persoonlijke
            `AGENTS.md`)

---

*Laatst bijgewerkt: alle proza-hoofdstukken (§9, §10, §11, §15, §19) hebben nu een
standalone demo (nu 15/15). **Volledig project 100%**: `AGENTS.md` (§1–§21) ·
`docs/` 17/17 · `examples/` 15/15 · cheat sheet. Zie `CHEAT_SHEET.md` voor het overzicht.*
*Open voorstellen (nog niet gestart) staan in §6: meta-uitbreidingen
(kwaliteitspas, README, run_all, git-commits).*
