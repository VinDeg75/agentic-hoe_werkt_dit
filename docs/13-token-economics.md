# 15. Token economics (verdieping)

> Dit document verdiept [§15 van `AGENTS.md`](../AGENTS.md). LLM-gebruik wordt
> verrekend **per token**. Token economics gaat over het *begrijpen en beheersen*
> van die kosten — vooral relevant zodra je **agents** (§5) draait met veel
> context, RAG (§8) en tool-iteraties (§6). Het sluit direct aan op compressie
> (§9), caching (§4.3) en self-hosting (§13).

Het doel van dit hoofdstuk is **elk detail concreet aan te tonen** met
minimalistische, leesbare Python-voorbeelden. De code is pedagogisch: ze toont
het *principe* (kostenopbouw, agent-loop-explosie, hefbomen, break-even) correct,
maar met vereenvoudigde prijzen.

---

## Functioneel: waarom praten we over token economics?

Een LLM rekent niet per vraag maar **per token** — voor input én output, plus
indirecte kosten van context (KV-cache, §9). Bij een eenmalige chatvraag (§4) is
dat overzichtelijk. Maar een **agentic loop** (§5) stuurt *elke stap* de volledige
context opnieuw naar het model: geschiedenis + RAG-chunks (§8) + tool-resultaten
(§6) stapelen zich op. Meerdere iteraties × grote context = **veel tokens per
taak**, en dus een onverwachte rekening.

De kernvraag: hoe houd je de token-rekening beheersbaar zonder de agent dommer
te maken? Daarvoor dienen de **hefbomen** (compressie, caching, kleinere modellen,
self-hosting, limieten) uit 15.3.

```mermaid
flowchart TD
    A[Agent-loop stap] --> B[Volledige context opnieuw naar model]
    B --> C[Input-tokens: sys + hist + RAG + tools + tussenstappen]
    C --> D[Output-tokens: antwoord / tool_call]
    D --> E[Kosten = f(input, output, cache)]
    E --> F[Hefbomen: compressie / caching / kleine modellen / self-host]
    F --> A
```

---

## Technisch

### 15.1 Waaruit bestaan de kosten?

**Functioneel.** De kosten hebben vier componenten:
- **Input-tokens** — system prompt + geschiedenis + RAG-chunks + tool-schemas +
  tussenstappen (groeit snel bij agents).
- **Output-tokens** — het gegenereerde antwoord.
- **Prijsmodel** — vaak verschillend tarief input vs. output; *cached input*
  (prefix-cache) is goedkoper (zie §4.3, §9.2).
- **KV-cache / context** — indirecte kosten van langdurige context (§9.1).

**Technisch (een `TokenBill` die de drie tarieven optelt).**

```python
# 15.1 — Kostenopbouw: input, output én cached input hebben eigen tarief
class TokenBill:
    def __init__(self, price_in=0.00001, price_out=0.00003, price_cached=0.0000025):
        # prijzen in $ per token (vereenvoudigde OpenAI-achtige tarieven)
        self.p = {"in": price_in, "out": price_out, "cached": price_cached}
        self.n = {"in": 0, "out": 0, "cached": 0}

    def add(self, kind: str, tokens: int):
        self.n[kind] += tokens

    def total(self) -> float:
        return sum(self.n[k] * self.p[k] for k in self.n)

bill = TokenBill()
bill.add("in", 1200)        # nieuwe input
bill.add("cached", 8000)    # herbruikte system-prefix (goedkoper)
bill.add("out", 350)
print(f"totale kosten: ${bill.total():.4f}  (cached helpt!)")
```

> **Cached input is spotgoedkoop.** Een stabiele system-prompt die je bij elke
> call hergebruikt (prefix-cache, §4.3) drukt de prijs fors — zie ook 15.3.

---

### 15.2 Waarom het bij agents explodeert

**Functioneel.** Een agent-loop stuurt de *volledige context* bij elke stap
opnieuw naar het model. Elke tool-call en elk RAG-resultaat voegt tokens toe aan
de geschiedenis, die in de volgende iteratie weer *helemaal* meegestuurd wordt.

**Technisch (simuleer een agent-loop en tel de cumulatieve tokens).**

```python
# 15.2 — Agent-loop: context groeit per stap -> tokens exploderen
def simulate_agent_loop(n_steps: int, base_ctx: int, per_step_growth: int):
    total_input = 0
    ctx = base_ctx
    for step in range(n_steps):
        total_input += ctx           # HELE context gaat elke stap mee
        ctx += per_step_growth       # tool/RAG-resultaat erbij
    return total_input

cheap = simulate_agent_loop(5, 500, 200)
expensive = simulate_agent_loop(20, 2000, 1500)
print("5 stappen, klein :", cheap, "input-tokens")
print("20 stappen, groot:", expensive, "input-tokens  <- explosie")
```

> **Inzicht.** Het aantal input-tokens is niet lineair maar *cumulatief*: elke
> stap telt de volledige (gegroeide) geschiedenis mee. Dat is precies waarom
> compressie (§9) en tool-result-limieten (§6.5/15.3) cruciaal zijn.

---

### 15.3 Heftactieken (cost levers)

**Functioneel.** Zes hefbomen verlagen de rekening:

| Hefboom | Werking | Hoofdstuk |
|---------|---------|-----------|
| **Compressie** | kortere geschiedenis / prompt-compressie | §9 |
| **Caching** | herbruik system-prefix via prompt-cache | §4.3, §9.2 |
| **Kleinere modellen** | goedkoper model voor routing/extractie | §11 |
| **Self-hosting** | vaste infra i.p.v. per-token | §13 |
| **Tool-result-limiet** | truncateer/vaat tool-output samen | §6, §9 |
| **Token-budget** | `max_tokens` / context-plafond | §4, §9 |

**Technisch (vergelijk kosten mét vs. zónder hefbomen).**

```python
# 15.3 — Hefbomen: compressie + caching + tool-limiet verlagen de rekening
def cost_without_levers(steps):
    return simulate_agent_loop(steps, 2000, 1500)   # groot, ongecomprimeerd

def cost_with_levers(steps):
    # compressie (§9): geschiedenis elke stap gehalveerd
    # tool-limiet (§6/§9): groei per stap kleiner
    return simulate_agent_loop(steps, 1000, 400) + steps * 8000  # + cached prefix

b = TokenBill()
raw = cost_without_levers(20)
lev = cost_with_levers(20)
b.add("in", raw)
saving = (raw - (lev - 20 * 8000)) / raw   # cached deel niet meegeteld
print(f"raw input-tokens : {raw}")
print(f"met hefbomen     : {lev} (incl. goedkope cached prefix)")
print(f"compressie bespaart ~{saving:.0%} op de groeiende geschiedenis")
```

> **Caching als hefboom.** De `8000` cached tokens per stap kosten een
> fractie van gewone input (zie 15.1) — een stabiele system-prompt + tool-schemas
> zijn dus dubbel winstgevend.

---

### 15.4 Cloud vs. self-host (break-even)

**Functioneel.**
- **Cloud** = variabele kosten per token, geen investering, direct beschikbaar
  — maar prijzig bij hoog volume.
- **Self-host** (§13) = capex (GPU's) + onderhoud, maar marginale kosten per
  token laag → **break-even** bij voldoende volume.
- **Kostenobservability** — track tokens per stap (§16) om dure patronen te
  spotten.

**Technisch (bereken het break-even volume).**

```python
# 15.4 — Break-even: wanneer is self-host goedkoper dan cloud?
def break_even_tokens(capex_per_month, maintenance, cloud_price_per_token):
    """Volume (tokens/maand) waarbij vaste kosten == cloud-kosten."""
    fixed = capex_per_month + maintenance
    return fixed / cloud_price_per_token

be = break_even_tokens(
    capex_per_month=2000,   # GPU-serverhuur e.d.
    maintenance=300,
    cloud_price_per_token=0.00001,   # $ per token (input+output gemiddeld)
)
print(f"break-even bij ~{be:,.0f} tokens/maand "
      f"(~{be/1_000_000:.1f} M tokens/maand)")
# boven dit volume wint self-hosting (§13) het op kosten
```

> **Observability first.** Vóór je de break-even beslist, meet je werkelijke
> volumes per stap (§16). Zonder die data is de keuze cloud vs. self-host (§13)
> een gok.

---

## Samenvatting (key takeaways)

- LLM-kosten zijn **per token**: input, output én *cached input* hebben eigen
  tarieven — caching (§4.3) is de goedkoopste hefboom.
- Een **agent-loop** stuurt *elke stap* de volledige (gegroeide) context mee →
  cumulatieve, niet-lineaire token-groei.
- Zes **hefbomen**: compressie (§9), caching, kleinere modellen (§11),
  self-hosting (§13), tool-result-limiet (§6) en token-budget (§4).
- **Break-even** tussen cloud en self-host hangt af van volume; meet eerst je
  werkelijke token-verbruik per stap (observability, §16).

Het volgende document (`14-evaluation.md`) behandelt **evaluatie, guardrails en
observability** — het gereedschap om te meten of je agent én je
kostenbeheersing (dit hoofdstuk) écht werken.
