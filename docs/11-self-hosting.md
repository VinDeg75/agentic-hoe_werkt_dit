# 13. Zelf model hosten met llama.cpp of alternatief (verdieping)

> Dit document verdiept [§13 van `AGENTS.md`](../AGENTS.md). Het beschrijft hoe je
> een Language Model **zelf draait** — lokaal op je eigen machine of op eigen
> servers — in plaats van een (cloud-)API aan te roepen. Waar §4 en §5 ervan
> uitgingen dat een externe server het model levert, nemen we hier die server zélf
> in beheer. Dit raakt direct de privacy (§14) en de kosten (§15).

Het doel van dit hoofdstuk is **elk detail concreet aan te tonen** met
minimalistische, leesbare Python-voorbeelden. De code is pedagogisch: ze toont
het *principe* correct, maar is geen productie-implementatie.

---

## Functioneel

**Self-hosting** betekent: in plaats van een prompt naar de server van een
cloud-aanbieder (OpenAI, Anthropic, Google ...) te sturen, draai je het model
zélf op hardware die je beheert. Dat kan een laptop zijn (CPU/GPU), een
werkstation, of een GPU-server in je eigen datacenter.

Het functionele verschil zit in *wie de inference-server beheert*:

- **Cloud-API** — de server, het model en de GPU's staan bij de aanbieder; jij
  stuurt alleen tekst heen en weer.
- **Self-hosted** — jij draait de *eigen inference-server*; het model
  (gewichten) staat lokaal en de prompt verlaat je infrastructuur niet.

**Waarom zou je self-hosten?**

- **Data blijft binnen** — gevoelige prompts (PII, broncode, bedrijfsdata)
  verlaten nooit je netwerk. Dit sluit direct aan op §14 (anonymizing proxy):
  als self-host niet haalbaar is, is zo'n proxy de tweede optie, maar "niets
  verstuurt" is nog altijd het sterkst.
- **Geen per-token kosten** — bij een cloud-API betaal je per token (§15). Bij
  self-hosting vervang je dat door eigen hardware (capex) en stroom; de marginale
  kost per token is laag, gunstig bij hoog of gelijkmatig volume.
- **Volledige controle** — je kiest het model, fine-tunes, permissies en kunt
  offline werken. Geen rate-limits of beleid van een derde partij.

**Wanneer self-hosten?**

- Bij **strenge privacy-eisen** (§14): regelgeving of bedrijfsbeleid verbiedt
  data-uitwisseling.
- Bij **hoog/gelijkmatig volume** (kosten, §15): de vaste hardware-investering
  verdient zich terug ten opzichte van variabele per-token facturatie.
- Bij **offline-eisen** of het gebruik van **specifieke / fine-tuned modellen**
  die niet als API beschikbaar zijn.

De pijplijn is verder identiek aan de chat-pipeline uit §4 — enkel het
*eindpunt* verandert van "cloud" naar "eigen server":

```mermaid
flowchart LR
    C[Client / Agent-loop] --> S[Eigen inference-server]
    S --> M[Model (gewichten lokaal)]
    M -->|logits / tokens| S
    S -->|streaming antwoord| C
```

> **Vereenvoudiging:** de diagram toont één server. In productie zie je vaak
> load-balancers, model-cache, en meerdere replica's — maar het principe (de
> client praat met *jouw* server, niet met die van een cloud-aanbieder) blijft
> hetzelfde.

---

## Technisch

### 13.1 Waarom zelf hosten? (technische afweging)

**Functioneel.** De drie hoofdredenen — data-binnen, geen per-token kosten,
controle — vertalen zich technisch naar een *keuze* in je architectuur. De
volgende beslissingsfunctie toont hoe die afweging concreet gemaakt wordt:

```python
# pedagogische keuze-hulp — geen productiecode
def hosting_recommendation(privacy_strict, volume_high, needs_offline):
    """Return 'self-host' or 'cloud' based on hard constraints."""
    # Privacy en offline zijn harde eisen: self-host wint
    if privacy_strict or needs_offline:
        return "self-host"
    # Hoog, gelijkmatig volume maakt capex voordeliger dan per-token OPEX
    if volume_high:
        return "self-host"
    # Anders: snelle start zonder GPU-investering -> cloud
    return "cloud"

print(hosting_recommendation(privacy_strict=True,  volume_high=False, needs_offline=False))
print(hosting_recommendation(privacy_strict=False, volume_high=True,  needs_offline=False))
```

**Technisch.** De functie boven is een *vereenvoudigd* beslisboom. In de
praktijk wegen ook latentie, beschikbare GPU's, en onderhouds-capaciteit mee
(zie 13.5). Het toont wel het kernprincipe: self-host is de default zodra een
*harde* eis (privacy, offline) of *volumetisch* voordeel speelt.

---

### 13.2 Inference-engines / runtimes

**Functioneel.** Een model-bestand (gewichten) is niet direct uitvoerbaar; je
hebt een **runtime** (inference-engine) nodig die de tensors laadt, de
forward-pass doet, en een server aanbiedt. De belangrijkste runtimes:

| Runtime | Kenmerken | Use-case |
|---------|-----------|----------|
| **llama.cpp** | GGUF-formaat, CPU+GPU, quantisatie, zeer breed model-ondersteuning | Lokaal, weinig VRAM |
| **Ollama** | Laagdrempelig, bouwt op llama.cpp, OpenAI-compatibele server | Snelle lokale dev |
| **vLLM** | Hoge throughput, continuous batching, paged-attention | GPU-server / productie |
| **TGI** (Text Generation Inference) | Geoptimaliseerd serveren, quantisatie | Productie (HuggingFace) |
| **LM Studio** | GUI + local server, OpenAI-compatibel | Lokaal experimenteren |
| **ExLlama / TensorRT-LLM** | GPU-specifiek, maximale snelheid | Datacenter-GPU's |

**Technisch (runtime-selectie op basis van constraints).** De keuze hangt af
van je hardware en doel. Dit snippet selecteert een runtime uit een klein
regelboek:

```python
# pedagogische selector — geen productiecode
RUNTIMES = {
    "llama.cpp": {"min_vram_gb": 0,  "mode": "local",      "openai_compat": True},
    "Ollama":    {"min_vram_gb": 0,  "mode": "local-dev",  "openai_compat": True},
    "vLLM":      {"min_vram_gb": 16, "mode": "gpu-prod",   "openai_compat": True},
    "TGI":       {"min_vram_gb": 16, "mode": "gpu-prod",   "openai_compat": True},
    "LM Studio": {"min_vram_gb": 0,  "mode": "local-gui",  "openai_compat": True},
    "ExLlama":   {"min_vram_gb": 24, "mode": "gpu-datacenter", "openai_compat": True},
}

def pick_runtime(available_vram_gb, want_production):
    for name, spec in RUNTIMES.items():
        if available_vram_gb < spec["min_vram_gb"]:
            continue
        if want_production and spec["mode"] not in ("gpu-prod", "gpu-datacenter"):
            continue
        return name
    return None  # geen runtime past binnen deze (gesimuleerde) constraints

print(pick_runtime(available_vram_gb=8,  want_production=False))  # -> Ollama / LM Studio / llama.cpp
print(pick_runtime(available_vram_gb=32, want_production=True))   # -> vLLM / TGI / ExLlama
```

> **Vereenvoudiging:** de echte keuze hangt ook af van model-formaat (GGUF vs.
> safetensors), besturingssysteem, en of je *batching* nodig hebt (vLLM/TGI
> scoren daar sterk op). Het snippet toont enkel het VRAM/doel-filter.

---

### 13.3 Serving & compatibiliteit (OpenAI-compatibele API)

**Functioneel.** De meeste runtimes bieden een **OpenAI-compatibele API** aan
onder `/v1/chat/completions`. Daardoor blijft je *agent-loop* (§5) en je
*client-code* (§4) ongewijzigd: je vervangt enkel het `base_url`- en
`api_key`-veld. De chat-pipeline uit §4 is nu *jouw* server in plaats van die
van een cloud-aanbieder.

**Technisch (OpenAI client → Ollama).** Ollama draait standaard op
`http://localhost:11434` en exposeert een OpenAI-compatibele server onder
`/v1`. De `openai`-SDK kan die rechtstreeks aanspreken:

```python
# pip install openai
from openai import OpenAI

# Point the client at your own local server instead of api.openai.com
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
    api_key="ollama",                       # placeholder; Ollama ignores the key
)

resp = client.chat.completions.create(
    model="llama3.1:8b",                    # model name as pulled in Ollama
    messages=[{"role": "user", "content": "Leg in één zin uit wat een agent is."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

**Technisch (curl én Python raw call).** Omdat het een standaard
`/v1/chat/completions` endpoint is, werkt ook elke HTTP-client. Zo bewijs je dat
er *geen* SDK nodig is — handig voor debuggen:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Wat is quantisatie?"}],
    "temperature": 0.2
  }'
```

```python
# equivalente raw call zonder SDK — geen productiecode
import json, urllib.request

payload = json.dumps({
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Wat is quantisatie?"}],
    "temperature": 0.2,
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:11434/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())
    print(data["choices"][0]["message"]["content"])
```

> **Vereenvoudiging:** streaming, tool-calls en authenticatie zijn weggelaten.
> Bij vLLM/TGI in productie voeg je wel een echte `api_key` / bearer-token en
> HTTPS toe; Ollama op localhost doet dat doorgaans niet.

---

### 13.4 Quantisatie (kwaliteit vs. middelen)

**Functioneel.** LLM-gewichten zijn standaard in float16/float32 (16–32 bits
per getal) — dat kost veel VRAM. Om modellen op bescheiden hardware te draaien,
worden de gewichten **gequantiseerd** naar lagere precisie. Dit verlaagt VRAM
en verhoogt snelheid, ten koste van (meestal beperkte) kwaliteit.

Veelgebruikte schema's:

- **GGUF q4 / q5 / q8** (llama.cpp) — q4 = ~4 bits, q8 = ~8 bits; q8 zit dicht
  bij de oorspronkelijke kwaliteit maar gebruikt dubbel zoveel geheugen als q4.
- **GPTQ** en **AWQ** — post-training quantisatie (vaak INT4) gericht op
  minimale kwaliteitsdaling bij activering van belangrijke gewichten.

| Schema | Precisie | VRAM-gebruik (relatief) | Kwaliteit | Typisch gebruik |
|--------|----------|-------------------------|-----------|-----------------|
| GGUF q4 | ~4 bit | 1.0× (laagst) | goed, mild verlies | weinig VRAM, CPU/oude GPU |
| GGUF q5 | ~5 bit | ~1.25× | zeer goed | middenweg lokaal |
| GGUF q8 | ~8 bit | ~2.0× | nagenoeg origineel | zoveel mogelijk kwaliteit lokaal |
| GPTQ / AWQ | INT4 | ~1.0× | goed (gewichts-bewust) | GPU-prod, vLLM/TGI |

**Technisch (VRAM-schatting per quant-niveau).** Een snelle vuistregel:
`VRAM ≈ parameters × bytes_per_parameter` (plus overhead voor KV-cache, zie §9).
Het snippet rekent dit voor een 7B-model:

```python
# pedagogische VRAM-schatting — geen productiecode
def vram_gb(n_params_billion, bits_per_weight):
    # 1e9 params * bits / 8 bits-per-byte / 1e9 bytes-per-GB
    bytes_used = n_params_billion * 1e9 * (bits_per_weight / 8)
    return bytes_used / 1e9  # in GB (model weights only, excl. KV-cache)

for label, bits in [("q4", 4), ("q5", 5), ("q8", 8), ("fp16", 16)]:
    print(f"{label:4s}: ~{vram_gb(7, bits):.1f} GB voor een 7B-model")

# afweging: agressievere quant bespaart geheugen maar kan precisie aantasten
# bij redeneer-/wiskundetaken (zie AGENTS.md 13.4)
```

> **Vereenvoudiging:** de echte VRAM omvat ook de KV-cache (§4.3, §9),
> activaties en de runtime-overhead — vaak +20–40%. En niet elke quant-methode
> verliest evenveel: AWQ/GPTQ zijn slimmer dan een dom "alles naar 4 bit".
> Bovenstaande formule is een *ondergrens*-schatting.

---

### 13.5 Keuze: cloud vs. self-host

**Functioneel.** Self-host bij strenge privacy (§14), hoog/gelijkmatig volume
(kosten, §15), offline eisen, of specifieke/fine-tuned modellen. Cloud bij
snelle start, weinig volume, of behoefte aan de grootste modellen zonder
GPU-investering.

**Technisch (break-even analyse).** De keuze is vaak een kwestie van
*kostendrempel*: vanaf hoeveel tokens per maand is self-host goedkoper dan
cloud? Een vereenvoudigd model:

```python
# pedagogische break-even — geen productiecode
def months_to_breakeven(gpu_capex, monthly_cloud_per_Mtok, price_per_Mtok_cloud):
    # maandelijkse besparing = (cloud-tarief - self-host marginale tarief≈0)
    saving_per_Mtok = price_per_Mtok_cloud  # self-host marginal kost ~0 hier
    monthly_tokens_M = gpu_capex / saving_per_Mtok if saving_per_Mtok else float("inf")
    # break-even in maanden = capex / maandelijkse besparing
    monthly_saving = monthly_tokens_M * saving_per_Mtok
    return gpu_capex / monthly_saving if monthly_saving else float("inf")

capex = 2500.0          # een tweedehands inference-GPU
price = 2.0             # $ per miljoen tokens bij cloud-aanbieder
breakeven_months = months_to_breakeven(capex, None, price)
print(f"Break-even bij ~{capex/price:.0f} miljoen tokens/maand "
      f"(= ~{breakeven_months:.1f} maand bij dat volume)")
```

**Aandachtspunten.** Je beheert zelf uptime, updates, schaling en GPU-kosten
(capex). Een goede tussenweg is een **anonymizing proxy** (§14) vóór een
cloud-API wanneer self-host niet haalbaar is: je houdt dan wél controle over
welke data vertrekt, zonder eigen GPU's.

> **Vereenvoudiging:** het snippet neemt de marginale self-host-kost als ~0
> (geen stroom/onderhoud) en een vast cloud-tarief. Echt wegen ook stroom,
> personeel, en variabele cloud-kortingen mee — maar het toont het *principe*
> van de break-even-drempel.

---

## Samenvatting (key takeaways)

- **Self-hosting** = je draait de *eigen inference-server*; de prompt verlaat je
  infrastructuur niet (direct relevant voor §14 privacy).
- De drie redenen: **data blijft binnen**, **geen per-token kosten**, **volledige
  controle** — sterk bij privacy-eisen, hoog volume en offline gebruik (§15).
- Runtimes kiezen op hardware/doel: **llama.cpp / Ollama / LM Studio** voor lokaal
  (weinig VRAM), **vLLM / TGI** voor GPU-productie, **ExLlama** voor maximale
  datacenter-snelheid.
- Dankzij een **OpenAI-compatibele API** (`/v1/chat/completions`) blijft je
  agent-loop (§5) ongewijzigd: enkel `base_url` + `api_key` wisselen (zie het
  Ollama-voorbeeld op `http://localhost:11434/v1`).
- **Quantisatie** (GGUF q4/q5/q8, GPTQ, AWQ) ruilt VRAM/snelheid tegen milde
  kwaliteitsdaling — de hefboom om grote modellen op bescheiden hardware te
  draaien.
- De **cloud-vs-self-host** keuze is een break-even-afweging: bij hoog/gelijkmatig
  volume wint capex het van variabele per-token OPEX.

Zie verder §14 (anonymizing proxy als tussenweg) en §15 (token economics voor
de exacte kostenvork).
