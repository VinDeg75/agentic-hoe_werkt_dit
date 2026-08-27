"""
Token economics demo (AGENTS.md §15 / docs/13-token-economics.md)

Shows why an agent loop makes tokens explode (§15.2) and which levers help
(§15.3):

  - naive    : de volledige context wordt bij elke stap opnieuw gestuurd
  - caching  : de gemeenschappelijke prefix (system prompt) is gecached
               (goedkoper dan re-senden, factor 0.1 in dit voorbeeld)
  - compress : context ingekort tot recente stappen (sliding window, §9)

Pure standard library so it runs without installs:
    python examples/token-economics-demo.py
"""

# Prijzen per 1k tokens (vereenvoudigde fictieve tarieven, USD)
PRICE_INPUT = 0.01
PRICE_CACHED = 0.001
PRICE_OUTPUT = 0.02


def approx_tokens(text: str) -> int:
    return len(text.split())


def simulate(iterations: int, system_prompt: str, with_cache: bool,
             with_compress: bool) -> tuple[int, list[int]]:
    """
    Bereken de gebillingde input-tokens per stap voor een agent-loop.
    `history` groeit elke stap (geschiedenis + RAG + tool-resultaten, §15.2).
    """
    history: list[str] = []
    sys_tok = approx_tokens(system_prompt)
    per_step_tokens: list[int] = []

    for i in range(1, iterations + 1):
        # elke stap voegt groeiende context toe (tool-resultaten stapelen)
        history.append(f"stap {i} resultaat " + "x " * (i * 2))
        ctx_parts = [system_prompt] + history
        if with_compress and len(history) > 3:
            # sliding window: enkel recente 3 stappen (§9.2)
            ctx_parts = [system_prompt] + history[-3:]

        total_ctx = approx_tokens(" ".join(ctx_parts))
        if with_cache:
            # prefix is een cache-hit: telt aan gereduceerd tarief
            suffix = total_ctx - sys_tok
            billed = suffix + int(sys_tok * 0.1)
        else:
            billed = total_ctx
        per_step_tokens.append(billed)

    return sum(per_step_tokens), per_step_tokens


def cost(per_step_tokens_total: int, output_tokens: int) -> float:
    cached_part = 0  # in dit voorbeeld rekenen we enkel input vs cached-input
    return per_step_tokens_total / 1000 * PRICE_INPUT + output_tokens / 1000 * PRICE_OUTPUT


def main() -> None:
    print("# Token economics demo — agent-loop token-explosie & hefbomen\n")

    system_prompt = "Je bent een agent die taken uitvoert. " * 8  # vaste prefix
    iterations = 6
    output_tokens = 120  # gegenereerd antwoord over de loop

    print(f"Iteraties: {iterations} | system-prefix tokens: "
          f"{approx_tokens(system_prompt)} | output: {output_tokens} tok\n")

    totals = {}
    for label, (cache, compress) in {
        "naive    ": (False, False),
        "caching  ": (True, False),
        "compress ": (False, True),
        "cache+comp": (True, True),
    }.items():
        total, per = simulate(iterations, system_prompt, cache, compress)
        totals[label.strip()] = total
        print(f"  {label.rstrip():9} totaal input-tokens: {total:5}  "
              f"(per stap: {per})")

    naive = totals["naive"]
    cached = totals["caching"]
    comp = totals["compress"]
    cc = totals["cache+comp"]

    print("\n## Besparing t.o.v. naive:")
    print(f"  caching   : {naive - cached:5} tok bespaard "
          f"({(1 - cached/naive)*100:.0f}%)")
    print(f"  compress  : {naive - comp:5} tok bespaard "
          f"({(1 - comp/naive)*100:.0f}%)")
    print(f"  cache+comp: {naive - cc:5} tok bespaard "
          f"({(1 - cc/naive)*100:.0f}%)")

    print("\n## Geschatte kosten (USD, input $0.01/1k, output $0.02/1k):")
    print(f"  naive     : ${cost(naive, output_tokens):.4f}")
    print(f"  cache+comp: ${cost(cc, output_tokens):.4f}  "
          f"(break-even met self-host bij hoog volume, §15.4)")


if __name__ == "__main__":
    main()
