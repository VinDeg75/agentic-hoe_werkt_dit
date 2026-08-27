"""
Compression demo (AGENTS.md §9 / docs/07-compression.md)

Shows context-window management for an agent loop (§5). Naarmate een agent
langer bezig is, groeit de context (geschiedenis + tool-resultaten + RAG).
Compression houdt die beheersbaar in lengte, kosten en latency:

  - history summarization : oude turns samenvatten tot één korte digest
  - sliding window         : enkel de meest recente turns houden (StreamingLLM:
                             belangrijkste + recentste behouden)
  - budget-gestuurde trig  : een ContextManager comprimeert als het token-budget
                             overschreden is

Pure standard library so it runs without installs:
    python examples/compression-demo.py
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Ruw token-proxy: aantal woorden. (Vereenvoudiging; een echte tokenizer
    telt subwoorden, zie §3.1.)"""
    return len(text.split())


def summarize_old_turns(turns: list[dict], keep_recent: int = 2) -> list[dict]:
    """
    History summarization (§9.2): collapse oude turns tot één system-digest.
    Houd de meest recente `keep_recent` turns intact; vervang de rest door een
    één-regelige samenvatting. Trade-off (§9.3): nuances gaan verloren, maar de
    hoofdlijn + het aantal tool-aanroepen blijven behouden.
    """
    if len(turns) <= keep_recent + 1:
        return turns
    old = turns[:-keep_recent]
    recent = turns[-keep_recent:]
    n_actions = sum(1 for t in old if t["role"] == "tool")
    first_q = old[0]["content"] if old[0]["role"] == "user" else "(geen user)"
    summary = {
        "role": "system",
        "content": (f"[SAMENVATTING van {len(old)} eerdere turns: "
                    f"{n_actions} tool-aanroep(en); "
                    f"eerste vraag: {first_q[:50]!r}]"),
    }
    return [summary] + recent


def sliding_window(turns: list[dict], max_turns: int = 4,
                   keep_first: bool = True) -> list[dict]:
    """
    Sliding window / context-window management (§9.2): houd enkel de recentste
    turns. Met `keep_first` behouden we de eerste (attention-sink, à la
    StreamingLLM) zodat de basisinstructie niet wegvalt.
    """
    if len(turns) <= max_turns:
        return turns
    if keep_first:
        return [turns[0]] + turns[-(max_turns - 1):]
    return turns[-max_turns:]


# ---------------------------------------------------------------------------
# Budget-driven context manager
# ---------------------------------------------------------------------------

class ContextManager:
    def __init__(self, token_budget: int = 45, strategy: str = "summarize"):
        self.budget = token_budget
        self.strategy = strategy
        self.turns: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self.compress_if_needed()

    def total_tokens(self) -> int:
        return sum(approx_tokens(t["content"]) for t in self.turns)

    def compress_if_needed(self) -> bool:
        if self.total_tokens() <= self.budget:
            return False
        before = len(self.turns)
        if self.strategy == "summarize":
            self.turns = summarize_old_turns(self.turns, keep_recent=2)
        elif self.strategy == "sliding":
            self.turns = sliding_window(self.turns, max_turns=4)
        print(f"  [compress {self.strategy}] {before} -> {len(self.turns)} turns "
              f"({self.total_tokens()} tokens)")
        return True


# ---------------------------------------------------------------------------
# Demo: een agent-loop die het budget overschrijdt
# ---------------------------------------------------------------------------

def main() -> None:
    print("# Compression demo — agent-loop met groeiende context\n")

    cm = ContextManager(token_budget=45, strategy="summarize")

    # Een realistische, langdurige agent-run (redeneer/act/observe herhaald)
    script = [
        ("user",  "Wat is de omzet in Q2 in Belgie?"),
        ("assistant", "Ik haal het Q2-rapport op via RAG en bereken de som."),
        ("tool",  "RAG hit: q2_rapport.pdf p.12 — 'Omzet Belgie Q2: 8% groei "
                  "naar 1.24M euro, vooral software en diensten."),
        ("assistant", "Ik tel de relevante regels op."),
        ("tool",  "calc(1.24M) = 1.24M euro omzet Belgie Q2."),
        ("user",  "En wat is de omzet in Nederland voor dezelfde periode?"),
        ("assistant", "Ook hier haal ik het rapport op en reken ik."),
        ("tool",  "RAG hit: q2_rapport.pdf p.14 — 'Omzet Nederland Q2: 3% groei "
                  "naar 0.98M euro, vooral hardware."),
        ("assistant", "Ik vergelijk Belgie en Nederland."),
        ("tool",  "calc(1.24M - 0.98M) = 0.26M euro verschil ten gunste van Belgie."),
    ]

    print("## Stap-voor-stap (elke turn wordt toegevoegd + evt. gecomprimeerd):")
    for role, content in script:
        cm.add(role, content)
        print(f"  + {role:9} ({approx_tokens(content):2} tok) -> totaal "
              f"{cm.total_tokens():2} tok, {len(cm.turns)} turns")

    print("\n## Gecomprimeerde context die naar het model gaat:")
    for t in cm.turns:
        print(f"  - {t['role']:9}: {t['content']}")

    print(f"\n## Effect: {sum(approx_tokens(c) for _, c in script)} ruwe tokens "
          f"-> {cm.total_tokens()} tokens in context "
          f"({(1 - cm.total_tokens()/sum(approx_tokens(c) for _, c in script))*100:.0f}% bespaard)")


if __name__ == "__main__":
    main()
