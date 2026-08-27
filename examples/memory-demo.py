"""
Memory demo (AGENTS.md §10 / docs/08-memory.md)

Shows the difference between short-term (working memory = the active context)
and long-term (external) memory in an agent:

  - short-term  : de huidige prompt-inhoud (werkt enkel binnen één run)
  - long-term   : externe store, hier een JSONL-bestand (werkt over sessies)
  - types       : episodisch (gebeurtenissen) / semantisch (feiten) /
                  procedureel (routines)
  - write/read  : save_memory / recall, lexical match (productie: via RAG §8)
  - consolidation & forget (TTL)

Pure standard library so it runs without installs:
    python examples/memory-demo.py
"""

import json
import os
import tempfile
import time


# ---------------------------------------------------------------------------
# Long-term memory (external store)
# ---------------------------------------------------------------------------

class LongTermMemory:
    def __init__(self, path: str | None = None):
        self.stores = {"episodic": {}, "semantic": {}, "procedural": {}}
        self.path = path or os.path.join(tempfile.gettempdir(),
                                          "agent_memory.jsonl")

    def save(self, kind: str, key: str, value, ttl: float | None = None) -> None:
        """Schrijf een herinnering weg (§10.4 write-patroon)."""
        if kind not in self.stores:
            raise ValueError(f"onbekend geheugentype: {kind}")
        self.stores[kind][key] = {"value": value, "ts": time.time(), "ttl": ttl}
        self._append_log(kind, key, value)

    def recall(self, kind: str, query: str) -> list[tuple[str, str]]:
        """Lees relevante herinneringen (§10.4 read-patroon).
        Lexicale match voor de demo; in productie embed je de vraag en zoek
        je semantisch via RAG (§8)."""
        q = query.lower()
        hits = []
        for k, e in self.stores[kind].items():
            blob = (k + " " + str(e["value"])).lower()
            if q in blob:
                hits.append((k, e["value"]))
        return hits

    def consolidate(self, episode_keys: list[str], fact_key: str,
                    fact_value: str) -> None:
        """Vat verspreide episodes samen tot één semantisch feit (§10.5)."""
        for k in episode_keys:
            self.stores["episodic"].pop(k, None)
        self.save("semantic", fact_key, fact_value)

    def forget_expired(self) -> int:
        """Verwijder verlopen entries (TTL, §10.5)."""
        now = time.time()
        removed = 0
        for kind in self.stores:
            for k in list(self.stores[kind]):
                e = self.stores[kind][k]
                if e["ttl"] is not None and now - e["ts"] > e["ttl"]:
                    del self.stores[kind][k]
                    removed += 1
        return removed

    def _append_log(self, kind: str, key: str, value) -> None:
        """Episodische/externe log (§10.2: file / episodic log)."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "key": key, "value": value},
                               ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("# Memory demo — kortetermijn vs langetermijn\n")

    # --- Kortetermijn (working memory): de actieve context ---
    working_memory = [
        {"role": "system", "content": "Je bent een behulpzame agent."},
        {"role": "user", "content": "Ik spreek Nederlands."},
    ]
    print("## Kortetermijn (working memory) — actieve context:")
    for m in working_memory:
        print(f"  - {m['role']}: {m['content']}")

    # --- Langetermijn: externe store ---
    ltm = LongTermMemory()
    print(f"\n## Langetermijn store (JSONL): {ltm.path}")

    ltm.save("semantic", "user_language", "Nederlands")
    ltm.save("episodic", "gesprek_2026-08-25",
             "besprak factuur X met klant Y")
    ltm.save("procedural", "facturatie_routine",
             "genereer PDF, verstuur per mail, log in SQL")

    # recall (semantic): de agent 'onthoudt' de taal
    print("\n## Recall (semantic) op 'nederlands':")
    for k, v in ltm.recall("semantic", "nederlands"):
        print(f"  - {k} = {v}")
    # recall (episodic): terugvinden van een eerdere gebeurtenis
    print("\n## Recall (episodic) op 'factuur':")
    for k, v in ltm.recall("episodic", "factuur"):
        print(f"  - {k} = {v}")

    # --- Consolidatie: twee episodes -> één feit ---
    ltm.save("episodic", "feedback_1", "user vroeg om korter antwoord")
    ltm.save("episodic", "feedback_2", "user vond lang antwoord onbruikbaar")
    ltm.consolidate(["feedback_1", "feedback_2"], "user_prefers_concise",
                    "True")
    print("\n## Na consolidatie (episodes -> semantisch feit):")
    print("  semantic:", list(ltm.stores["semantic"].keys()))

    # --- TTL / vergeten ---
    ltm.save("episodic", "tijdelijke_note", "vervalt meteen", ttl=0)
    removed = ltm.forget_expired()
    print(f"\n## Forget (TTL): {removed} verlopen entry verwijderd.")

    # --- Toon de externe log ---
    print("\n## Inhoud van het externe JSONL-bestand:")
    with open(ltm.path, encoding="utf-8") as f:
        for line in f:
            print("  " + line.strip())


if __name__ == "__main__":
    main()
