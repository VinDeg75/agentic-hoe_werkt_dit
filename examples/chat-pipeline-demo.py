"""
chat-pipeline-demo.py  --  illustratie bij §4 (LLM in chatmodus)

Dit script toont wat de inference-SERVER doet vanaf het moment dat een
plaintext-bericht binnenkomt, tot de geassembleerde prompt klaar is om
getokeniseerd en door het model verwerkt te worden (zie §4.1):

  plaintext vraag
    -> inputvalidatie
    -> prompt-assemblage (system + history + user)
    -> chat template (ChatML-achtige special tokens)
    -> klaar voor tokenisatie (§3.1) -> forward pass (§4.2)

De code is pedagogisch. Er wordt GEEN echt model aangeroepen; we tonen enkel de
server-zijde "van plaintext tot prompt-string". Vereenvoudigingen staan in
commentaar.
"""

from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Stap 1: inputvalidatie (server-side, §4.1)
# ----------------------------------------------------------------------------
def validate_input(user_text: str, max_chars: int = 8000) -> str:
    """Minimale server-side validatie: leeg? te lang?"""
    text = user_text.strip()
    if not text:
        raise ValueError("Lege gebruikersinput geweigerd.")
    if len(text) > max_chars:
        raise ValueError(f"Input te lang: {len(text)} > {max_chars} tekens.")
    # In productie: rate-limiting, injection-checks, permissies, ... (§6.5, §14)
    return text


# ----------------------------------------------------------------------------
# Stap 2 + 3: prompt-assemblage + chat template (§4.1)
# ----------------------------------------------------------------------------
CHAT_TEMPLATE = (  # ChatML-achtig; echte modellen hebben eigen template
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class Conversation:
    system: str = "Je bent een behulpzame assistent."
    history: list = field(default_factory=list)   # list of (role, content)
    user: str = ""

    def assemble(self) -> str:
        """Bouw de volledige prompt-string uit system, history en user."""
        # History krijgt per turn zijn eigen im_start/im_end-blok
        parts = []
        for role, content in self.history:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        history_blob = "".join(parts)

        # System-prompt eerst, dan history, dan de nieuwe user-turn
        prompt = (
            f"<|im_start|>system\n{self.system}<|im_end|>\n"
            f"{history_blob}"
            f"<|im_start|>user\n{self.user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt


def summarize_history(history, keep_recent: int = 2):
    """Plaatsvervanger voor §9 compression: houd enkel recente turns.

    Vereenvoudiging: echte compressie (§9.2.1) laat een LM de oudere turns
    samenvatten; hier houden we ze enkel over.
    """
    return history[-keep_recent:]


if __name__ == "__main__":
    raw = "  Wat is de omzet in België?  "
    user = validate_input(raw)

    conv = Conversation(
        system="Je bent een financieel analist.",
        history=[
            ("user", "Welke regio's hebben we?"),
            ("assistant", "BE, NL en FR."),
            ("user", "Haal het Q2-rapport op."),
            ("assistant", "Ik gebruik RAG om het rapport op te halen."),
        ],
        user=user,
    )

    # (optioneel) pas compressie toe vóór assemblage (zie §9)
    conv.history = summarize_history(conv.history, keep_recent=1)

    prompt = conv.assemble()
    print("=== Geassembleerde prompt (klaar voor tokenisatie) ===\n")
    print(prompt)
    print("=== (na deze string volgt de forward pass -> logits -> sampling,")
    print("     zie §4.2 / §4.3 en 02-llm-chatmode.md) ===")

    # Tel "woorden" als ruwe grootte-schatter (echte telling: tokenizer, §3.1)
    print(f"\nRuw aantal karakters in prompt: {len(prompt)}")
