"""
agentic-loop-demo.py  --  illustratie bij §5 (Agentic loop)

Een minimale maar WERKENDE agentic loop (§5):

    perceive -> reason -> act -> observe -> repeat -> respond

De 'model' is hier gemockt (een vastgelegde 'thought/action'-script) zodat je
de LOOP-struktuur ziet zonder een API-key. De tool die de loop aanroept is WEL
echt: een rekenmachine-functie. Zo toon je de 'dirigent'-rol van de loop:
hij beslist per iteratie welke stap volgt en voegt het resultaat terug in de
context voor de volgende redeneer-stap.

Vereenvoudiging: in productie is `reason()` een echte LLM-call die zélf
beslist of een tool aangeroepen wordt (zie §5, §6). Hier is dat scripted.
"""

# ----------------------------------------------------------------------------
# De 'Act'-kant: een echte tool die de runtime uitvoert (§6)
# ----------------------------------------------------------------------------
def calculator(expression: str) -> str:
    """Een simpele, veilige rekentool (alleen + - * / en getallen)."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "ERROR: niet-toegestane tekens in expressie"
    try:
        # eval() is hier enkel OK omdat de input gevarenbestendigd is;
        # in productie gebruik je een echte parser (§6.5 veiligheid).
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # tool-fouten gaan terug naar de loop (§6.5)
        return f"ERROR: {exc}"


TOOLS = {"calculator": calculator}


# ----------------------------------------------------------------------------
# De 'Reason'-kant: een gemockt model dat thought/action produceert
# ----------------------------------------------------------------------------
class MockModel:
    """Levert een vastgelegde redeneer/actie-cyclus (geen echte LLM)."""
    def __init__(self, script):
        self.script = script
        self.i = 0

    def reason(self, context: str) -> str:
        line = self.script[self.i]
        self.i += 1
        return line


# ----------------------------------------------------------------------------
# De loop zelf (§5): perceive -> reason -> act -> observe -> respond
# ----------------------------------------------------------------------------
def run_agent(goal: str, model: MockModel, max_steps: int = 6):
    context = f"Goal: {goal}"
    trajectory = []

    for step in range(max_steps):
        # 1. REASON: het model beslist de volgende zet op basis van context
        decision = model.reason(context)

        if decision.startswith("ANSWER:"):
            return decision.replace("ANSWER:", "").strip(), trajectory

        if decision.startswith("ACTION:"):
            # 2. ACT: roep de tool aan (runtime, niet het model)
            tool_call = decision[len("ACTION:"):].strip()   # "calculator(120+90)"
            name, arg = tool_call.split("(", 1)
            arg = arg.rstrip(")")
            result = TOOLS[name](arg)            # 3. OBSERVE: resultaat terug
            context += f"\nObservation: {result}"
            trajectory.append((tool_call, result))
            continue

        # Geen expliciete ACTIE/ANTWOORD -> beschouw als interne 'Thought'
        # (redenering); voeg toe aan de context en laat de loop opnieuw redeneren.
        context += f"\nThought: {decision}"
        continue

    return "MAX_STEPS bereikt", trajectory


if __name__ == "__main__":
    script = [
        "Ik moet de totale omzet berekenen uit de opgehaalde delen.",
        "ACTION: calculator(120+90+70)",
        "De som is bekend; ik kan nu antwoorden.",
        "ANSWER: De totale omzet is 280.",
    ]
    answer, traj = run_agent("wat is de totale omzet?", MockModel(script))
    print("Antwoord :", answer)
    print("Traject  :")
    for call, res in traj:
        print(f"  - {call}  ->  {res}")
    print("\nZie 03-agentic-loop.md (§5) en 04-tools.md (§6) voor de echte, "
          "LLM-gestuurde variant met function calling.")
