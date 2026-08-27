"""
Demo: §20 Agentic & model types (generiek vs coding).

Toont concreet:
 1. Router tussen een generieke en een coding agent.
 2. Verschil in output-stijl van een codex-type vs algemeen model.
 3. Hoe de loop de "instructie" van het model parsed en uitvoert (signaling).

Draait met enkel de Python-standaardbibliotheek (geen externe dependencies).
Zie docs/16-agentic-model-types.md voor de volledige uitleg.
"""

import json


def agent_type_for(task: str) -> str:
    """Kies een coding-agent wanneer de taak code-acties vereist, anders generiek."""
    code_signals = ("schrijf", "refactor", "test", "bug", "repo", "fix", "code")
    if any(s in task.lower() for s in code_signals):
        return "coding"      # + filesystem/shell/git/test-runner
    return "generic"         # + search/rag/calculator


def model_output_style(is_codex_model: bool, task: str) -> str:
    """Een codex-type model neigt naar een gestructureerde actie; een
    algemeen model naar vrije tekst-uitspraak."""
    if is_codex_model:
        return json.dumps(
            {"tool": "run_shell", "arguments": {"command": f"pytest {task}"}}
        )
    return f"Je zou de test kunnen draaien met: pytest {task}"


def run_loop_step(model_instruction: str):
    """De loop parsed de instructie van het model en voert ze uit (de runtime,
    niet het model zelf)."""
    try:
        call = json.loads(model_instruction)        # bv. {"tool": ..., "arguments": {...}}
    except json.JSONDecodeError:
        return "ANSWER", model_instruction          # geen actie -> eindantwoord

    tool, args = call["tool"], call["arguments"]
    if tool == "run_shell":
        result = f"(uitvoer van: {args['command']})"
    else:
        result = f"(resultaat van {tool})"
    return "OBSERVE", result                        # terug in context voor volgende stap


if __name__ == "__main__":
    print("1. Agent-type router")
    print("   ", agent_type_for("schrijf een test voor parser.py"))
    print("   ", agent_type_for("wat is de omzet in Q2?"))

    print("\n2. Output-stijl (codex vs algemeen)")
    print("   codex    ->", model_output_style(True, "tests/test_parser.py"))
    print("   algemeen ->", model_output_style(False, "tests/test_parser.py"))

    print("\n3. Loop voert model-instructie uit (signaling)")
    kind, payload = run_loop_step(
        '{"tool": "run_shell", "arguments": {"command": "pytest tests/test_agent.py"}}'
    )
    print("   ", kind, "->", payload)
    kind2, payload2 = run_loop_step("Het antwoord is 42.")
    print("   ", kind2, "->", payload2)
