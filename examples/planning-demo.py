"""
Planning demo (AGENTS.md §11 / docs/09-planning.md)

Shows plan-and-execute for a complex, multi-step task (§11.2):

  - Planner  : zet een doel om in een volledig plan (lijst van stappen)
  - Executor : voert elke stap uit via een tool; bij een fout -> error recovery
  - vs ReAct : hier maken we EERST het plan, dan voeren we uit (§11.5)

Pure standard library so it runs without installs:
    python examples/planning-demo.py
"""

from dataclasses import dataclass


@dataclass
class Step:
    name: str
    tool: str
    args: dict
    done: bool = False
    attempts: int = 0


# ---------------------------------------------------------------------------
# Planner (in productie: een LLM, zie §11.2 / §20)
# ---------------------------------------------------------------------------

def planner(goal: str) -> list[Step]:
    """Regel-gebaseerde planner: vertaalt een doel naar een plan."""
    g = goal.lower()
    plan: list[Step] = []
    if "concurrent" in g or "onderzoek" in g:
        plan.append(Step("verzamel concurrentdata", "web_search",
                         {"query": "concurrenten"}))
        plan.append(Step("analyseer data", "analyze", {}))
    if "rapport" in g:
        plan.append(Step("schrijf rapport", "write_doc",
                         {"title": "Concurrentieanalyse"}))
    if "mail" in g or "e-mail" in g:
        plan.append(Step("mail rapport", "send_email", {"to": "baas@co.example"}))
    return plan


# ---------------------------------------------------------------------------
# Tools (gesimuleerd; send_email faalt de eerste keer: geen Go)
# ---------------------------------------------------------------------------

def run_tool(tool: str, args: dict, state: dict) -> tuple[bool, str]:
    if tool == "web_search":
        state["data"] = ["CompA", "CompB", "CompC"]
        return True, "3 concurrenten gevonden"
    if tool == "analyze":
        n = len(state.get("data", []))
        return True, f"analyse klaar over {n} concurrenten"
    if tool == "write_doc":
        state["report"] = "Concurrentieanalyse v1"
        return True, "rapport geschreven"
    if tool == "send_email":
        # §6.5: niet-idempotente actie vereist expliciete gebruikers-Go
        if not state.get("confirmed"):
            return False, "geweigerd: geen expliciete gebruikers-Go (§6.5)"
        return True, "mail verzonden"
    return False, "onbekende tool"


# ---------------------------------------------------------------------------
# Executor met error recovery (§11.4)
# ---------------------------------------------------------------------------

def execute(plan: list[Step]) -> tuple[bool, dict]:
    state: dict = {}
    for step in plan:
        step.attempts += 1
        ok, msg = run_tool(step.tool, step.args, state)
        if ok:
            step.done = True
            print(f"  [OK]   {step.name}: {msg}")
        else:
            print(f"  [FAIL] {step.name}: {msg}")
            # error recovery: corrigeer en probeer opnieuw (reflexion-stijl)
            if step.tool == "send_email":
                print("        -> recovery: gebruikers-Go vragen + retry")
                state["confirmed"] = True
                step.attempts += 1
                ok2, msg2 = run_tool(step.tool, step.args, state)
                if ok2:
                    step.done = True
                    print(f"  [OK]   {step.name} (na recovery): {msg2}")
                else:
                    print("  [ESC]  escaleren naar gebruiker")
    return all(s.done for s in plan), state


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("# Planning demo — plan-and-execute met error recovery\n")

    goal = "Onderzoek de concurrenten, schrijf een rapport en mail het."
    print(f"## Doel: {goal!r}\n")

    plan = planner(goal)
    print("## Plan (eerst volledig opstellen, dan uitvoeren):")
    for i, s in enumerate(plan, 1):
        print(f"  {i}. {s.name}  [{s.tool}]")
    print()

    print("## Executie (stap voor stap):")
    success, _ = execute(plan)
    print(f"\n## Resultaat: {'alle stappen voltooid' if success else 'niet voltooid'}")


if __name__ == "__main__":
    main()
