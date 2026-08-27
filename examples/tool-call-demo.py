"""
tool-call-demo.py  --  illustratie bij §6 (Tools / Function calling)

Toont de tool-call cyclus (§6.1) met een JSON-Schema tool-definitie (§6.2):

  1. Definitie : een tool-schema (naam, description, parameters).
  2. Keuze     : het model 'kiest' de tool + vult argumenten in.
  3. Uitvoering: de runtime voert de functie uit (HIER een echte aanroep).
  4. Terugkopp : het resultaat wordt als tool_result terug in de context.
  5. Vervolg   : het model redeneert verder en formuleert een antwoord.

De 'model'-output is hier een dictionary (nzero native function calling, §6.3);
de runtime parsed en executeert echt. In productie geeft de API structured
`tool_calls` terug en doet de SDK het parsen.

Vereenvoudiging: `mock_model_tool_call` levert een kant-en-klare aanroep;
een echt model genereert die uit de prompt + het schema.
"""

import json


# ----------------------------------------------------------------------------
# 1. Tool-definitie (§6.2) -- JSON-Schema-achtig
# ----------------------------------------------------------------------------
GET_WEATHER = {
    "name": "get_weather",
    "description": "Haal de huidige weersvoorspelling op voor een locatie.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Stad en land, bv. 'Brussel, België'",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}


# ----------------------------------------------------------------------------
# 2/3. Runtime: parse de tool-aanroep en voer de echte functie uit (§6.1)
# ----------------------------------------------------------------------------
def get_weather(location: str, unit: str = "celsius") -> str:
    """Echte (hier gestubde) tool-implementatie.

    Retourneert een JSON-string (dubbele quotes) zodat de caller hem met
    json.loads kan terugparsen — net zoals een echte tool het doet (§6.1).
    """
    # In productie: HTTP-call naar een weer-API. Hier een vaste return.
    return json.dumps({"location": location, "temp": 18,
                       "unit": unit, "sky": "bewolkt"}, ensure_ascii=False)


def execute_tool_call(tool_call: dict) -> str:
    """Voer een {name, arguments} aanroep uit en geef het resultaat terug."""
    name = tool_call["name"]
    args = tool_call.get("arguments", {})

    # 3. validatie van argumenten vóór executie (§6.5 veiligheid)
    if name == "get_weather":
        if "location" not in args:
            return "ERROR: verplicht argument 'location' ontbreekt"
        return get_weather(**args)
    return f"ERROR: onbekende tool '{name}'"


def mock_model_tool_call() -> dict:
    """Plaatsvervanger voor een natieve function-calling API-aanroep (§6.3)."""
    return {
        "name": "get_weather",
        "arguments": {"location": "Brussel, België", "unit": "celsius"},
    }


if __name__ == "__main__":
    print("Beschikbare tool-schema:")
    print(json.dumps(GET_WEATHER, indent=2, ensure_ascii=False))

    # 2. model 'kiest' de tool
    tool_call = mock_model_tool_call()
    print("\nModel -> tool_call:", tool_call)

    # 3+4. runtime voert uit en krijgt resultaat terug
    result = execute_tool_call(tool_call)
    print("Runtime -> tool_result:", result)

    # 5. model redeneert verder met het resultaat in de context
    final_answer = (
        f"Het weer in {tool_call['arguments']['location']} is momenteel "
        f"{json.loads(result)['temp']}°C en bewolkt."
    )
    print("\nModel -> eindantwoord:", final_answer)
    print("\nZie 04-tools.md (§6) voor parallelle calls, chaining, sub-agenten "
          "en de ReAct-variant (§6.3).")
