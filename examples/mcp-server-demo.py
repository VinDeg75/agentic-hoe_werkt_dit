"""
mcp-server-demo.py  --  illustratie bij §7 (MCP — Model Context Protocol)

Een MINIMALE, dependency-vrije MCP-server die spreekt over **stdio** met
JSON-RPC 2.0 (§7.1). Hij toont de twee belangrijkste primitieven (§7.2):

  - Tools      : een uitvoerbare functie ("search_kb")
  - Resources  : leesbare data die de host in de context kan laden

De server verwerkt één JSON-RPC-bericht per regel op stdin en schrijft
antwoorden naar stdout. Start hem vanuit een MCP-host, of test handmatig:

    python mcp-server-demo.py
    {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
    {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
    {"jsonrpc":"2.0","id":3,"method":"tools/call",\\
       "params":{"name":"search_kb","arguments":{"query":"omzet"}}}

VERGEET de `\\` niet als je dit in een shell typt; in een echte host doet de
MCP-client de serialisatie voor jou.

De echte MCP-SDK (pip install mcp) doet dit als:

    from mcp.server import Server
    s = Server("demo")
    @s.list_tools()
    async def list_tools(): ...
    @s.call_tool()
    async def call_tool(name, args): ...

Hier schrijven we het protocol met de stdlib om het mechanisme te tonen.
Vereenvoudiging: geen authenticatie, geen resources/subscriben, geen
notifications — enkel request/response over stdio.
"""

import json
import sys


# ----------------------------------------------------------------------------
# De "Tools"-primitief (§7.2): een uitvoerbare functie
# ----------------------------------------------------------------------------
def search_kb(query: str) -> str:
    """Een fictieve kennisbank-zoektool (in productie: echte retrieval, §8)."""
    kb = {
        "omzet": "Q2-omzet BE: 1.2M, NL: 0.9M, FR: 0.7M.",
        "prijs": "Standaardlicentie kost 99 EUR/maand per gebruiker.",
    }
    for key, answer in kb.items():
        if key in query.lower():
            return answer
    return "Geen resultaat gevonden in de kennisbank."


TOOLS = {
    "search_kb": {
        "name": "search_kb",
        "description": "Zoek in de interne kennisbank op trefwoord.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Het zoekwoord, bv. 'omzet'."}
            },
            "required": ["query"],
        },
    }
}


# ----------------------------------------------------------------------------
# JSON-RPC 2.0 dispatch (§7.1)
# ----------------------------------------------------------------------------
def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        # Handshake: de host en server wisselen capabilities uit.
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "demo-mcp", "version": "0.1"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {"tools": list(TOOLS.values())},
        }

    if method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments", {})
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32601, "message": f"unknown tool {name}"}}
        content = search_kb(**args)
        # MCP-tools retourneren content-blokken (§7.2 Tool == model-controlled)
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {"content": [{"type": "text", "text": content}],
                       "isError": False},
        }

    # Onbekende method -> JSON-RPC error
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main():
    """Lees JSON-RPC-berichten (één per regel) van stdin, antwoord op stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"jsonrpc": "2.0", "id": None,
                              "error": {"code": -32700, "message": str(exc)}}))
            sys.stdout.flush()
            continue
        response = handle(request)
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    # Werk je in een shell, dan is bovenstaande handmatige test nuttig.
    # Via een MCP-host (bijv. Claude Desktop / een agent §5) gebeurt dit
    # automatisch; de host spawnT deze server als subprocess (§7.1, stdio).
    main()
