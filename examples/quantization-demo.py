"""
quantization-demo.py  --  illustratie bij §13.4 (Quantisatie)

Toont de kernafweging van quantisatie: **VRAM / snelheid** ruilen tegen
**(meestal beperkte) kwaliteit**. We berekenen de ondergrens-VRAM per
quant-niveau voor een 7B-model, en tonen hoe agressievere quantisatie de
geheugendruk verlaagt (§13.4).

Formule (ondergrens):  VRAM ≈ n_params × bits_per_weight / 8  (excl. KV-cache).

Vereenvoudiging: de echte VRAM omvat ook de KV-cache (§4.3, §9), activaties
en runtime-overhead — vaak +20–40%. En niet elke methode verliest evenveel:
AWQ/GPTQ zijn slimmer dan dom "alles naar N bits". De formule hier is een
ondergrens-schatting, verwant aan 11-self-hosting.md §13.4.
"""

# bits_per_weight per schema (benadering)
SCHEMES = {
    "GGUF q4": 4,
    "GGUF q5": 5,
    "GGUF q8": 8,
    "GPTQ/AWQ (INT4)": 4,
    "fp16 (basis)": 16,
}


def vram_gb(n_params_billion: float, bits_per_weight: int) -> float:
    """Ondergrens VRAM in GB voor de gewichten alleen."""
    bytes_used = n_params_billion * 1e9 * (bits_per_weight / 8)
    return bytes_used / 1e9


if __name__ == "__main__":
    model_b = 7.0
    print(f"Ondergrens VRAM voor een {model_b:.0f}B-model (gewichten alleen):\n")
    base = vram_gb(model_b, 16)
    for label, bits in SCHEMES.items():
        v = vram_gb(model_b, bits)
        factor = base / v if v else float("inf")
        print(f"  {label:18s}: ~{v:4.1f} GB   ({factor:4.1f}x minder dan fp16)")

    print("\nAfweging (§13.4):")
    print("  - q4 bespaart het meeste geheugen, maar kan precisie aantasten")
    print("    bij redeneer-/wiskundetaken.")
    print("  - q8 zit dicht bij de oorspronkelijke kwaliteit, tegen ~2x VRAM.")
    print("  - Kies op basis van beschikbare VRAM (zie 11-self-hosting.md).")
