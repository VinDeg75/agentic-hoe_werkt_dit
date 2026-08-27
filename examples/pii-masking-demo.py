"""
pii-masking-demo.py  --  illustratie bij §14 (Anonymizing proxy / PII)

Toont de WERKWIJZE van een anonymizing proxy (§14.2) die voor de prompt de
API bereikt:

  1. DETECTIE : vind PII (e-mail, IBAN, telefoon, naam) via regex / lijst.
  2. MASKERING: vervang PII door een CONSISTENTE placeholder (bv. <<EMAIL_1>>).
  3. UN-MASK  : vertaal in het antwoord de placeholders terug naar de echte waarde.

De mock "LLM" hier geeft een antwoord dat de placeholders letterlijk teruggeeft;
in productie stuurt de proxy de gemaakte prompt naar de echte API en maskeert
daarna het antwoord terug (§14.2).

Vereenvoudiging: de detectie is minimaal (regex + een kleine namen-lijst).
Echte proxies gebruiken NER-modellen / Microsoft Presidio (§14.2) en bewaren
de mapping beveiligd (de mapping IS zelf gevoelig, §14.3). Ook wordt hier NIET
getoond dat maskering soms context wegneemt die het model nodig heeft (§14.3).

Antwoord op de twee open punten:
  * CONSISTENTIE: dezelfde waarde krijgt altijd dezelfde placeholder, zodat het
    model de verbanden (en de un-mask mapping) correct houdt. Zie de
    reverse-mapping in AnonymizingProxy._placeholder_for().
  * ZONDER LM: er bestaan inderdaad tools die geen LLM nodig hebben:
      - Regelgebaseerd / lexicaal: regex (zoals hier), allowlists, dictionaries.
      - Ged ediceerde bibliotheken: Microsoft Presidio (recognizer-catalog +
        spaCy NER), AWS Comprehend, Azure Text Analytics — allemaal geen LLM.
      - Een KLEIN zelf-getraind neuraal netwerk (bv. BiLSTM-CRF of een kleine
        transformer fijn-gestemd op gelabelde PII-data, zoals spaCy NER of een
        HuggingFace-model) kan PII herkennen en draait lokaal op CPU, goedkoop
        en deterministisch. Zie tiny_ner_demo() hieronder voor een werkend
        (mini-) voorbeeld van zo'n model-vrije aanpak.
"""

import re


# ----------------------------------------------------------------------------
# 1. Detectie-patronen (regex + namenlijst)
# ----------------------------------------------------------------------------
PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "IBAN": re.compile(r"BE\d{2} ?\d{4} ?\d{4} ?\d{4}"),
    "TEL": re.compile(r"\+?\d[\d\s-]{7,}\d"),
}
NAMES = ["Jan Peeters", "Alice Vermeulen"]   # vereenvoudigde NER-vervanging


# ----------------------------------------------------------------------------
# 2+3. De proxy: mask (reversible + consistent) + unmask
# ----------------------------------------------------------------------------
class AnonymizingProxy:
    def __init__(self):
        self.mapping = {}        # placeholder -> origineel (reversibel, §14.3)
        self.reverse = {}        # origineel  -> placeholder (consistentie)
        self.counters = {}       # per categorie een teller voor unieke placeholders

    def _next_placeholder(self, category: str) -> str:
        self.counters[category] = self.counters.get(category, 0) + 1
        return f"<<{category}_{self.counters[category]}>>"

    def _placeholder_for(self, category: str, value: str) -> str:
        # CONSISTENTIE: bestaat er al een placeholder voor deze exacte waarde,
        # hergebruik die zodat het model dezelfde entiteit herkent (§14.2/§14.3).
        if value in self.reverse:
            return self.reverse[value]
        ph = self._next_placeholder(category)
        self.mapping[ph] = value
        self.reverse[value] = ph
        return ph

    def mask(self, text: str) -> str:
        # regex-gebaseerde PII
        for category, pat in PATTERNS.items():
            for match in pat.findall(text):
                ph = self._placeholder_for(category, match)
                text = text.replace(match, ph)
        # namen (vereenvoudigde NER)
        for name in NAMES:
            if name in text:
                ph = self._placeholder_for("PERSOON", name)
                text = text.replace(name, ph)
        return text

    def unmask(self, text: str) -> str:
        for ph, original in self.mapping.items():
            text = text.replace(ph, original)
        return text


def mock_llm(masked_prompt: str) -> str:
    """Doe alsof de LLM een antwoord geeft dat de placeholders letterlijk teruggeeft."""
    return ("Ik heb uw verzoek verwerkt. Uw contact is <<PERSOON_1>> en ik gebruik "
            "rekening <<IBAN_1>> zoals aangegeven in uw bericht.")


# ----------------------------------------------------------------------------
# 4. Model-vrije PII-detectie: een KLEIN zelf-getraind neuraal netwerk
#    (perceptron) als illustratie dat je geen LLM nodig hebt voor detectie.
# ----------------------------------------------------------------------------
def _token_features(token: str):
    """Minimale, domein-onafhankelijke features voor een token."""
    return [
        1.0,                                   # bias
        float("@" in token),                   # e-mail indicator
        float(any(c.isdigit() for c in token)),  # bevat cijfers
        float(token.istitle()),                # Hoofdletter-start (naam/entiteit)
        float(len(token) > 6),                 # lang token
        float("." in token),                   # punt (domein / afkorting)
    ]


def tiny_ner_demo():
    """
    Mini perceptron (geen LLM) die leert of een token PII is.
    In productie vervang je dit door Presidio / een kleine transformer-NER.
    """
    # Labeled trainingsdata: (token, is_pii)  -- 1 = PII, 0 = geen PII
    train = [
        ("jan@example.com", 1), ("BE68", 1), ("1234", 1), ("Peeters", 1),
        ("Hoi", 0), ("ik", 0), ("ben", 0), ("en", 0), ("mijn", 0),
        ("mail", 0), ("rekening", 0), ("moet", 0), ("worden", 0),
    ]
    weights = [0.0] * 6
    for _ in range(20):                 # epochs
        for token, label in train:
            feat = _token_features(token)
            score = sum(w * f for w, f in zip(weights, feat))
            pred = 1 if score >= 0 else 0
            error = label - pred
            if error != 0:
                for i in range(len(weights)):
                    weights[i] += 0.1 * error * feat[i]

    print("\n[tiny_ner_demo] zelf-getraind perceptron (GEEN LLM):")
    print("  {:16} -> PII?".format("token"))
    for t in ["Alice", "info@bedrijf.be", "90210", "hallo"]:
        feat = _token_features(t)
        score = sum(w * f for w, f in zip(weights, feat))
        pred = "JA" if score >= 0 else "nee"
        print("  {:16} -> {}".format(t, pred))
    print("  (In productie: Presidio / spaCy-NER / kleine transformer,")
    print("   niet dit speelgoed-model.)")


if __name__ == "__main__":
    proxy = AnonymizingProxy()

    # Dezelfde e-mail komt 2x voor -> moet dezelfde placeholder krijgen.
    user_msg = ("Hoi, ik ben Jan Peeters. Mijn mail is jan@example.com en "
                "mijn rekening BE68 1234 5678 9012 moet worden gebruikt. "
                "Nogmaals: jan@example.com is mijn mail.")
    print("ORIGINEEL :", user_msg)

    masked = proxy.mask(user_msg)
    print("GEMASKEERD:", masked)
    print("  -> dit is wat de (cloud-)API ontvangt (§14.2).")
    print("  -> 'jan@example.com' komt 2x voor en krijgt DEZELFDE placeholder")
    print("     (consistentie, §14.2).")

    # de API verwerkt de gemaskeerde prompt
    llm_answer = mock_llm(masked)
    print("API-ANTWOORD (met placeholders):", llm_answer)

    # un-mask het antwoord voor de eindgebruiker
    final = proxy.unmask(llm_answer)
    print("VOOR GEBRUIKER:", final)

    # model-vrije detectie (geen LLM nodig)
    tiny_ner_demo()

    print("\nZie docs/12-data-privacy.md (§14) voor afwegingen: reversibele")
    print("mapping beveiligen, kwaliteitsverlies, en logging zonder ruwe PII.")
