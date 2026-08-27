"""
Om dit met md2pdf-mermaid te doen, bundel je alle Markdown-bestanden én de Python-broncode eerst in het geheugen tot één grote Markdown-string. De Python-code voeg je toe door het in een syntax-highlighting codeblok (```python) te wikkelen. Daarna stuur je deze gebundelde string in één keer naar de converter. [1, 2]

Hier is het complete Python-script dat dit automatisch voor je regelt:
## 1. Installeer de vereisten (indien nog niet gedaan)
Zorg dat je naast de bibliotheek ook de benodigde browser-engine (Playwright) installeert, die op de achtergrond de Mermaid-diagrammen tekent: [2]

pip install md2pdf-mermaid
playwright install chromium
"""
## 2. Het Python-script (bundel_naar_pdf.py)

import os
from md2pdf_mermaid import convert_markdown_to_pdf

def bundle_project_to_pdf(output_pdf_name="project_bundel.pdf"):
    # 1. Start de verzameling van alle inhoud
    gebundelde_markdown = "# Project Documentatie & Broncode\n\n"

    # [TOC] voegt automatisch een inhoudsopgave toe (ondersteund door md2pdf-mermaid)
    gebundelde_markdown += "[TOC]\n\n"
    gebundelde_markdown += "<div style='page-break-after: always;'></div>\n\n"

    # --- DEEL 1: ALLES VAN .MD BESTANDEN BUNDELEN ---
    # Je kunt hier een specifieke volgorde opgeven, of een map scannen
    md_bestanden = ["intro.md", "architectuur.md", "gebruik.md"]

    print("Bezig met verwerken van Markdown bestanden...")
    for md_file in md_bestanden:
        if os.path.exists(md_file):
            with open(md_file, "r", encoding="utf-8") as f:
                # Voeg de inhoud toe + een pagina-einde voor het volgende hoofdstuk
                gebundelde_markdown += f"## Documentatie: {md_file}\n\n"
                gebundelde_markdown += f.read()
                gebundelde_markdown += "\n\n<div style='page-break-after: always;'></div>\n\n"
        else:
            print(f"Waarschuwing: {md_file} niet gevonden. Wordt overgeslagen.")

    # --- DEEL 2: PYTHON SOURCE CODE BUNDELEN ---
    # Geef hier de Python bestanden op die in de PDF moeten komen
    python_bestanden = ["main.py", "utils.py"]

    print("Bezig met verwerken van Python broncode...")
    for py_file in python_bestanden:
        if os.path.exists(py_file):
            with open(py_file, "r", encoding="utf-8") as f:
                code_inhoud = f.read()

                # Wikkel de code in een markdown codeblok met syntax-highlighting
                gebundelde_markdown += f"## Broncode: {py_file}\n\n"
                gebundelde_markdown += "```python\n"
                gebundelde_markdown += code_inhoud
                gebundelde_markdown += "\n```\n\n"
                gebundelde_markdown += "<div style='page-break-after: always;'></div>\n\n"
        else:
            print(f"Waarschuwing: {py_file} niet gevonden. Wordt overgeslagen.")

    # --- DEEL 3: CONVERTEREN NAAR PDF ---
    print(f"PDF genereren via md2pdf-mermaid... (dit kan even duren ivm Mermaid-diagrammen)")
    convert_markdown_to_pdf(
        markdown_text=gebundelde_markdown,
        output_path=output_pdf_name,
        title="Project Rapport",
        enable_mermaid=True
    )
    print(f"Klaar! Je PDF is opgeslagen als: {output_pdf_name}")
if __name__ == "__main__":
    bundle_project_to_pdf("Mijn_Compleet_Project.pdf")

"""
Waarom deze methode perfect werkt:
   1. Inhoudsopgave: Dankzij de [TOC] tag bovenaan maakt md2pdf-mermaid automatisch een klikbare inhoudsopgave op basis van je # en ## koppen. [2]
   2. Prachtige Code: Omdat md2pdf-mermaid ingebouwde syntax-highlighting heeft, wordt je Python-code niet als saaie tekst getoond, maar krijgt het mooie programmeer-kleuren (zoals in een IDE). [2]
   3. Mermaid blijft werken: Alle Mermaid-diagrammen die in je .md bestanden stonden, worden door de compiler netjes omgezet naar strakke afbeeldingen in de uiteindelijke PDF. [2]

Als je wilt dat het script automatisch álle .md en .py bestanden uit je mappen plukt zonder dat je ze handmatig in een lijst hoeft te zetten=> breid het script uit met een automatische map-scanner!
[1] [https://github.com](https://github.com/rbutinar/md2pdf-mermaid/blob/master/GETTING_STARTED.md)
[2] [https://pypi.org](https://pypi.org/project/md2pdf-mermaid/)
"""
