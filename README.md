# Interactive Camera System

## Teilnehmer

* Keno Oelrichs García
* Max Stargardt

## Ziel des Projekts

Ein Kamerasystem bestehend aus zwei Kameras nimmt zum einen einen Sprecher und zum anderen eine Topdownsicht eines
Papiers o.Ä. auf. Sobald der Sprecher eine *aktivierende Geste* der zweiten Kamera zeigt, folgt diese dem gezeigten
relevanten Bereich und zeigt diesen als *Picture in Picture* im Bild der ersten Kamera, der des Sprechers also.

## Ausführen des Projektes

### Vor Ausführung und Installation benötigt

- Python3.8
- git

```bash
# Lade das GIt-Repository herunter. Hierfür sind möglicherweise weitere Anmeldedaten nötig.
git clone https://github.com/uol-mediaprocessing-202021/medienverarbeitung-e-interactive-camera-system.git

# Navigiere in das heruntergeladene Verzeichnis
cd medienverarbeitung-e-interactive-camera-system/

# Erstelle eine virtuelle Python-Umgebung
python3 -m venv venv

# Aktiviere die zuvor erstellte Python-Umgebung
source venv/bin/activate

# Aktualisiere PIP
pip install --upgrade pip

# Installiere die benötigten Bibliotheken
pip3 install -r requirements.txt

# Navigiere in das Programmverzeichnis
cd main

# Starte die Software
python3 __init__.py
```



