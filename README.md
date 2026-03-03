# ML Research Agent

Ein interaktives ML-Forschungssystem: Du diskutierst mit einem lokalen LLM, das Experimente vorschlägt, im Docker ausführt, Ergebnisse interpretiert — und am Ende ein Paper schreibt.

## Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                      Research Agent                         │
│                                                             │
│  ┌─────────────┐     ┌──────────────────┐                   │
│  │  Researcher │     │  Coder           │                   │
│  │  Agent      │────▶│  Agent           │                   │
│  │  (Diskussion│     │  (Code-Erstellung│                   │
│  │   Strategie)│     │   frischer Ctx)  │                   │
│  └──────┬──────┘     └────────┬─────────┘                   │
│         │                     │                             │
│  ┌──────▼─────────────────────▼─────────┐                   │
│  │         Docker Sandbox               │                   │
│  │         (ML Training, isoliert)      │                   │
│  └──────────────────────────────────────┘                   │
│                                                             │
│  ┌──────────────────┐   ┌───────────────────┐               │
│  │  Telegram / TTY  │   │  Paper Generator  │               │
│  │  (User-Input)    │   │  (LaTeX → PDF)    │               │
│  └──────────────────┘   └───────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Flow

1. **User** gibt ein Forschungsziel vor
2. **Researcher-Agent** diskutiert Hypothesen und Strategie mit dem User
3. Wenn einig: Researcher schlägt Experiment vor (`ACTION: RUN`)
4. User bestätigt → **Coder-Agent** schreibt Python-Code
5. **Docker-Sandbox** führt Code isoliert aus (Image wird beim ersten Mal automatisch gebaut)
6. **Researcher-Agent** interpretiert Ergebnisse, fragt User nach nächsten Schritten
7. Wiederholen bis User `write paper` / `write report` eingibt → **Paper Generator** erzeugt LaTeX + PDF

## Plattform-Unterstützung

Der Agent erkennt das Betriebssystem automatisch und wählt das passende Docker-Image:

| Plattform              | Docker-Image      | PyTorch       |
|------------------------|-------------------|---------------|
| macOS M1 (Apple Silicon) | `ml-sandbox:cpu` | CPU (ARM64)   |
| Linux ohne GPU         | `ml-sandbox:cpu`  | CPU (x86_64)  |
| Linux mit NVIDIA GPU   | `ml-sandbox:gpu`  | CUDA 12.4     |

> Das Docker-Image wird beim ersten Experiment-Start automatisch gebaut — kein manueller `docker build` nötig.

### Voraussetzungen Linux (NVIDIA GPU)

`nvidia-container-toolkit` muss auf dem Host installiert sein:

```bash
# Prüfen ob bereits installiert
nvidia-container-cli --version

# Falls nicht:
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Quickstart

```bash
# 1. Abhängigkeiten installieren (in .venv)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Ollama (oder anderes Backend) starten
ollama serve  # Standard: http://localhost:11434

# 3. Modell laden
ollama pull qwen2.5-coder:14b  # oder llama3, codestral, deepseek-coder, ...

# 4. Agent starten
python main.py --goal "Compare SGD vs Adam optimizer on MNIST"

# Mit GPU (Linux)
python main.py --goal "Compare SGD vs Adam optimizer on MNIST" --gpu

# Bestimmte GPU auswählen (z.B. RTX A6000 auf Index 1)
python main.py --goal "..." --gpu --gpu-device 1
```

## CLI-Argumente

| Argument        | Standard              | Beschreibung                              |
|-----------------|-----------------------|-------------------------------------------|
| `--goal`        | *(Pflicht)*           | Forschungsfrage oder Hypothese            |
| `--model`       | `qwen2.5-coder:14b`   | LLM-Modell überschreiben                  |
| `--backend`     | `ollama`              | LLM-Backend (`ollama\|lmstudio\|vllm`)    |
| `--config`      | `config/settings.yaml`| Pfad zur Konfigurationsdatei              |
| `--gpu`         | `false`               | GPU im Docker aktivieren (nur Linux)      |
| `--gpu-device`  | *(aus config)*        | GPU-Index (`0`=RTX 3070, `1`=A6000)       |
| `--template`    | `plain`               | Paper-Template (`ieee\|neurips\|icml\|plain`) |
| `--log-level`   | `INFO`                | Log-Level                                 |

## Telegram (optional)

Für Benachrichtigungen und Steuerung vom Handy:

1. `@BotFather` → `/newbot` → Token kopieren
2. Bot anschreiben → `@userinfobot` → Chat-ID kopieren
3. In `config/settings.yaml` eintragen:
   ```yaml
   telegram:
     bot_token: "123456:ABC..."
     chat_id: "987654321"
   ```

### Bot-Befehle

| Befehl         | Funktion                              |
|----------------|---------------------------------------|
| `/save`        | Aktuellen Checkpoint speichern        |
| `/load latest` | Neuesten Checkpoint laden             |
| `/load <id>`   | Bestimmten Checkpoint laden           |
| `/checkpoints` | Liste aller gespeicherten Checkpoints |
| `/status`      | Aktuellen Stand anzeigen              |

## Konfiguration

Alle Optionen in `config/settings.yaml`:

| Abschnitt    | Wichtige Felder                                    |
|--------------|----------------------------------------------------|
| `llm`        | Backend, Modell, Temperatur, Retries               |
| `docker`     | `image_cpu`, `image_gpu`, RAM/CPU-Limit, GPU, Timeout |
| `experiment` | Max. Iterationen, Checkpoint-Verzeichnis           |
| `paper`      | LaTeX-Template, Compiler                           |
| `telegram`   | Bot-Token, Chat-ID, Reply-Timeout                  |

## Unterstützte LLM-Backends

| Backend   | Hinweis                      |
|-----------|------------------------------|
| Ollama    | Empfohlen für lokale Nutzung |
| LM Studio | OpenAI-kompatibler Endpunkt  |
| llama.cpp | Direkt GGUF-Modelle laden    |
| vLLM      | Hoher Durchsatz              |

## Projektstruktur

```
ml-research-agent/
├── main.py                      # Einstiegspunkt
├── requirements.txt
├── config/
│   └── settings.yaml            # Alle Konfigurationsoptionen
├── core/
│   ├── agent.py                 # Interaktiver Research-Loop (Dual-Agent)
│   ├── state.py                 # ExperimentState (History, Metriken)
│   └── checkpoint.py            # Session speichern & laden
├── llm/
│   ├── client.py                # LLM-API-Abstraktion (Ollama, OpenAI-compat.)
│   ├── prompts.py               # System- & Task-Prompts (Researcher, Coder)
│   └── parser.py                # Code/Actions aus LLM-Output extrahieren
├── sandbox/
│   ├── sandbox.py               # Docker-Container starten, Code ausführen
│   ├── Dockerfile.sandbox       # ML-Python-Umgebung (CPU / macOS M1)
│   └── Dockerfile.sandbox-gpu   # ML-Python-Umgebung (Linux CUDA 12.4)
├── paper/
│   ├── generator.py             # LaTeX kompilieren → PDF
│   └── templates/               # LaTeX-Vorlagen (ieee, neurips, icml, arxiv, plain)
├── telegram/
│   └── notifier.py              # Telegram-Bot (senden & empfangen)
├── experiments/                 # Gespeicherte Experiment-Logs (JSON)
│   └── checkpoints/             # Session-Checkpoints
├── papers/                      # Generierte PDFs
├── logs/                        # Agent-Logs
└── tests/
    └── test_core.py
```

## Ausgabe-Dateien

Jede Session bekommt eine eindeutige ID (Timestamp) und einen eigenen Unterordner:

```
experiments/
  20260304_143022/                  ← Session 1
    experiment.json                 ← vollständige History + Config-Snapshot
    iter_1/
      experiment.py                 ← generierter Code
      plot_loss.png                 ← Plots aus /results/
    iter_2/
      experiment.py
  20260304_161500/                  ← Session 2 (kein Konflikt)
    experiment.json
    iter_1/
      experiment.py
  checkpoints/
    ckpt_20260304_142500.json       ← Auto-Save nach jedem Experiment

papers/
  ml_research_compare_sgd_vs_adam.pdf
  ml_research_compare_sgd_vs_adam.tex   ← LaTeX-Quelle, immer gespeichert
```

Die `experiment.json` enthält pro Iteration: Hypothesis, Code, Output, Metriken,
Artefakt-Pfade — sowie einen `config_snapshot` mit LLM-Modell, Docker-Image
und allen relevanten Einstellungen für Reproduzierbarkeit.
