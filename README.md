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
5. **Docker-Sandbox** führt Code isoliert aus
6. **Researcher-Agent** interpretiert Ergebnisse, fragt User nach nächsten Schritten
7. Wiederholen bis User fertig ist → **Paper Generator** erzeugt LaTeX-PDF

## Quickstart

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Ollama (oder anderes Backend) starten
ollama serve  # Standard: http://localhost:11434

# 3. Modell laden
ollama pull llama3  # oder codestral, deepseek-coder, qwen2.5-coder, ...

# 4. ML-Sandbox Docker-Image bauen
docker build -f docker/Dockerfile.sandbox -t ml-sandbox .

# 5. Agent starten
python main.py --goal "Compare SGD vs Adam optimizer on MNIST"
```

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

| Abschnitt    | Wichtige Felder                          |
|--------------|------------------------------------------|
| `llm`        | Backend, Modell, Temperatur, Retries     |
| `docker`     | Image, RAM/CPU-Limit, GPU, Timeout       |
| `experiment` | Max. Iterationen, Checkpoint-Verzeichnis |
| `paper`      | LaTeX-Template, Compiler                 |
| `telegram`   | Bot-Token, Chat-ID, Reply-Timeout        |

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
├── docker/
│   ├── sandbox.py               # Docker-Container starten, Code ausführen
│   ├── Dockerfile.sandbox       # ML-Python-Umgebung (CPU)
│   └── Dockerfile.sandbox-gpu   # ML-Python-Umgebung (CUDA)
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

Jede abgeschlossene (oder abgebrochene) Session speichert:

```
experiments/
  experiment_20240302_143022.json   ← vollständige History + Config-Snapshot
  checkpoints/
    ckpt_20240302_142500.json       ← Auto-Save nach jedem Experiment
papers/
  ml_research_compare_sgd_vs_adam.pdf
```

Die Experiment-JSON enthält pro Iteration: Hypothesis, Code, Output, Metriken,
Artefakt-Pfade — sowie einen `config_snapshot` mit LLM-Modell, Docker-Image
und allen relevanten Einstellungen für Reproduzierbarkeit.
