"""
CheckpointManager — speichert und lädt den vollständigen Session-Zustand.

Checkpoint-Format (JSON):
  {
    "id":                  "ckpt_20240302_143022",
    "goal":                "Compare SGD vs Adam on MNIST",
    "saved_at":            "2024-03-02T14:30:22",
    "experiment_count":    3,
    "researcher_messages": [...],   # volle Gesprächshistorie
    "history_summaries":   [...],   # kompakte Metriken-Summaries
    "exp_state":           {...}    # ExperimentState.to_dict()
  }
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, config: dict):
        ckpt_dir = config["experiment"].get("checkpoint_dir", "experiments/checkpoints")
        self.checkpoint_dir = Path(ckpt_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        goal: str,
        researcher_messages: list,
        history_summaries: list,
        exp_state,          # ExperimentState
        experiment_count: int,
    ) -> str:
        """Speichert den aktuellen Session-Zustand. Gibt die Checkpoint-ID zurück."""
        ckpt_id = f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data = {
            "id": ckpt_id,
            "goal": goal,
            "saved_at": datetime.now().isoformat(),
            "experiment_count": experiment_count,
            "researcher_messages": researcher_messages,
            "history_summaries": history_summaries,
            "exp_state": exp_state.to_dict(),
        }
        path = self.checkpoint_dir / f"{ckpt_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Checkpoint gespeichert: {ckpt_id}")
        return ckpt_id

    def load(self, checkpoint_id: str) -> dict | None:
        """
        Lädt einen Checkpoint anhand der ID.
        Akzeptiert auch 'latest' oder einen Teilstring der ID.
        """
        if checkpoint_id.lower() == "latest":
            return self.load_latest()

        # Exakter Match
        path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        # Partieller Match (z.B. nur Datum/Zeit-Suffix)
        matches = sorted(
            self.checkpoint_dir.glob(f"*{checkpoint_id}*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return json.loads(matches[0].read_text(encoding="utf-8"))

        logger.warning(f"Checkpoint nicht gefunden: {checkpoint_id}")
        return None

    def load_latest(self) -> dict | None:
        """Lädt den neuesten Checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not checkpoints:
            return None
        return json.loads(checkpoints[0].read_text(encoding="utf-8"))

    def list_checkpoints(self) -> list[dict]:
        """Gibt die letzten 10 Checkpoints als kompakte Liste zurück."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:10]
        result = []
        for p in checkpoints:
            data = json.loads(p.read_text(encoding="utf-8"))
            result.append({
                "id": data["id"],
                "goal": data["goal"][:50],
                "saved_at": data["saved_at"][:16],
                "experiments": data["experiment_count"],
            })
        return result
