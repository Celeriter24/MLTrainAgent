"""
ExperimentState — tracks the full lifecycle of a research session.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExperimentState:
    goal: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    history: list[dict] = field(default_factory=list)
    latest_metrics: dict = field(default_factory=dict)
    status: str = "running"   # running | done | failed

    def add_iteration(
        self,
        code: str,
        output: str,
        metrics: dict,
        hypothesis: str | None = None,
        artifacts: dict[str, str] | None = None,
        success: bool = True,
    ):
        entry = {
            "iteration": len(self.history) + 1,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "code": code,
            "output": output,
            "metrics": metrics,
            "artifacts": artifacts or {},
            "success": success,
        }
        self.history.append(entry)

        # Letzte erfolgreiche Metriken merken (für Paper-Zusammenfassung)
        if success and metrics:
            self.latest_metrics.update(metrics)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentState":
        obj = cls(goal=data["goal"])
        obj.created_at = data.get("created_at", obj.created_at)
        obj.status = data.get("status", "running")
        obj.latest_metrics = data.get("latest_metrics", {})
        obj.history = data.get("history", [])
        return obj

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "created_at": self.created_at,
            "status": self.status,
            "iterations": len(self.history),
            "latest_metrics": self.latest_metrics,
            "history": self.history,
        }

    @property
    def last_output(self) -> str:
        if not self.history:
            return ""
        return self.history[-1].get("output", "")

    @property
    def successful_runs(self) -> int:
        return sum(1 for h in self.history if h.get("success"))
