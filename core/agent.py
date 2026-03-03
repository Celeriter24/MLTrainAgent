"""
Agent — interaktiver Research-Loop mit Checkpoint-System.

Flow:
  1. Start: vorhandene Checkpoints anbieten
  2. Researcher begrüßt User, diskutiert das Ziel
  3. User ist federführend — Researcher schlägt Experimente vor (ACTION: RUN)
  4. User bestätigt → Coder → Docker → Auto-Save Checkpoint → Researcher interpretiert
  5. Wiederholen bis User "fertig" sagt (ACTION: DONE) → Paper
  6. Timeout → Checkpoint speichern + sauber beenden

Telegram-Befehle (jederzeit eingebbar):
  /save              → aktuellen Checkpoint speichern
  /load <id|latest>  → Checkpoint laden und Session fortsetzen
  /checkpoints       → Liste aller Checkpoints
  /status            → aktueller Stand (Experimente, Metriken)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from llm.client import LLMClient
from llm.prompts import (
    SYSTEM_PROMPT,
    build_paper_prompt,
    RESEARCHER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    build_coder_prompt,
)
from llm.parser import parse_llm_response, extract_metrics, extract_error
from docker.sandbox import DockerSandbox
from paper.generator import PaperGenerator
from core.state import ExperimentState
from core.checkpoint import CheckpointManager
from telegram.notifier import TelegramNotifier

logger = logging.getLogger(__name__)

_CONFIRM_KEYWORDS = {"ja", "yes", "y", "start", "run", "starten", "go", "ok", "okay"}


@dataclass
class _SessionState:
    """Vollständiger Zustand einer laufenden Research-Session."""
    exp_state: ExperimentState
    researcher_messages: list = field(default_factory=list)
    history_summaries: list = field(default_factory=list)
    experiment_count: int = 0


class ResearchAgent:
    def __init__(self, config: dict):
        self.config = config
        self.llm = LLMClient(config)
        self.sandbox = DockerSandbox(config)
        self.paper_gen = PaperGenerator(config)
        self.notifier = TelegramNotifier(config)
        self.checkpoint_mgr = CheckpointManager(config)
        self.max_iter = config["experiment"].get("max_iterations", 10)
        self.save_dir = Path(config["experiment"].get("save_dir", "experiments"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reply_timeout = config.get("telegram", {}).get("reply_timeout", 3600)

    # ── Haupt-Loop ────────────────────────────────────────────────────────────

    def run(self, goal: str) -> Path | None:
        """
        Interaktiver Research-Loop.
        Returns Pfad zur PDF, oder None wenn Session durch Timeout/Abbruch beendet wurde.
        Der finally-Block stellt sicher dass immer etwas gespeichert wird.
        """
        session = _SessionState(exp_state=ExperimentState(goal=goal))
        completion_status = "interrupted"
        stopped_reason: str | None = "Session wurde nicht abgeschlossen"
        paper_path: Path | None = None

        logger.info(f"🔬 Research goal: {goal}")
        logger.info(f"🐳 Docker image: {self.config['docker']['image']}")

        try:
            if not self.sandbox.image_exists():
                logger.info("Docker image not found — building it now...")
                self.sandbox.build_image()

            # ── Startup: Checkpoint anbieten ──────────────────────────────
            latest = self.checkpoint_mgr.load_latest()
            if latest:
                resume_msg = (
                    f"Vorhandener Checkpoint gefunden:\n"
                    f"• ID: `{latest['id']}`\n"
                    f"• Ziel: {latest['goal'][:60]}\n"
                    f"• Experimente: {latest['experiment_count']}\n"
                    f"• Gespeichert: {latest['saved_at'][:16]}\n\n"
                    f"Weitermachen? *ja* / *nein*\n"
                    f"(oder `/load <id>` für einen anderen Checkpoint, `/checkpoints` für alle)"
                )
                choice = self._await_user(resume_msg, session, timeout=120)
                if choice is None:
                    stopped_reason = "Timeout beim Startup"
                    return None
                if self._user_confirmed(choice) or choice.strip().lower().startswith("/load"):
                    if choice.strip().lower().startswith("/load"):
                        self._handle_command(choice, session)
                    else:
                        self._apply_checkpoint(latest, session)
                    self.notifier.send(
                        f"Checkpoint geladen — wir machen weiter ab Experiment {session.experiment_count}."
                    )
                else:
                    self.notifier.send("Okay, wir starten von vorne.")

            # ── Eröffnung (nur bei neuer Session) ─────────────────────────
            if not session.researcher_messages:
                session.researcher_messages = [{"role": "system", "content": RESEARCHER_SYSTEM_PROMPT}]
                opening_msg = (
                    f"Research goal: {goal}\n\n"
                    f"Introduce yourself briefly, summarize your understanding of the goal, "
                    f"and ask the user one focused question to start the discussion."
                )
                session.researcher_messages.append({"role": "user", "content": opening_msg})
                logger.info("💭 Researcher: Eröffnung...")
                opening_raw = self.llm.chat(session.researcher_messages)
                session.researcher_messages.append({"role": "assistant", "content": opening_raw})
                user_reply = self._await_user(opening_raw, session)
                if user_reply is None:
                    stopped_reason = "Timeout nach Eröffnung"
                    return None
            else:
                user_reply = (
                    f"[Session wird aus Checkpoint fortgesetzt. "
                    f"Bisherige Experimente: {session.experiment_count}. "
                    f"Fasse kurz zusammen wo wir stehen und frage wie wir weitermachen.]"
                )

            # ── Haupt-Diskussions-Loop ─────────────────────────────────────
            while session.experiment_count < self.max_iter:

                session.researcher_messages.append({"role": "user", "content": user_reply})
                logger.info("💭 Researcher: denkt nach...")
                researcher_raw = self.llm.chat(session.researcher_messages)
                session.researcher_messages.append({"role": "assistant", "content": researcher_raw})

                researcher_parsed = parse_llm_response(researcher_raw)

                if researcher_parsed.action == "DONE":
                    logger.info("✅ DONE — Paper wird generiert")
                    self.notifier.send("Alright, ich generiere jetzt das Paper. Einen Moment...")
                    break

                if researcher_parsed.action not in (None, "RUN", "DONE"):
                    logger.warning(
                        f"Unbekannte Researcher-Action '{researcher_parsed.action}' "
                        f"— wird als Diskussions-Antwort behandelt"
                    )

                if researcher_parsed.action == "RUN" and researcher_parsed.experiment_spec:
                    confirm_msg = (
                        f"{researcher_raw}\n\n"
                        f"▶ Experiment starten? (*ja* / *nein* / Änderungen beschreiben)"
                    )
                    user_reply = self._await_user(confirm_msg, session)
                    if user_reply is None:
                        stopped_reason = "Timeout während Experiment-Bestätigung"
                        return None

                    if self._user_confirmed(user_reply):
                        session.experiment_count += 1
                        exec_result, coder_code = self._run_experiment(
                            researcher_parsed.experiment_spec, session.experiment_count
                        )

                        if exec_result is not None:
                            formatted_output = self._format_docker_output(exec_result)
                            metrics = extract_metrics(exec_result.output)
                            success = (exec_result.exit_code == 0)
                            artifact_paths = self._save_artifacts(exec_result, session.experiment_count)

                            session.exp_state.add_iteration(
                                code=coder_code,
                                output=exec_result.output,
                                metrics=metrics,
                                hypothesis=researcher_parsed.hypothesis,
                                artifacts=artifact_paths,
                                success=success,
                            )
                            session.history_summaries.append({
                                "iteration": session.experiment_count,
                                "hypothesis": researcher_parsed.hypothesis,
                                "metrics": metrics,
                                "success": success,
                            })

                            ckpt_id = self._save_checkpoint(session)
                            logger.info(f"Auto-Save: {ckpt_id}")

                            interp_prompt = (
                                f"The experiment just finished. Results:\n\n"
                                f"{formatted_output}\n\n"
                                f"Interpret these results for the user and ask what direction "
                                f"they want to take next."
                            )
                            session.researcher_messages.append({"role": "user", "content": interp_prompt})
                            logger.info("💭 Researcher: interpretiert Ergebnisse...")
                            interp_raw = self.llm.chat(session.researcher_messages)
                            session.researcher_messages.append({"role": "assistant", "content": interp_raw})

                            user_reply = self._await_user(interp_raw, session)
                            if user_reply is None:
                                stopped_reason = "Timeout nach Experiment-Interpretation"
                                return None
                        else:
                            user_reply = self._await_user(
                                "Das Experiment ist leider nach zwei Versuchen fehlgeschlagen. "
                                "Was sollen wir als nächstes tun?",
                                session,
                            )
                            if user_reply is None:
                                stopped_reason = "Timeout nach fehlgeschlagenem Experiment"
                                return None
                    else:
                        user_reply = f"[User möchte das Experiment nicht starten]: {user_reply}"

                else:
                    user_reply = self._await_user(researcher_raw, session)
                    if user_reply is None:
                        stopped_reason = "Timeout während Diskussion"
                        return None

            # ── Paper generieren ───────────────────────────────────────────
            logger.info("\n📝 Generiere Paper...")
            paper_path = self._generate_paper(session.exp_state.goal, session.exp_state)
            self.notifier.send(f"Paper fertig gespeichert unter: {paper_path}")
            completion_status = "done"
            stopped_reason = None
            return paper_path

        except KeyboardInterrupt:
            completion_status = "interrupted"
            stopped_reason = "Vom User abgebrochen (Strg+C)"
            logger.warning(stopped_reason)
            self.notifier.send("Session abgebrochen.")
            return None

        except Exception as e:
            completion_status = "failed"
            stopped_reason = f"{type(e).__name__}: {e}"
            logger.error(f"Unerwarteter Fehler: {stopped_reason}", exc_info=True)
            self.notifier.send(f"Fehler aufgetreten: {stopped_reason}")
            raise

        finally:
            # Immer speichern — egal wie die Session endete
            if session.exp_state.history or stopped_reason:
                session.exp_state.status = completion_status
                self._save_experiment(
                    session.exp_state,
                    stopped_reason=stopped_reason,
                )

    # ── Kommunikation ─────────────────────────────────────────────────────────

    def _await_user(
        self,
        message: str,
        session: _SessionState,
        timeout: int | None = None,
    ) -> str | None:
        """
        Sendet eine Nachricht und wartet auf Antwort des Users.
        Befehle (/save, /load, ...) werden transparent abgefangen und beantwortet.
        Returns None wenn Timeout erreicht wurde (Checkpoint wurde bereits gespeichert).
        """
        if timeout is None:
            timeout = self.reply_timeout

        self.notifier.send(message)

        while True:
            reply = self.notifier.wait_for_reply(timeout)

            if not reply:
                # Timeout — Checkpoint speichern und sauber beenden
                ckpt_id = self._save_checkpoint(session)
                self.notifier.send(
                    f"Keine Antwort nach {timeout // 60} Minuten.\n"
                    f"Checkpoint gespeichert: `{ckpt_id}`\n"
                    f"Starte das System neu und antworte mit *ja* um weiterzumachen."
                )
                logger.info(f"Timeout — Session gespeichert als {ckpt_id}, beende...")
                return None

            if reply.startswith("/"):
                response = self._handle_command(reply, session)
                self.notifier.send(response)
                continue  # Weiter auf echte Antwort warten

            return reply

    def _handle_command(self, cmd: str, session: _SessionState) -> str:
        """Verarbeitet einen Telegram-Befehl. Mutiert session bei /load in-place."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/save":
            ckpt_id = self._save_checkpoint(session)
            return f"Checkpoint gespeichert: `{ckpt_id}`"

        if command == "/load":
            ckpt_id = arg or "latest"
            data = self.checkpoint_mgr.load(ckpt_id)
            if data:
                self._apply_checkpoint(data, session)
                return (
                    f"Checkpoint geladen: `{data['id']}`\n"
                    f"Ziel: {data['goal'][:60]}\n"
                    f"Experimente: {data['experiment_count']}\n"
                    f"Weiter mit deiner nächsten Nachricht."
                )
            return f"Checkpoint `{ckpt_id}` nicht gefunden. Nutze /checkpoints für eine Liste."

        if command == "/checkpoints":
            checkpoints = self.checkpoint_mgr.list_checkpoints()
            if not checkpoints:
                return "Keine Checkpoints gefunden."
            lines = ["Gespeicherte Checkpoints (neueste zuerst):\n"]
            for c in checkpoints:
                lines.append(
                    f"• `{c['id']}` — {c['goal']} "
                    f"({c['experiments']} Exp., {c['saved_at']})"
                )
            lines.append("\nLaden mit: /load <id> oder /load latest")
            return "\n".join(lines)

        if command == "/status":
            s = session.exp_state
            metrics_str = (
                json.dumps(s.latest_metrics, indent=2)
                if s.latest_metrics
                else "noch keine Metriken"
            )
            return (
                f"*Status der laufenden Session*\n"
                f"• Ziel: {s.goal[:60]}\n"
                f"• Experimente durchgeführt: {session.experiment_count}\n"
                f"• Erfolgreiche Runs: {s.successful_runs}\n"
                f"• Beste Metriken:\n```\n{metrics_str}\n```"
            )

        return (
            f"Unbekannter Befehl: `{command}`\n"
            f"Verfügbar: /save · /load <id|latest> · /checkpoints · /status"
        )

    # ── Checkpoint-Hilfsmethoden ──────────────────────────────────────────────

    def _save_checkpoint(self, session: _SessionState) -> str:
        return self.checkpoint_mgr.save(
            goal=session.exp_state.goal,
            researcher_messages=session.researcher_messages,
            history_summaries=session.history_summaries,
            exp_state=session.exp_state,
            experiment_count=session.experiment_count,
        )

    def _apply_checkpoint(self, data: dict, session: _SessionState):
        """Stellt session in-place aus Checkpoint-Daten wieder her."""
        session.researcher_messages = data["researcher_messages"]
        session.history_summaries = data["history_summaries"]
        session.experiment_count = data["experiment_count"]
        session.exp_state = ExperimentState.from_dict(data["exp_state"])

    # ── Experiment-Ausführung ─────────────────────────────────────────────────

    def _run_experiment(self, experiment_spec: str, iteration: int):
        """
        Coder schreibt Code, Docker führt aus (inkl. einem Debug-Retry).
        Returns (ExecutionResult, code_str) oder (None, "") bei komplettem Fehler.
        """
        logger.info("🖊️  Coder: schreibt Code...")
        code_messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": build_coder_prompt(experiment_spec)},
        ]
        coder_raw = self.llm.chat(code_messages)
        coder_parsed = parse_llm_response(coder_raw)

        if not coder_parsed.code:
            logger.warning("⚠️  Coder hat keinen Code produziert")
            return None, ""

        logger.info("🐳 Experiment läuft in Docker...")
        exec_result = self.sandbox.run_code(coder_parsed.code)
        logger.info(f"⏱  {exec_result.duration:.1f}s | Exit: {exec_result.exit_code}")

        if exec_result.exit_code != 0:
            error = extract_error(exec_result.output, exec_result.exit_code)
            logger.warning(f"❌ Fehler — Debug-Retry:\n{error}")
            code_messages.append({"role": "assistant", "content": coder_raw})
            code_messages.append({
                "role": "user",
                "content": build_coder_prompt(experiment_spec, last_error=error),
            })
            retry_raw = self.llm.chat(code_messages)
            retry_parsed = parse_llm_response(retry_raw)
            if retry_parsed.code:
                logger.info("🔁 Retry: Code wird nochmal ausgeführt...")
                exec_result = self.sandbox.run_code(retry_parsed.code)
                coder_parsed = retry_parsed
                logger.info(f"⏱  Retry: {exec_result.duration:.1f}s | Exit: {exec_result.exit_code}")
            else:
                logger.warning("⚠️  Retry hat keinen Code produziert")

        return exec_result, coder_parsed.code

    def _save_artifacts(self, exec_result, iteration: int) -> dict:
        artifact_paths = {}
        if exec_result.files:
            artifact_dir = self.save_dir / f"iter_{iteration}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for fname, data in exec_result.files.items():
                fpath = artifact_dir / fname
                fpath.write_bytes(data)
                artifact_paths[fname] = str(fpath)
            logger.info(f"💾 {len(artifact_paths)} Artifacts gespeichert")
        return artifact_paths

    def _format_docker_output(self, exec_result) -> str:
        """Kompakte, strukturierte Ansicht des Docker-Outputs für den Researcher."""
        stdout_tail = exec_result.stdout[-500:] if exec_result.stdout else "(empty)"
        metrics = extract_metrics(exec_result.output)
        metrics_str = json.dumps(metrics) if metrics else "none found"
        error = extract_error(exec_result.output, exec_result.exit_code)
        error_str = error if error else "None"
        return (
            f"--- STDOUT (last 500 chars) ---\n{stdout_tail}\n"
            f"--- METRICS ---\n{metrics_str}\n"
            f"--- ERRORS ---\n{error_str}"
        )

    @staticmethod
    def _user_confirmed(text: str) -> bool:
        first_word = text.strip().lower().split()[0] if text.strip() else ""
        return first_word in _CONFIRM_KEYWORDS

    # ── Paper & Speichern ─────────────────────────────────────────────────────

    def _generate_paper(self, goal: str, state: ExperimentState) -> Path:
        paper_prompt = build_paper_prompt(
            goal=goal,
            experiment_log=state.history,
            final_results=state.latest_metrics,
            template=self.config["paper"].get("template", "ieee"),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paper_prompt},
        ]
        logger.info("💭 Generiere LaTeX Paper via LLM...")
        raw = self.llm.chat(messages)
        parsed = parse_llm_response(raw)
        if not parsed.latex:
            logger.warning("LLM hat keinen latex PAPER Block produziert — nutze Raw-Output")
            parsed.latex = raw
        return self.paper_gen.compile(
            latex=parsed.latex,
            title=f"ML Research: {goal[:60]}",
            artifacts={k: v for h in state.history for k, v in h.get("artifacts", {}).items()},
        )

    def _save_experiment(
        self,
        state: ExperimentState,
        stopped_reason: str | None = None,
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.save_dir / f"experiment_{ts}.json"
        data = state.to_dict()
        data["config_snapshot"] = self._safe_config_snapshot(self.config)
        if stopped_reason:
            data["stopped_reason"] = stopped_reason
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"💾 Experiment gespeichert: {out}")

    @staticmethod
    def _safe_config_snapshot(config: dict) -> dict:
        """Config-Snapshot ohne sensible Felder (Tokens etc.)."""
        return {
            "llm": {
                "backend":     config["llm"]["backend"],
                "model":       config["llm"]["model"],
                "temperature": config["llm"].get("temperature"),
                "max_tokens":  config["llm"].get("max_tokens"),
            },
            "docker": {
                "image":        config["docker"]["image"],
                "memory_limit": config["docker"].get("memory_limit"),
                "cpu_limit":    config["docker"].get("cpu_limit"),
                "gpu":          config["docker"].get("gpu"),
                "timeout":      config["docker"].get("timeout"),
            },
            "paper": {
                "template": config["paper"].get("template"),
                "compiler": config["paper"].get("compiler"),
            },
            "experiment": {
                "max_iterations": config["experiment"].get("max_iterations"),
            },
        }
