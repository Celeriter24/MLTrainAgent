"""
TelegramNotifier — sendet Nachrichten an den User und wartet auf Antworten.

Wenn bot_token und chat_id konfiguriert sind, läuft die Kommunikation
bidirektional via Telegram. Andernfalls automatischer Fallback auf Terminal.

Setup:
  1. @BotFather → /newbot → Token kopieren
  2. Bot anschreiben → @userinfobot → Chat-ID kopieren
  3. Beides in config/settings.yaml unter telegram: eintragen
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

# Telegram API Nachrichtenlimit
_MAX_MSG_LEN = 4096


class TelegramNotifier:
    def __init__(self, config: dict):
        tg = config.get("telegram", {})
        self.token = tg.get("bot_token", "").strip()
        self.chat_id = str(tg.get("chat_id", "")).strip()
        self.enabled = bool(self.token and self.chat_id)
        self._last_update_id: int | None = None

        if self.enabled:
            self._api = f"https://api.telegram.org/bot{self.token}"
            logger.info(f"Telegram aktiviert (chat_id={self.chat_id})")
        else:
            logger.info("Telegram nicht konfiguriert — Terminal-Modus aktiv")

    # ── Senden ───────────────────────────────────────────────────────────────

    def send(self, text: str):
        """Sendet eine Nachricht. Teilt bei >4096 Zeichen automatisch auf."""
        if self.enabled:
            self._send_telegram(text)
        else:
            self._print_terminal(text)

    def _send_telegram(self, text: str):
        # Aufteilen wenn zu lang
        chunks = [text[i:i + _MAX_MSG_LEN] for i in range(0, len(text), _MAX_MSG_LEN)]
        for chunk in chunks:
            try:
                resp = requests.post(
                    f"{self._api}/sendMessage",
                    json={"chat_id": self.chat_id, "text": chunk, "parse_mode": "Markdown"},
                    timeout=10,
                )
                if not resp.ok:
                    # Markdown-Fehler? Retry ohne Formatierung
                    requests.post(
                        f"{self._api}/sendMessage",
                        json={"chat_id": self.chat_id, "text": chunk},
                        timeout=10,
                    )
            except Exception as e:
                logger.warning(f"Telegram send fehlgeschlagen: {e} — gebe auf Terminal aus")
                self._print_terminal(chunk)

    @staticmethod
    def _print_terminal(text: str):
        print(f"\n{'─'*60}\n{text}\n{'─'*60}")

    # ── Empfangen ────────────────────────────────────────────────────────────

    def wait_for_reply(self, timeout: int = 3600) -> str:
        """
        Wartet auf eine neue Nachricht vom User.
        - Telegram: Long-Polling mit /getUpdates
        - Terminal: input()
        Returns den Nachrichtentext.
        """
        if self.enabled:
            return self._poll_telegram(timeout)
        return self._read_terminal()

    def _poll_telegram(self, timeout: int) -> str:
        deadline = time.time() + timeout
        poll_timeout = 30  # Telegram long-poll Fenster in Sekunden

        # Beim ersten Aufruf: aktuelle Update-ID ermitteln um alte Msgs zu ignorieren
        if self._last_update_id is None:
            self._last_update_id = self._get_current_update_id()

        logger.debug("Warte auf Telegram-Antwort...")

        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{self._api}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": poll_timeout,
                        "allowed_updates": ["message"],
                    },
                    timeout=poll_timeout + 5,
                )
                if not resp.ok:
                    time.sleep(2)
                    continue

                updates = resp.json().get("result", [])
                for update in updates:
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    # Nur Nachrichten von der konfigurierten Chat-ID akzeptieren
                    if str(msg.get("chat", {}).get("id")) == self.chat_id:
                        text = msg.get("text", "").strip()
                        if text:
                            logger.info(f"Telegram-Antwort empfangen: {text[:80]}")
                            return text

            except Exception as e:
                logger.warning(f"Telegram poll Fehler: {e}")
                time.sleep(3)

        logger.warning("Telegram-Timeout — kein Input erhalten, nutze leeren String")
        return ""

    def _get_current_update_id(self) -> int:
        """Gibt die ID des letzten vorhandenen Updates zurück um alte zu überspringen."""
        try:
            resp = requests.get(
                f"{self._api}/getUpdates",
                params={"offset": -1, "limit": 1},
                timeout=10,
            )
            updates = resp.json().get("result", [])
            if updates:
                return updates[-1]["update_id"]
        except Exception:
            pass
        return 0

    @staticmethod
    def _read_terminal() -> str:
        try:
            return input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "exit"

    # ── Kombiniert ───────────────────────────────────────────────────────────

    def notify_and_wait(self, message: str, timeout: int = 3600) -> str:
        """Sendet eine Nachricht und wartet auf Antwort des Users."""
        self.send(message)
        return self.wait_for_reply(timeout)
