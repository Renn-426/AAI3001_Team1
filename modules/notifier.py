"""Telegram notification base (plain Bot API over HTTPS, no extra library).

Setup:
1. Talk to @BotFather on Telegram -> /newbot -> copy the token.
2. Send any message to your new bot, then open
   https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}
3. Put both in .streamlit/secrets.toml (or env vars) - see README.
"""
from __future__ import annotations

import os
import time

import cv2
import requests

from modules.classifier import ClassificationResult, DangerLevel


def _get_secret(name: str) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None,
                 cooldown_s: int = 60) -> None:
        self.token = token or _get_secret("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or _get_secret("TELEGRAM_CHAT_ID")
        self.cooldown_s = cooldown_s
        self._last_sent = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send_text(self, text: str) -> bool:
        if not self.configured:
            return False
        r = requests.post(self._url("sendMessage"),
                          data={"chat_id": self.chat_id, "text": text}, timeout=10)
        return r.ok

    def send_photo(self, frame_bgr, caption: str = "") -> bool:
        if not self.configured:
            return False
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return False
        r = requests.post(self._url("sendPhoto"),
                          data={"chat_id": self.chat_id, "caption": caption},
                          files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
                          timeout=15)
        return r.ok

    def can_send(self) -> bool:
        return (time.time() - self._last_sent) >= self.cooldown_s

    def send_alert(self, result: ClassificationResult, frame_bgr=None,
                   min_level: DangerLevel = DangerLevel.HIGH) -> bool:
        """Send an alert if level >= min_level and cooldown has passed.

        NOT called anywhere automatically yet - the UI has a manual test
        button. Wire it into the live loop in app.py when ready.
        """
        if result.level < min_level or not self.can_send():
            return False
        caption = (f"⚠️ Danger level: {result.level.name} "
                   f"(score {result.score:.2f})\n{result.reason}")
        sent = (self.send_photo(frame_bgr, caption) if frame_bgr is not None
                else self.send_text(caption))
        if sent:
            self._last_sent = time.time()
        return sent
