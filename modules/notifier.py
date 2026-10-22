"""Telegram notification (plain Bot API over HTTPS, no extra library).

Setup:
1. Talk to @BotFather on Telegram -> /newbot -> copy the token.
2. Put it in .streamlit/secrets.toml as TELEGRAM_BOT_TOKEN (or env var).
Chat IDs are no longer fixed: parents subscribe per camera through a
deep link (see subscriptions.py).
"""
from __future__ import annotations

import os
import time

import cv2
import requests

from modules.classifier import ClassificationResult, DangerLevel
from modules.subscriptions import SubscriptionStore


def _get_secret(name: str) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


class TelegramNotifier:
    def __init__(self, token: str | None = None, cooldown_s: int = 60) -> None:
        self.token = token or _get_secret("TELEGRAM_BOT_TOKEN")
        self.cooldown_s = cooldown_s
        self._last_sent: dict[str, float] = {}      # per camera code
        self._bot_username: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def bot_username(self) -> str | None:
        """Needed to build t.me deep links. Looked up once via getMe."""
        if self._bot_username or not self.configured:
            return self._bot_username
        try:
            r = requests.get(self._url("getMe"), timeout=10).json()
            self._bot_username = r["result"]["username"]
        except Exception:
            self._bot_username = _get_secret("TELEGRAM_BOT_USERNAME")
        return self._bot_username

    def deep_link(self, code: str) -> str | None:
        name = self.bot_username()
        return f"https://t.me/{name}?start={code}" if name else None

    # ---- low-level senders ----
    def send_text(self, chat_id: int, text: str) -> bool:
        if not self.configured:
            return False
        r = requests.post(self._url("sendMessage"),
                          data={"chat_id": chat_id, "text": text}, timeout=10)
        return r.ok

    def send_photo(self, chat_id: int, frame_bgr, caption: str = "") -> bool:
        if not self.configured:
            return False
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return False
        r = requests.post(self._url("sendPhoto"),
                          data={"chat_id": chat_id, "caption": caption},
                          files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
                          timeout=15)
        return r.ok

    # ---- high-level ----
    def broadcast_text(self, store: SubscriptionStore, camera_code: str, text: str) -> int:
        """Send text to every subscriber of a camera. Returns how many succeeded."""
        return sum(self.send_text(c, text) for c in store.chats_for(camera_code))

    def can_send(self, camera_code: str) -> bool:
        return (time.time() - self._last_sent.get(camera_code, 0.0)) >= self.cooldown_s

    def send_alert(self, result: ClassificationResult, frame_bgr,
                   camera_code: str, store: SubscriptionStore,
                   min_level: DangerLevel = DangerLevel.HIGH,
                   force: bool = False) -> bool:
        """Alert all subscribers of `camera_code` if level >= min_level and
        the per-camera cooldown has passed. force=True skips both checks
        (used by the manual test button)."""
        if not force and (result.level < min_level or not self.can_send(camera_code)):
            return False
        chats = store.chats_for(camera_code)
        if not chats:
            return False

        cam_name = store.cameras().get(camera_code, "camera")
        caption = (f"⚠️ [{cam_name}] Danger level: {result.level.name} "
                   f"(score {result.score:.2f})\n{result.reason}")
        sent = 0
        for chat_id in chats:
            if frame_bgr is not None:
                sent += self.send_photo(chat_id, frame_bgr, caption)
            else:
                sent += self.send_text(chat_id, caption)
        if sent:
            self._last_sent[camera_code] = time.time()
        return bool(sent)