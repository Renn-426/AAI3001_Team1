"""Telegram notification (plain Bot API over HTTPS, no extra library).

Setup:
1. Talk to @BotFather on Telegram -> /newbot -> copy the token.
2. Put it in .streamlit/secrets.toml as TELEGRAM_BOT_TOKEN (or env var).
Recipients are not fixed: parents join a home through its invite link and
alerts go to that home's members (see subscriptions.py).

Built for many users:
- every send goes through ONE background queue, paced under Telegram's limit
  (~30 messages/s per bot) and retried when Telegram answers 429 "slow down";
- an alert photo is uploaded once, then re-sent to other members by file_id;
- chats that blocked the bot are removed automatically;
- the cooldown is per room and shared by all viewers, so two browsers watching
  the same room don't double-alert.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import requests

from modules.classifier import ClassificationResult
from modules.subscriptions import SubscriptionStore

MIN_GAP_S = 1 / 25                     # stay below Telegram's ~30 msg/s bot limit


def _get_secret(name: str) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


class TelegramNotifier:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or _get_secret("TELEGRAM_BOT_TOKEN")
        # TELEGRAM_API_BASE is only for tests / a self-hosted Bot API server.
        self.api_base = _get_secret("TELEGRAM_API_BASE") or "https://api.telegram.org"
        self._bot_username: str | None = None
        self._queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="telegram-send")
        self._last_call = 0.0
        self._cooldown_lock = threading.Lock()
        self._last_alert: dict[str, float] = {}       # per room id

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def _url(self, method: str) -> str:
        return f"{self.api_base}/bot{self.token}/{method}"

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

    def deep_link(self, invite_code: str) -> str | None:
        name = self.bot_username()
        return f"https://t.me/{name}?start={invite_code}" if name else None

    # ---- cooldown (shared by every viewer of a room) ----
    def claim_cooldown(self, room_id: str, cooldown_s: float) -> bool:
        """True (and starts the cooldown) if this room may alert now."""
        with self._cooldown_lock:
            now = time.time()
            if now - self._last_alert.get(room_id, 0.0) < cooldown_s:
                return False
            self._last_alert[room_id] = now
            return True

    def cooldown_left(self, room_id: str, cooldown_s: float) -> float:
        return max(0.0, cooldown_s - (time.time() - self._last_alert.get(room_id, 0.0)))

    # ---- low level: one API call, paced + retried (runs on the queue thread) ----
    def _call(self, method: str, data: dict, files: dict | None = None) -> dict:
        for _ in range(3):
            wait = self._last_call + MIN_GAP_S - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                r = requests.post(self._url(method), data=data, files=files, timeout=20).json()
            except (requests.RequestException, ValueError):
                time.sleep(1)
                continue
            if r.get("error_code") == 429:                 # Telegram asks us to slow down
                time.sleep(r.get("parameters", {}).get("retry_after", 1))
                continue
            return r
        return {"ok": False}

    def _deliver(self, chats: list[int], text: str, jpeg: bytes | None,
                 store: SubscriptionStore | None) -> int:
        sent, photo_id = 0, None
        for chat_id in chats:
            if jpeg is None:
                r = self._call("sendMessage", {"chat_id": chat_id, "text": text})
            elif photo_id:                                  # reuse the uploaded photo
                r = self._call("sendPhoto", {"chat_id": chat_id, "caption": text,
                                             "photo": photo_id})
            else:
                r = self._call("sendPhoto", {"chat_id": chat_id, "caption": text},
                               {"photo": ("alert.jpg", jpeg, "image/jpeg")})
                if r.get("ok"):
                    photo_id = r["result"]["photo"][-1]["file_id"]
            if r.get("ok"):
                sent += 1
            elif r.get("error_code") == 403 and store:     # user blocked the bot
                store.leave_all(chat_id)
        return sent

    # ---- high level (non-blocking: return a Future with the number delivered) ----
    def send_text(self, chats: list[int], text: str) -> Future:
        return self._queue.submit(self._deliver, list(chats), text, None, None)

    def send_alert(self, result: ClassificationResult, frame_bgr, room_id: str,
                   store: SubscriptionStore) -> Future:
        """Alert everyone subscribed to `room_id`. Policy (level, cooldown) is the caller's."""
        caption = (f"⚠️ [{store.room_label(room_id)}] Danger level: {result.level.name} "
                   f"(score {result.score:.2f})\n{result.reason}\n{time.strftime('%H:%M:%S')}")
        jpeg = None
        if frame_bgr is not None:
            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg = buf.tobytes() if ok else None
        return self._queue.submit(self._deliver, store.recipients(room_id), caption, jpeg, store)
