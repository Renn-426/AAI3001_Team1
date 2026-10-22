"""Camera <-> Telegram chat subscriptions (deep-link flow).

Flow:
  owner creates camera -> unique code -> link t.me/<bot>?start=<code>
  parent opens link, presses Start -> bot receives "/start <code>"
  poller saves chat_id under that camera -> alerts go only to those chats.

Storage is a local JSON file (fine for a demo). On Streamlit Cloud the disk is
wiped on restart; swap this class for Supabase/Firebase/Sheets later - the
public methods are all the rest of the app uses.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path

import requests


class SubscriptionStore:
    def __init__(self, path: str = "subscribers.json") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        # cameras: {code: {"name": str, "chats": [int, ...]}}
        self.data: dict = {"offset": 0, "cameras": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                pass

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    # ---- cameras ----
    def create_camera(self, name: str) -> str:
        code = secrets.token_urlsafe(8)       # unguessable, URL-safe, < 64 chars
        with self._lock:
            self.data["cameras"][code] = {"name": name, "chats": []}
            self._save()
        return code

    def delete_camera(self, code: str) -> None:
        """Revoke a camera: its link stops working and subscribers are dropped."""
        with self._lock:
            self.data["cameras"].pop(code, None)
            self._save()

    def cameras(self) -> dict[str, str]:
        """{code: name}"""
        return {c: v["name"] for c, v in self.data["cameras"].items()}

    # ---- subscribers ----
    def subscribe(self, code: str, chat_id: int) -> str | None:
        """Returns camera name, or None if the code is invalid."""
        with self._lock:
            cam = self.data["cameras"].get(code)
            if not cam:
                return None
            if chat_id not in cam["chats"]:
                cam["chats"].append(chat_id)
            self._save()
            return cam["name"]

    def unsubscribe(self, chat_id: int) -> None:
        with self._lock:
            for cam in self.data["cameras"].values():
                if chat_id in cam["chats"]:
                    cam["chats"].remove(chat_id)
            self._save()

    def chats_for(self, code: str) -> list[int]:
        return list(self.data["cameras"].get(code, {}).get("chats", []))

    # ---- polling offset ----
    def set_offset(self, offset: int) -> None:
        with self._lock:
            self.data["offset"] = offset
            self._save()


def _reply(token: str, chat_id: int, text: str) -> None:
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat_id, "text": text}, timeout=10)


def poll_updates(token: str, store: SubscriptionStore) -> None:
    """Fetch new Telegram messages once and handle /start <code> and /stop."""
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                     params={"offset": store.data["offset"], "timeout": 0},
                     timeout=15).json()
    for u in r.get("result", []):
        store.set_offset(u["update_id"] + 1)
        msg = u.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = msg.get("chat", {}).get("id")
        if not text or chat_id is None:
            continue

        if text.startswith("/start "):
            name = store.subscribe(text.split(maxsplit=1)[1], chat_id)
            _reply(token, chat_id,
                   f"✅ You'll now get alerts for: {name}\nSend /stop to unsubscribe."
                   if name else "❌ This link is invalid or has been revoked.")
        elif text == "/start":
            _reply(token, chat_id, "Please open the link from the monitor app to subscribe.")
        elif text == "/stop":
            store.unsubscribe(chat_id)
            _reply(token, chat_id, "You've been unsubscribed from all cameras.")


def start_poller(token: str, store: SubscriptionStore, interval_s: float = 3.0) -> threading.Thread:
    """Background thread. Call ONCE per process (wrap in st.cache_resource)."""
    def loop() -> None:
        while True:
            try:
                poll_updates(token, store)
            except Exception:
                pass                          # network hiccup, retry next tick
            time.sleep(interval_s)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t