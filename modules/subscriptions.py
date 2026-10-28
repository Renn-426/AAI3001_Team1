"""Homes, rooms and Telegram members (deep-link flow).

Model:
  Home   - one household. Has a private `manage_key` (opens it in the app) and
           an `invite_code` (the Telegram link). One link covers ALL its rooms.
  Room   - one camera location in a home ("Living room", "Bedroom").
  Member - a Telegram chat (a parent, or a family group) that joined a home.
           Members can mute single rooms from Telegram with /rooms.

Flow:
  app: create home -> add rooms -> share t.me/<bot>?start=<invite_code>
  parent opens link, presses Start -> bot gets "/start <invite_code>" -> member
  camera in "Bedroom" sees danger -> alert goes to every member of that home
  who has not muted "Bedroom".

Storage is SQLite (safe with many users and threads, one file). On hosts that
wipe the disk on restart, point `path` at persistent storage or swap this
class for Postgres/Supabase - the public methods are all the app uses.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass

import requests

SCHEMA = """
CREATE TABLE IF NOT EXISTS homes (
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    manage_key TEXT UNIQUE NOT NULL, invite_code TEXT UNIQUE NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS rooms (
    id TEXT PRIMARY KEY, home_id TEXT NOT NULL REFERENCES homes(id) ON DELETE CASCADE,
    name TEXT NOT NULL, UNIQUE (home_id, name));
CREATE TABLE IF NOT EXISTS members (
    home_id TEXT NOT NULL REFERENCES homes(id) ON DELETE CASCADE,
    chat_id INTEGER NOT NULL, display TEXT, joined REAL,
    PRIMARY KEY (home_id, chat_id));
CREATE INDEX IF NOT EXISTS members_by_chat ON members(chat_id);
CREATE TABLE IF NOT EXISTS mutes (
    chat_id INTEGER NOT NULL,
    room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    PRIMARY KEY (chat_id, room_id));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True)
class Home:
    id: str
    name: str
    manage_key: str
    invite_code: str


def _token(n: int = 6) -> str:
    return secrets.token_urlsafe(n)


class SubscriptionStore:
    def __init__(self, path: str = "subscribers.db") -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(SCHEMA)

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock, self._db:          # one transaction per call
            return self._db.execute(sql, args).fetchall()

    # ---- homes ----
    def create_home(self, name: str) -> Home:
        home = Home(_token(), name, secrets.token_urlsafe(12), _token(8))
        self._q("INSERT INTO homes VALUES (?, ?, ?, ?, ?)",
                (home.id, home.name, home.manage_key, home.invite_code, time.time()))
        return home

    def _home(self, rows) -> Home | None:
        return Home(**{k: rows[0][k] for k in ("id", "name", "manage_key", "invite_code")}) \
            if rows else None

    def home_by_key(self, manage_key: str) -> Home | None:
        return self._home(self._q("SELECT * FROM homes WHERE manage_key = ?", (manage_key,)))

    def home(self, home_id: str) -> Home | None:
        return self._home(self._q("SELECT * FROM homes WHERE id = ?", (home_id,)))

    def new_invite(self, home_id: str) -> str:
        """Replace the invite link (old link stops working, members stay)."""
        code = _token(8)
        self._q("UPDATE homes SET invite_code = ? WHERE id = ?", (code, home_id))
        return code

    def delete_home(self, home_id: str) -> None:
        self._q("DELETE FROM homes WHERE id = ?", (home_id,))

    # ---- rooms ----
    def add_room(self, home_id: str, name: str) -> str | None:
        """Returns the room id, or None if the home already has a room with that name."""
        room_id = _token()
        try:
            self._q("INSERT INTO rooms VALUES (?, ?, ?)", (room_id, home_id, name))
        except sqlite3.IntegrityError:
            return None
        return room_id

    def delete_room(self, room_id: str) -> None:
        self._q("DELETE FROM rooms WHERE id = ?", (room_id,))

    def rooms(self, home_id: str) -> dict[str, str]:
        """{room_id: name}"""
        return {r["id"]: r["name"] for r in
                self._q("SELECT id, name FROM rooms WHERE home_id = ? ORDER BY name", (home_id,))}

    def room_label(self, room_id: str) -> str:
        r = self._q("SELECT h.name AS home, r.name AS room FROM rooms r "
                    "JOIN homes h ON h.id = r.home_id WHERE r.id = ?", (room_id,))
        return f"{r[0]['home']} · {r[0]['room']}" if r else "Unknown room"

    # ---- members ----
    def join(self, invite_code: str, chat_id: int, display: str) -> Home | None:
        rows = self._q("SELECT * FROM homes WHERE invite_code = ?", (invite_code,))
        home = self._home(rows)
        if home:
            self._q("INSERT OR REPLACE INTO members VALUES (?, ?, ?, ?)",
                    (home.id, chat_id, display, time.time()))
        return home

    def leave(self, home_id: str, chat_id: int) -> None:
        self._q("DELETE FROM mutes WHERE chat_id = ? AND room_id IN "
                "(SELECT id FROM rooms WHERE home_id = ?)", (chat_id, home_id))
        self._q("DELETE FROM members WHERE home_id = ? AND chat_id = ?", (home_id, chat_id))

    def leave_all(self, chat_id: int) -> None:
        self._q("DELETE FROM mutes WHERE chat_id = ?", (chat_id,))
        self._q("DELETE FROM members WHERE chat_id = ?", (chat_id,))

    def members(self, home_id: str) -> list[tuple[int, str]]:
        return [(r["chat_id"], r["display"]) for r in self._q(
            "SELECT chat_id, display FROM members WHERE home_id = ? ORDER BY joined", (home_id,))]

    def homes_for_chat(self, chat_id: int) -> list[Home]:
        rows = self._q("SELECT h.* FROM homes h JOIN members m ON m.home_id = h.id "
                       "WHERE m.chat_id = ? ORDER BY h.name", (chat_id,))
        return [self._home([r]) for r in rows]

    def toggle_mute(self, chat_id: int, room_id: str) -> bool:
        """Returns True if the room is now muted for this chat."""
        if self._q("SELECT 1 FROM mutes WHERE chat_id = ? AND room_id = ?", (chat_id, room_id)):
            self._q("DELETE FROM mutes WHERE chat_id = ? AND room_id = ?", (chat_id, room_id))
            return False
        self._q("INSERT INTO mutes VALUES (?, ?)", (chat_id, room_id))
        return True

    def muted_rooms(self, chat_id: int) -> set[str]:
        return {r["room_id"] for r in self._q("SELECT room_id FROM mutes WHERE chat_id = ?",
                                              (chat_id,))}

    def recipients(self, room_id: str) -> list[int]:
        """Chats that should get alerts for this room (home members minus mutes)."""
        return [r["chat_id"] for r in self._q(
            "SELECT m.chat_id FROM members m JOIN rooms r ON r.home_id = m.home_id "
            "WHERE r.id = ? AND m.chat_id NOT IN "
            "(SELECT chat_id FROM mutes WHERE room_id = ?)", (room_id, room_id))]

    # ---- polling offset ----
    def offset(self) -> int:
        r = self._q("SELECT value FROM meta WHERE key = 'offset'")
        return int(r[0]["value"]) if r else 0

    def set_offset(self, offset: int) -> None:
        self._q("INSERT OR REPLACE INTO meta VALUES ('offset', ?)", (str(offset),))


# ---------------- Telegram bot side ----------------
HELP = ("Commands:\n/rooms - choose which rooms alert you\n"
        "/stop - stop all alerts\n/help - this message")


class BotPoller:
    """Handles /start <code>, /rooms, /stop and the mute buttons.

    Uses long polling: one request waits up to 25 s for new messages, so
    replies are instant without hammering Telegram. Only ONE process may poll
    a bot token; if you run several app servers, switch to a webhook.
    """

    def __init__(self, token: str, store: SubscriptionStore, api_base: str) -> None:
        self.store = store
        self.base = f"{api_base}/bot{token}"

    def _call(self, method: str, **params) -> dict:
        r = requests.post(f"{self.base}/{method}", json=params, timeout=35)
        return r.json()

    def _rooms_keyboard(self, home: Home, chat_id: int) -> dict:
        muted = self.store.muted_rooms(chat_id)
        rows = [[{"text": f"{'🔕' if rid in muted else '🔔'} {name}",
                  "callback_data": f"m|{rid}"}]
                for rid, name in self.store.rooms(home.id).items()]
        rows.append([{"text": f"🚪 Leave {home.name}", "callback_data": f"l|{home.id}"}])
        return {"inline_keyboard": rows}

    def _send_rooms(self, chat_id: int, home: Home) -> None:
        rooms = self.store.rooms(home.id)
        text = (f"🏠 {home.name}\nTap a room to mute / unmute its alerts."
                if rooms else f"🏠 {home.name}\nNo rooms yet - they will appear here once added.")
        self._call("sendMessage", chat_id=chat_id, text=text,
                   reply_markup=self._rooms_keyboard(home, chat_id))

    def handle(self, u: dict) -> None:
        if "callback_query" in u:
            return self._handle_button(u["callback_query"])
        msg = u.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        if not text or chat_id is None:
            return
        cmd, _, arg = text.partition(" ")
        cmd = cmd.split("@")[0]              # "/rooms@MyBot" in group chats

        if cmd == "/start" and arg:
            display = chat.get("title") or " ".join(
                filter(None, [chat.get("first_name"), chat.get("last_name")])) or str(chat_id)
            home = self.store.join(arg.strip(), chat_id, display)
            if not home:
                self._call("sendMessage", chat_id=chat_id,
                           text="❌ This link is invalid or has been replaced. "
                                "Ask for a new link from the monitor app.")
                return
            self._call("sendMessage", chat_id=chat_id,
                       text=f"✅ You joined {home.name}. You'll get alerts from all its rooms.\n\n"
                            + HELP)
            self._send_rooms(chat_id, home)
        elif cmd == "/rooms":
            homes = self.store.homes_for_chat(chat_id)
            if not homes:
                self._call("sendMessage", chat_id=chat_id,
                           text="You haven't joined any home yet. Open the invite link from the app.")
            for home in homes:
                self._send_rooms(chat_id, home)
        elif cmd == "/stop":
            self.store.leave_all(chat_id)
            self._call("sendMessage", chat_id=chat_id,
                       text="You left all homes and will get no more alerts.")
        else:
            self._call("sendMessage", chat_id=chat_id,
                       text=("Open the invite link from the monitor app to subscribe.\n\n"
                             if cmd == "/start" else "") + HELP)

    def _handle_button(self, q: dict) -> None:
        chat_id = q.get("message", {}).get("chat", {}).get("id")
        kind, _, ref = (q.get("data") or "").partition("|")
        note = ""
        if chat_id is not None and kind == "m":
            home_ids = {h.id for h in self.store.homes_for_chat(chat_id)}
            home = next((h for h in map(self.store.home, home_ids)
                         if h and ref in self.store.rooms(h.id)), None)
            if home:                          # only members can mute that home's rooms
                note = "Muted" if self.store.toggle_mute(chat_id, ref) else "Unmuted"
                self._call("editMessageReplyMarkup", chat_id=chat_id,
                           message_id=q["message"]["message_id"],
                           reply_markup=self._rooms_keyboard(home, chat_id))
        elif chat_id is not None and kind == "l":
            home = self.store.home(ref)
            self.store.leave(ref, chat_id)
            note = "Left"
            self._call("editMessageText", chat_id=chat_id, message_id=q["message"]["message_id"],
                       text=f"You left {home.name if home else 'this home'}.")
        self._call("answerCallbackQuery", callback_query_id=q["id"], text=note)

    def poll_once(self, timeout: int = 25) -> None:
        r = requests.get(f"{self.base}/getUpdates", timeout=timeout + 10, params={
            "offset": self.store.offset(), "timeout": timeout,
            "allowed_updates": json.dumps(["message", "callback_query"])}).json()
        for u in r.get("result", []):
            self.store.set_offset(u["update_id"] + 1)
            try:
                self.handle(u)
            except Exception as e:            # one bad update must not stop the bot
                print(f"[telegram] update {u.get('update_id')} failed: {e!r}")


def start_poller(token: str, store: SubscriptionStore,
                 api_base: str = "https://api.telegram.org") -> threading.Thread:
    """Background thread. Call ONCE per process (wrap in st.cache_resource)."""
    bot = BotPoller(token, store, api_base)

    def loop() -> None:
        try:
            bot._call("setMyCommands", commands=[
                {"command": "rooms", "description": "Choose which rooms alert you"},
                {"command": "stop", "description": "Stop all alerts"},
                {"command": "help", "description": "Show help"}])
        except Exception:
            pass
        while True:
            try:
                bot.poll_once()
            except Exception:
                time.sleep(3)                 # network hiccup, retry

    t = threading.Thread(target=loop, daemon=True, name="telegram-poller")
    t.start()
    return t
