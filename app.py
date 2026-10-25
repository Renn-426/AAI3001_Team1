"""Child safety monitor - Streamlit UI.

Run:  python -m streamlit run app.py

Integration: the UI only talks to `get_classifier()` in modules/classifier.py.
Whatever it returns just needs `predict(frame_bgr) -> ClassificationResult`.
"""
import threading
import time

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from modules.camera import CameraProcessor, draw_result
from modules.classifier import ClassificationResult, DangerLevel, get_classifier
from modules.notifier import TelegramNotifier
from modules.subscriptions import SubscriptionStore, start_poller

st.set_page_config(page_title="Child Safety Monitor", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.badge {padding:18px;border-radius:12px;color:#fff;text-align:center;
        font-size:32px;font-weight:700;letter-spacing:1px;margin-bottom:4px}
.reason {text-align:center;opacity:.85;margin-bottom:10px}
</style>""", unsafe_allow_html=True)
st.title("🛡️ Child Safety Monitor")

EMOJI = {DangerLevel.SAFE: "🟢", DangerLevel.LOW: "🟡", DangerLevel.MEDIUM: "🟠",
         DangerLevel.HIGH: "🔴"}


# ---------------- shared services (one per server process) ----------------
@st.cache_resource
def get_notifier() -> TelegramNotifier:
    return TelegramNotifier()


@st.cache_resource
def get_store() -> SubscriptionStore:
    store = SubscriptionStore()
    token = get_notifier().token
    if token:
        start_poller(token, store)        # listens for /start <code> and /stop
    return store


@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    """Loaded once and shared by the live camera and the image tester."""
    return get_classifier()


notifier = get_notifier()
store = get_store()
classifier = load_classifier()

ss = st.session_state
ss.setdefault("history", [])            # alert events, newest first
ss.setdefault("danger_since", None)     # when the level first reached the alert level
ss.setdefault("last_alert", 0.0)


# ---------------- small UI helpers ----------------
def level_badge(result: ClassificationResult | None) -> None:
    if result is None:
        st.markdown("<div class='badge' style='background:#7f8c8d'>NO SIGNAL</div>",
                    unsafe_allow_html=True)
        return
    st.markdown(f"<div class='badge' style='background:{result.level.color}'>"
                f"{EMOJI[result.level]} {result.level.name}</div>"
                f"<div class='reason'>{result.reason or '&nbsp;'}</div>",
                unsafe_allow_html=True)


def result_details(result: ClassificationResult | None, fps: float | None, ms: float | None):
    c = st.columns(3)
    c[0].metric("Score", f"{result.score:.2f}" if result else "–")
    c[1].metric("Model FPS", f"{fps:.1f}" if fps else "–")
    c[2].metric("Latency (ms)", f"{ms:.0f}" if ms else "–")
    if result and result.detections:
        st.dataframe([{"label": d.label, "confidence": round(d.confidence, 2),
                       "bbox": str(d.bbox)} for d in result.detections], hide_index=True)
    else:
        st.caption("No detections")


def send_alert_async(result: ClassificationResult, frame, camera_code: str) -> None:
    """Log the alert and send it on a background thread (the UI never waits on Telegram)."""
    thumb = None
    if frame is not None:
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        thumb = cv2.imencode(".jpg", small)[1].tobytes()
    event = {"time": time.strftime("%H:%M:%S"), "level": result.level, "score": result.score,
             "reason": result.reason, "labels": [d.label for d in result.detections],
             "thumb": thumb, "status": "sending..."}
    ss.history.insert(0, event)
    del ss.history[50:]
    ss.last_alert = time.time()

    def job() -> None:
        ok = notifier.send_alert(result, frame, camera_code, store, force=True)
        event["status"] = "✅ sent" if ok else "❌ failed"
    threading.Thread(target=job, daemon=True).start()


# ---------------- sidebar: all tunable settings live here ----------------
with st.sidebar:
    st.header("Detection")
    st.caption(f"Model: `{type(classifier).__name__}`")
    max_fps = st.slider("Max model runs / second", 1, 15, 5)
    show_overlay = st.checkbox("Draw overlay on video", True)

    st.header("Alerts")
    auto_alerts = st.toggle("Enable automatic alerts", value=False,
                            help="Off by default. Turn on once the real model is connected.")
    min_level = DangerLevel[st.selectbox("Alert at level ≥", [l.name for l in DangerLevel][1:],
                                         index=2)]
    confirm_s = st.slider("Level must last (s)", 0.0, 5.0, 1.5, 0.5,
                          help="Avoids alerts from a single noisy frame.")
    cooldown = st.number_input("Cooldown between alerts (s)", 5, 3600, 60)
    notifier.cooldown_s = int(cooldown)

    st.header("Telegram")
    if not notifier.configured:
        st.error("Missing TELEGRAM_BOT_TOKEN (see README).")

    cameras = store.cameras()                      # {code: name}
    camera_code = None
    if cameras:
        camera_code = st.selectbox(
            "Active camera", list(cameras.keys()), format_func=lambda c: cameras[c])
        st.caption(f"👥 {len(store.chats_for(camera_code))} subscriber(s) on this camera")
    else:
        st.info("No camera yet. Create one below.")

    with st.expander("➕ New camera / invite link", expanded=not cameras):
        new_name = st.text_input("Camera name", placeholder="Living room")
        if st.button("Generate link", disabled=not (notifier.configured and new_name.strip())):
            code = store.create_camera(new_name.strip())
            ss["last_link"] = notifier.deep_link(code)
            st.rerun()
        if ss.get("last_link"):
            st.caption("Send this link to parents - they open it and press Start:")
            st.code(ss["last_link"])

    c1, c2 = st.columns(2)
    if c1.button("Test message", disabled=not (notifier.configured and camera_code)):
        sent = notifier.broadcast_text(store, camera_code, "✅ Test message from Child Safety Monitor")
        st.toast(f"Sent to {sent} subscriber(s)" if sent else "No subscribers, or sending failed")
    if camera_code and c2.button("🗑️ Revoke camera"):
        store.delete_camera(camera_code)
        ss.pop("last_link", None)
        st.rerun()


def alert_status(result: ClassificationResult) -> str:
    """Decide whether to alert, and explain why / why not."""
    now = time.time()
    if result.level >= min_level:
        ss.danger_since = ss.danger_since or now
    else:
        ss.danger_since = None

    if not auto_alerts:
        return "⏸️ Automatic alerts are OFF (toggle in sidebar)"
    if not camera_code:
        return "⚠️ No camera selected"
    if not store.chats_for(camera_code):
        return "⚠️ No subscribers yet - open the invite link in Telegram"
    if ss.danger_since is None:
        return f"✅ Below alert level ({min_level.name})"
    if now - ss.danger_since < confirm_s:
        return f"👀 Confirming... {now - ss.danger_since:.1f}/{confirm_s:.1f}s"
    if now - ss.last_alert < notifier.cooldown_s:
        return f"⏳ Cooldown ({notifier.cooldown_s - (now - ss.last_alert):.0f}s left)"
    return "SEND"


tab_live, tab_test, tab_history = st.tabs(["📹 Live", "🖼️ Test image", "📜 Alert history"])

# ---------------- live camera ----------------
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with tab_live:
    col_video, col_status = st.columns([3, 2])
    with col_video:
        ctx = webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=CameraProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    @st.fragment(run_every=0.5)             # refreshes only this panel, twice a second
    def live_status() -> None:
        proc: CameraProcessor | None = ctx.video_processor
        if not proc or not ctx.state.playing:
            level_badge(None)
            st.info("Press START on the video widget to begin.")
            return
        proc.classifier = classifier
        proc.max_infer_fps = max_fps
        proc.draw_overlay = show_overlay
        if proc.error:
            st.error(f"Model error: {proc.error}")

        result = proc.latest_result
        level_badge(result)
        if result:
            status = alert_status(result)
            if status == "SEND":
                send_alert_async(result, proc.latest_frame, camera_code)
                status = "🚨 Alert sent - see Alert history"
            st.caption(status)
        result_details(result, proc.infer_fps, proc.latency_ms)

        if st.button("Send test ALERT (photo)", disabled=not camera_code):
            fake = ClassificationResult(DangerLevel.HIGH, 1.0, [], "Manual test alert")
            send_alert_async(fake, proc.latest_frame, camera_code)
            st.toast("Test alert queued - see Alert history")

    with col_status:
        live_status()

# ---------------- image tester: same model, no camera needed ----------------
with tab_test:
    st.caption("Runs the same model as the live camera. Handy for checking a new model, "
               "or when there is no camera (e.g. on the cloud).")
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if up:
        frame = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        result = classifier.predict(frame)
        ms = (time.perf_counter() - t0) * 1000
        c1, c2 = st.columns([3, 2])
        c1.image(draw_result(frame, result), channels="BGR")
        with c2:
            level_badge(result)
            result_details(result, None, ms)

# ---------------- alert history ----------------
with tab_history:
    @st.fragment(run_every=2)
    def history() -> None:
        if not ss.history:
            st.info("No alerts yet.")
        for e in ss.history:
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                if e["thumb"]:
                    c1.image(e["thumb"])
                c2.markdown(f"**{EMOJI[e['level']]} {e['level'].name}** · "
                            f"score {e['score']:.2f} · {e['time']}")
                c2.write(e["reason"])
                if e["labels"]:
                    c2.caption("Detected: " + ", ".join(sorted(set(e["labels"]))))
                c2.caption(f"Telegram: {e['status']}")

    history()
