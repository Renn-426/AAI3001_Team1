"""Child safety monitor - Streamlit UI base.

Run:  python -m streamlit run app.py
"""
import time

import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from modules.camera import CameraProcessor
from modules.classifier import ClassificationResult, DangerLevel, get_classifier
from modules.notifier import TelegramNotifier
from modules.subscriptions import SubscriptionStore, start_poller

st.set_page_config(page_title="Child Safety Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ Child Safety Monitor")


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


notifier = get_notifier()
store = get_store()

# ---------------- sidebar: all tunable settings live here ----------------
with st.sidebar:
    st.header("Settings")
    infer_every = st.slider("Run model every N frames", 1, 30, 10)
    show_overlay = st.checkbox("Draw overlay on video", True)

    st.subheader("Telegram")
    if not notifier.configured:
        st.error("Missing TELEGRAM_BOT_TOKEN (see README).")

    # --- camera selection / creation ---
    cameras = store.cameras()                      # {code: name}
    camera_code = None
    if cameras:
        camera_code = st.selectbox(
            "Active camera", list(cameras.keys()), format_func=lambda c: cameras[c])
        n_subs = len(store.chats_for(camera_code))
        st.caption(f"👥 {n_subs} subscriber(s) on this camera")
    else:
        st.info("No camera yet. Create one below.")

    with st.expander("➕ New camera / invite link", expanded=not cameras):
        new_name = st.text_input("Camera name", placeholder="Living room")
        if st.button("Generate link", disabled=not (notifier.configured and new_name.strip())):
            code = store.create_camera(new_name.strip())
            st.session_state["last_link"] = notifier.deep_link(code)
            st.rerun()
        if st.session_state.get("last_link"):
            st.caption("Send this link to parents - they open it and press Start:")
            st.code(st.session_state["last_link"])

    if camera_code and st.button("🗑️ Revoke this camera"):
        store.delete_camera(camera_code)
        st.session_state.pop("last_link", None)
        st.rerun()

    # --- alert behaviour ---
    min_level_name = st.selectbox("Alert at level ≥", [l.name for l in DangerLevel][1:], index=2)
    cooldown = st.number_input("Cooldown between alerts (s)", 5, 3600, 60)
    notifier.cooldown_s = int(cooldown)
    auto_alerts = st.toggle("Enable automatic alerts", value=False,
                            help="Off by default. Turn on once the real model is connected.")

    if st.button("Send test message", disabled=not (notifier.configured and camera_code)):
        sent = notifier.broadcast_text(
            store, camera_code, "✅ Test message from Child Safety Monitor")
        if sent:
            st.success(f"Sent to {sent} subscriber(s)")
        else:
            st.error("No subscribers yet, or sending failed")

# ---------------- live camera ----------------
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

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

with col_status:
    st.subheader("Status")
    level_box = st.empty()
    detail_box = st.empty()

# ---------------- live status loop ----------------
if ctx.video_processor:
    proc = ctx.video_processor
    if proc.classifier is None:
        proc.classifier = get_classifier()      # swap real model in classifier.get_classifier()
    proc.infer_every_n_frames = infer_every
    proc.draw_overlay = show_overlay

    min_level = DangerLevel[min_level_name]

    # manual test: sends the current frame as a fake HIGH alert, ignoring the model
    if st.sidebar.button("Send test ALERT (photo)", disabled=not camera_code):
        frame = proc.latest_frame
        fake = ClassificationResult(DangerLevel.HIGH, 1.0, [], "Manual test alert")
        if frame is None:
            st.sidebar.error("No video frame yet - press START first")
        elif notifier.send_alert(fake, frame, camera_code, store, force=True):
            st.sidebar.success("Test alert sent")
        else:
            st.sidebar.error("Not sent - does this camera have subscribers?")

    alert_box = col_status.empty()

    while ctx.state.playing:
        result = proc.latest_result
        if result:
            # explain why an alert is / isn't going out
            if not auto_alerts:
                status = "⏸️ Automatic alerts are OFF (toggle in sidebar)"
            elif not camera_code:
                status = "⚠️ No camera selected"
            elif not store.chats_for(camera_code):
                status = "⚠️ No subscribers yet - open the invite link in Telegram"
            elif result.level < min_level:
                status = f"Below alert threshold ({min_level.name})"
            elif not notifier.can_send(camera_code):
                status = "⏳ Cooldown active"
            else:
                status = "🚨 Sending alert..."
            alert_box.caption(status)

            level_box.markdown(
                f"<div style='background:{result.level.color};color:white;padding:16px;"
                f"border-radius:8px;font-size:28px;text-align:center'>"
                f"{result.level.name}</div>", unsafe_allow_html=True)
            detail_box.write({
                "score": round(result.score, 3),
                "reason": result.reason,
                "detections": [d.label for d in result.detections],
            })

            if auto_alerts and camera_code:
                notifier.send_alert(result, proc.latest_frame, camera_code, store,
                                    min_level=min_level)
        time.sleep(0.3)
else:
    level_box.info("Press START on the video widget to begin.")