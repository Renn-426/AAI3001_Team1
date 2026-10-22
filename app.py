"""Child safety monitor - Streamlit UI base.

Run:  streamlit run app.py
"""
import time

import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from modules.camera import CameraProcessor
from modules.classifier import DangerLevel, get_classifier
from modules.notifier import TelegramNotifier

st.set_page_config(page_title="Child Safety Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ Child Safety Monitor")

# ---------------- sidebar: all tunable settings live here ----------------
with st.sidebar:
    st.header("Settings")
    infer_every = st.slider("Run model every N frames", 1, 30, 10)
    show_overlay = st.checkbox("Draw overlay on video", True)

    st.subheader("Telegram")
    min_level_name = st.selectbox("Alert at level ≥", [l.name for l in DangerLevel][1:], index=2)
    cooldown = st.number_input("Cooldown between alerts (s)", 5, 3600, 60)
    notifier = TelegramNotifier(cooldown_s=int(cooldown))
    st.caption("Telegram: ✅ configured" if notifier.configured else "Telegram: ❌ missing token/chat id")
    if st.button("Send test message", disabled=not notifier.configured):
        st.success("Sent!") if notifier.send_text("✅ Test message from Child Safety Monitor") \
            else st.error("Failed - check token / chat id")

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
    proc.classifier = get_classifier()          # swap real model in classifier.get_classifier()
    proc.infer_every_n_frames = infer_every
    proc.draw_overlay = show_overlay

    while ctx.state.playing:
        result = proc.latest_result
        if result:
            level_box.markdown(
                f"<div style='background:{result.level.color};color:white;padding:16px;"
                f"border-radius:8px;font-size:28px;text-align:center'>"
                f"{result.level.name}</div>", unsafe_allow_html=True)
            detail_box.write({
                "score": round(result.score, 3),
                "reason": result.reason,
                "detections": [d.label for d in result.detections],
            })

            # ---- TELEGRAM TRIGGER: disabled for now. Uncomment when ready ----
            # notifier.send_alert(result, proc.latest_frame,
            #                     min_level=DangerLevel[min_level_name])
        time.sleep(0.3)
else:
    level_box.info("Press START on the video widget to begin.")
