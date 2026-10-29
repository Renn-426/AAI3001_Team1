"""Child Safety Monitor: python -m streamlit run app.py.

Uses the shared classifier pipeline. Dummy predictions remain labelled as
simulated, and automatic Telegram alerts require a real classifier.
"""
from __future__ import annotations

from base64 import b64encode
from html import escape
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from modules.camera import CameraProcessor, analyse_video, draw_result
from modules.classifier import (
    BaseClassifier, ClassificationResult, DangerLevel, DummyClassifier, get_classifier,
)
from modules.notifier import TelegramNotifier
from modules.subscriptions import SubscriptionStore, start_poller


st.set_page_config(page_title="Child Safety Monitor", page_icon="🛡️", layout="wide")
st.html("""
<style>
:root { --preview-navy: #102240; --preview-teal: #10888e; }
.stApp { background: #f7f9fc; color: #102240; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1550px; padding: 2rem 2.2rem 2rem; }
h1, h2, h3, p { color: #102240; }
h1 { font-size: 2rem !important; letter-spacing: -.055rem; padding-top: .35rem !important; }
h3 { font-size: 1.1rem !important; }
.brand { display: flex; align-items: center; gap: 8px; font-size: 18px; font-weight: 750; }
.brand-shield { display: grid; place-items: center; width: 37px; height: 42px;
  background: #10888e; color: white; clip-path: polygon(50% 0, 93% 15%, 86% 67%, 50% 100%, 14% 67%, 7% 15%); }
.preview-label { color: #64748b; font-size: 12px; margin: 8px 0 0 49px; }
[data-testid="stTabs"] [role="tablist"] {
  gap: 8px; padding: 8px 0; background: #f7f9fc;
  border-top: 1px solid #dce3ed; border-bottom: 1px solid #dce3ed;
}
[data-testid="stTabs"] [data-testid="stTab"] {
  display: inline-flex; align-items: center; justify-content: center; gap: 12px;
  flex: 0 0 auto; height: 48px; padding: 0 24px; border: 0;
  border-radius: 8px; background: transparent; color: #233954;
  white-space: nowrap; transition: background-color .15s ease;
}
[data-testid="stTabs"] [data-testid="stTab"] p {
  color: inherit !important; font-size: 16px; font-weight: 500; line-height: 1.3;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
  background: #08777e !important; color: #fff !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="false"]:hover { background: #e9f0f5; }
[data-testid="stTabs"] [data-testid="stTab"]:focus-visible { outline: 2px solid #10888e; outline-offset: 2px; }
/* Outline icons inherit the active tab colour; native tabs retain keyboard navigation. */
[data-testid="stTabs"] [data-testid="stTab"]::before {
  content: ""; display: inline-block; flex: 0 0 20px; width: 20px; height: 20px;
  background-color: currentColor;
  -webkit-mask: var(--nav-icon) center / contain no-repeat;
  mask: var(--nav-icon) center / contain no-repeat;
}
[data-testid="stTabs"] [data-testid="stTab"][data-key="0"] {
  --nav-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='6' width='13' height='12' rx='2'/%3E%3Cpath d='m16 10 5-3v10l-5-3'/%3E%3C/svg%3E");
}
[data-testid="stTabs"] [data-testid="stTab"][data-key="1"] {
  --nav-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2'/%3E%3Ccircle cx='8.5' cy='8.5' r='1.5'/%3E%3Cpath d='m21 15-5-5L5 21'/%3E%3C/svg%3E");
}
[data-testid="stTabs"] [data-testid="stTab"][data-key="2"] {
  --nav-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='2'/%3E%3Cpath d='M3 8h18M3 16h18M7 4v4M12 4v4M17 4v4M7 16v4M12 16v4M17 16v4'/%3E%3C/svg%3E");
}
[data-testid="stTabs"] [data-testid="stTab"][data-key="3"] {
  --nav-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M9 7h6M9 11h6M9 15h4'/%3E%3C/svg%3E");
}
[data-testid="stTabs"] [data-testid="stTab"][data-key="4"] {
  --nav-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m22 2-7 20-4-9-9-4 20-7ZM22 2 11 13'/%3E%3C/svg%3E");
}
[data-testid="stTabs"] .react-aria-SelectionIndicator,
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
[data-testid="stExpander"] { border-radius: 10px; background: #fff; }
[data-testid="stButton"] button[kind="primary"] { background: #10888e; border-color: #10888e; color: white; }
[data-testid="stPopover"] button { border-color: #d5deea; color: #102240; background: white; }
.pill { display: inline-flex; align-items: center; gap: 7px; background: #e8edf3;
  border-radius: 999px; padding: 6px 13px; color: #40516b; font-size: 13px; font-weight: 600; }
.pill.live { background: #dcf5eb; color: #06735a; }
.panel-title { font-weight: 700; font-size: 16px; margin: 1px 0 13px; }
.state-box { background: #edf1f6; border-radius: 9px; padding: 22px 14px; text-align: center; margin-bottom: 4px; }
.state-box strong { font-size: 21px; color: #102240; }
.state-box p { margin: 5px 0 0; font-size: 14px; color: #52637d; }
.state-box.compact { padding: 15px 12px; }
.state-box.compact strong { font-size: 16px; }
.state-box.error { background: #fff0ed; }
.state-box.warning { background: #fff3d8; }
.state-box.success { background: #e4f4ec; }
.state-box.severity-safe { background: #e3f6eb; border: 1px solid #9dd9b7; }
.state-box.severity-safe strong { color: #126c40; }
.state-box.severity-low { background: #fff7d6; border: 1px solid #ead27b; }
.state-box.severity-low strong { color: #806000; }
.state-box.severity-medium { background: #fff0e0; border: 1px solid #f4b16a; }
.state-box.severity-medium strong { color: #9a4600; }
.state-box.severity-high { background: #ffebeb; border: 1px solid #efaaaa; }
.state-box.severity-high strong { color: #a51f1f; }
.camera-placeholder { min-height: 280px; border-radius: 10px; display: grid;
  place-content: center; text-align: center; background: linear-gradient(135deg,#e8edf3,#f3f6f9);
  border: 1px dashed #c8d3df; color: #667a91; margin-bottom: 8px; }
.camera-placeholder .camera-symbol { font-size: 42px; margin-bottom: 8px; }
.camera-placeholder strong { font-size: 20px; color: #334b66; }
.camera-placeholder p { font-size: 14px; color: #64748b; margin-top: 8px; }
.object-row { padding: 10px 0; border-bottom: 1px solid #e4eaf1; display: flex; justify-content: space-between; }
.object-row span { color: #64748b; }
.footer { margin-top: 25px; text-align: right; font-size: 12px; color: #6b7c93; }
/* Desktop composition follows the reference; the camera scales with viewport height. */
:root { --camera-height: clamp(220px, calc(100dvh - 435px), 650px); }
[data-testid="stMainBlockContainer"] { max-width: none !important; padding: 12px 2.4vw 8px !important; }
[data-testid="stVerticalBlock"] { gap: 10px; }
.brand { height: 30px; font-family: Arial, sans-serif; font-size: 18px; }
.brand-shield { width: 23px; height: 27px; clip-path: none; background: transparent; }
.brand-shield img { display: block; width: 23px; height: 27px; }
.stApp, [data-testid="stMarkdownContainer"], input, select, button { font-family: Arial, sans-serif; }
.st-key-preview_navigation { position: relative; gap: 0; }
.st-key-preview_navigation > [data-testid="stLayoutWrapper"]:has(.st-key-preview_settings) {
  position: absolute; right: 0; top: 8px; width: 126px; z-index: 20;
}
.st-key-preview_settings [data-testid="stPopoverButton"] { height: 46px; border-radius: 8px; }
.st-key-preview_settings [data-testid="stPopoverButton"] > div > div[aria-hidden="true"] { display: none; }
[data-testid="stTabs"] [role="tablist"] { padding-right: 145px; }
[data-testid="stTabPanel"] { padding-top: 12px; }
h1 { font-family: Arial, sans-serif; font-size: 32px !important; margin: 0 !important; padding: 0 0 4px !important; }
.live-heading { display: flex; align-items: center; gap: 22px; min-height: 48px; flex-wrap: wrap; }
.live-heading h1 { flex: 0 0 auto; }
.camera-card-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 28px; }
.camera-card-header strong { font-size: 17px; }
.camera-view-tag { font-size: 12px; background: #e8edf3; color: #4b5f7b; padding: 6px 12px; border-radius: 7px; }
.st-key-preview_camera_card { background: white; }
.st-key-preview_video_frame { position: relative; height: var(--camera-height); min-height: var(--camera-height); overflow: hidden; border-radius: 8px; }
.st-key-preview_video_frame iframe { height: var(--camera-height) !important; }
.st-key-preview_video_frame:has(.camera-placeholder) iframe { position: absolute; opacity: 0; pointer-events: none; }
.camera-placeholder { height: var(--camera-height); min-height: 0; margin: 0; border: 0; }
.panel-title { display: flex; align-items: center; gap: 10px; margin: 0 0 8px; }
.panel-title { margin-bottom: 0; }
.st-key-preview_live_assessment, .st-key-preview_live_objects, .st-key-preview_live_proximity, .st-key-preview_live_alerts { padding: 11px 15px; }
.panel-title img { width: 23px; height: 23px; flex: 0 0 23px; }
.state-box { padding: 16px 12px; margin: 0; }
.state-box p { line-height: 1.35; }
.state-box.compact { padding: 12px; }
.alert-content { background: #edf1f6; border-radius: 8px; padding: 12px 15px; }
.alert-content p { margin: 0 0 7px; font-size: 14px; }
.alert-content small { color: #52637d; font-size: 13px; }
.alert-heading { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.alert-heading .panel-title { margin: 0; }
.footer { margin-top: 0; font-size: 11px; }
/* Keep widgets readable even if a viewer switched Streamlit to dark mode. */
[data-testid="stCode"] pre, [data-testid="stCode"] code {
  background: #eef3f8 !important; color: #102240 !important; }
[data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
  background: #fff !important; color: #102240 !important; }
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {
  background: #fff !important; color: #102240 !important; }
[data-testid="stExpander"] summary p, [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], button[kind="secondary"] p, button[kind="tertiary"] p { color: #102240; }
button[kind="secondary"] { background: #fff; border-color: #d5deea; }
.st-key-preview_alert_controls { padding: 11px 15px; }
.member-row, .room-row { display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-bottom: 1px solid #e4eaf1; }
.member-row span, .room-row span { color: #64748b; font-size: 13px; }
[data-testid="stExpander"] summary { min-height: 38px; }
@media (min-width: 900px) and (max-height: 900px) {
  [data-testid="stVerticalBlock"] { gap: 7px; }
  .state-box { padding: 12px; }
  .state-box strong { font-size: 20px; }
  .state-box.compact { padding: 10px; }
  .state-box.compact p { font-size: 13px; }
  }
@media (min-width: 900px) and (max-height: 800px) {
  .brand { height: 28px; }
  .st-key-preview_live_assessment, .st-key-preview_live_objects, .st-key-preview_live_proximity, .st-key-preview_live_alerts { padding: 6px 12px; }
  .state-box { padding: 9px; }
  .state-box strong { font-size: 18px; }
  .state-box.compact { padding: 8px; }
  .panel-title { font-size: 15px; margin-bottom: 0; }
  .alert-content { padding: 9px 12px; }
  .alert-content p, .alert-content small { font-size: 12px; }
}
@media (max-width: 700px) {
  .block-container { padding: 1.5rem 1rem; }
  .brand { font-size: 18px; }
  [data-testid="stTabs"] [data-testid="stTab"] { padding: 0 12px; height: 44px; gap: 8px; }
  [data-testid="stTabs"] [data-testid="stTab"] p { font-size: 14px; }
  [data-testid="stTabs"] [role="tablist"] { gap: 3px; }
}
</style>
""")


@st.cache_resource
def load_classifier() -> tuple[BaseClassifier, bool, str | None]:
    """(classifier, is_demo, problem). A broken model file falls back to demo mode."""
    try:
        model = get_classifier()
    except Exception as exc:
        return DummyClassifier(), True, f"The model could not be loaded: {exc}"
    return model, isinstance(model, DummyClassifier), None


@st.cache_resource
def load_services():
    notifier, store = TelegramNotifier(), SubscriptionStore()
    if notifier.token:
        start_poller(notifier.token, store, notifier.api_base)   # bot: /start, /rooms, /stop
    return notifier, store


classifier, is_demo, model_problem = load_classifier()
spatial_connected = type(getattr(classifier, "spatial", None)).__name__ not in ("NoneType", "ObjectRiskAnalyzer")
notifier, store = load_services()
ss = st.session_state
for name, default in {
    "preview_history": [], "preview_danger_since": None,
    "preview_last_alert": 0.0, "preview_image_result": None,
    "preview_image_frame": None, "preview_image_ms": None,
    "preview_image_key": None, "preview_image_error": None,
    "preview_video_report": None, "preview_video_key": None,
}.items():
    ss.setdefault(name, default)

# Each browser only sees ITS home. The private home key lives in the URL
# (?home=...), so a bookmark reopens it and other visitors can't see it.
home_key = st.query_params.get("home")
home = store.home_by_key(home_key) if home_key else None
rooms = store.rooms(home.id) if home else {}


class TimedClassifier(BaseClassifier):
    """Track result freshness within this preview, without changing shared modules."""

    def __init__(self, delegate: BaseClassifier):
        self.delegate = delegate
        self.last_completed_at = 0.0

    def predict(self, frame_bgr: np.ndarray) -> ClassificationResult:
        result = self.delegate.predict(frame_bgr)
        self.last_completed_at = time.monotonic()
        return result


def make_processor():
    processor = CameraProcessor()
    processor.draw_overlay = False
    processor.classifier = TimedClassifier(classifier)
    return processor


ICONS = {
    "assessment": '<path d="M9 4H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-3"/><rect x="9" y="2" width="6" height="4" rx="1"/><path d="M8 11h8M8 15h6"/>',
    "objects": '<path d="M8 5H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3M8 5l2-3h5l2 3h3a2 2 0 0 1 2 2v5"/><circle cx="12" cy="12" r="4"/><path d="m15 15 6 6"/>',
    "proximity": '<circle cx="12" cy="4" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/><path d="m10 6-4 10m8-10 4 10M8 19h8"/>',
    "bell": '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/>',
}


def svg_data_uri(svg):
    return "data:image/svg+xml;base64," + b64encode(svg.encode("utf-8")).decode("ascii")


def svg_icon(name):
    # st.html sanitizes inline SVG; an image preserves the vector without scripts.
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#102240" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    return f'<img src="{svg_data_uri(svg)}" width="23" height="23" alt="" aria-hidden="true">'


def panel_title(text, icon=None):
    st.html(f'<div class="panel-title">{svg_icon(icon) if icon else ""}{escape(text)}</div>')


def state_box(title, detail="", tone="", compact=False):
    style = f"state-box {tone} {'compact' if compact else ''}"
    description = f'<p>{escape(detail)}</p>' if detail else ""
    st.html(
        f'<div class="{style}"><strong>{escape(title)}</strong>'
        f'{description}</div>',
    )


def record_preview_alert(result=None, frame=None, simulated=True):
    """Record an event; delivery is handled separately by queue_alert()."""
    thumb = None
    if frame is not None:
        small = cv2.resize(frame, (320, max(1, int(320 * frame.shape[0] / frame.shape[1]))))
        ok, encoded = cv2.imencode(".jpg", small)
        if ok:
            thumb = encoded.tobytes()
    event = {
        "time": time.strftime("%H:%M:%S"),
        "level": result.level.name if result else "TEST",
        "reason": result.reason if result else "Manual preview of a caregiver alert",
        "thumb": thumb,
        "status": "Simulated delivery · No message sent" if simulated else "Sending...",
    }
    ss.preview_history.insert(0, event)
    del ss.preview_history[50:]
    ss.preview_last_alert = time.monotonic()
    return event


def queue_alert(result, frame, room_id):
    """Send a real alert in the background and retain its delivery status."""
    event = record_preview_alert(result, frame, simulated=False)
    event["room"] = rooms.get(room_id, "")

    def done(future):
        try:
            n = future.result()
            event["status"] = f"Sent to {n} Telegram chat(s)" if n else "Telegram delivery failed"
        except Exception:
            event["status"] = "Telegram delivery failed"

    notifier.send_alert(result, frame, room_id, store).add_done_callback(done)


def render_assessment(result=None, error=None, stale=False, idle=False, prefix="live"):
    with st.container(border=True, key=f"preview_{prefix}_assessment"):
        panel_title("Assessment", "assessment")
        if error:
            state_box("Assessment unavailable", "Processing failed. No current danger assessment.", "error")
            with st.expander("Error details"):
                st.text(error)
        elif is_demo:
            if result is None:
                state_box("Simulated assessment", "Start monitoring or analyse an image to generate a test result.")
            else:
                tone = f"severity-{result.level.name.lower()}"
                state_box(f"Simulated level: {result.level.name}", f"Random score: {result.score:.2f}", tone)
        elif idle:
            state_box("Monitoring stopped", "Start the camera to request an assessment.")
        elif stale:
            state_box("Assessment outdated", "Waiting for a fresh result. Previous result is hidden.", "warning")
        elif result is None:
            state_box("Waiting for assessment", "Camera and model readiness are shown separately.")
        else:
            tone = f"severity-{result.level.name.lower()}"
            state_box(result.level.name, result.reason or "No explanation provided.", tone)
            st.caption(f"Danger score: {result.score:.2f}")

    with st.container(border=True, key=f"preview_{prefix}_objects"):
        panel_title("Detected objects", "objects")
        if is_demo:
            state_box("Detector not connected", "Object labels will appear here.", compact=True)
        elif error or stale or idle:
            state_box("Detections unavailable", "Waiting for a current successful assessment.", compact=True)
        elif result is None:
            state_box("Waiting for detections", "Objects will appear after processing.", compact=True)
        elif not result.detections:
            state_box("No objects detected", "An empty result does not establish that the scene is safe.", compact=True)
        else:
            for item in result.detections:
                st.html(
                    f'<div class="object-row"><b>{escape(item.label)}</b><span>{item.confidence:.0%} confidence</span></div>',

                )
    with st.container(border=True, key=f"preview_{prefix}_proximity"):
        panel_title("Proximity assessment", "proximity")
        if is_demo:
            state_box("Waiting for spatial analysis", compact=True)
        elif not spatial_connected:
            state_box("Spatial analysis not connected yet",
                      "The danger level currently comes from the detected objects only.", compact=True)
        elif result is not None and not (error or stale or idle):
            st.write(result.reason or "No spatial explanation provided.")
            st.caption("Distance measurements are not exposed by the current classifier interface.")
        else:
            state_box("Proximity unavailable", "Waiting for current spatial analysis.", compact=True)


@st.cache_resource
def load_camera_controls():
    """Custom footer controls; streamlit-webrtc still owns capture and transport."""
    return st.components.v2.component(
        "preview_camera_controls",
        html="""
        <div class="camera-controls">
          <div class="input-group"><label for="camera-source">Video input:</label>
            <div class="select-wrap">
              <svg class="camera-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/></svg>
              <select id="camera-source" aria-label="Video input"><option value="">Built-in webcam</option></select>
              <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m7 10 5 5 5-5"/></svg>
            </div>
          </div>
          <button id="monitor-toggle" type="button" aria-pressed="false">
            <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><rect class="stop" x="3" y="3" width="14" height="14" rx="2"/><path class="play" d="M5 3v14l12-7z"/></svg>
            <span>Start monitoring</span>
          </button>
        </div>
        """,
        css="""
        * { box-sizing: border-box; }
        .camera-controls { display: flex; align-items: center; justify-content: space-between; gap: 20px; min-height: 50px; font: 15px Arial, sans-serif; color: #263b5d; }
        .input-group { display: flex; align-items: center; gap: 12px; min-width: 0; }
        label { white-space: nowrap; }
        .select-wrap { position: relative; width: 350px; max-width: 100%; }
        select { appearance: none; width: 100%; height: 46px; padding: 0 38px 0 44px; border: 1px solid #cbd7e5; border-radius: 8px; background: #fff; color: #263b5d; font: inherit; cursor: pointer; }
        .camera-icon { position: absolute; left: 15px; top: 12px; width: 22px; height: 22px; pointer-events: none; }
        .chevron { position: absolute; right: 12px; top: 12px; width: 22px; height: 22px; pointer-events: none; }
        button { display: flex; align-items: center; justify-content: center; gap: 13px; min-width: 216px; height: 46px; padding: 0 24px; border: 0; border-radius: 8px; background: #10888e; color: white; font: 600 15px Arial, sans-serif; cursor: pointer; }
        button:hover { background: #08777e; }
        button svg { width: 19px; height: 19px; }
        button[aria-pressed="false"] .stop, button[aria-pressed="true"] .play { display: none; }
        button:focus-visible, select:focus-visible { outline: 2px solid #08777e; outline-offset: 3px; }
        @media (max-width: 600px) { .camera-controls { gap: 10px; flex-wrap: wrap; } .select-wrap { width: 220px; } button { min-width: 180px; } }
        """,
        js="""
        export default function(component) {
          const { parentElement, data, setStateValue } = component;
          const select = parentElement.querySelector('select');
          const button = parentElement.querySelector('button');
          let control = { running: false, device: '', ...data };
          let mounted = true;
          function render() {
            button.setAttribute('aria-pressed', String(control.running));
            button.querySelector('span').textContent = control.running ? 'Stop monitoring' : 'Start monitoring';
            select.value = control.device;
            select.disabled = control.running;
            select.title = control.running ? 'Stop monitoring before changing the video input' : 'Select a browser camera';
          }
          async function devices() {
            if (!navigator.mediaDevices?.enumerateDevices) return;
            try {
              const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
              if (!mounted) return;
              select.replaceChildren(new Option('Built-in webcam', ''));
              inputs.forEach((d, i) => { if (d.deviceId) select.add(new Option(d.label || `Camera ${i + 1}`, d.deviceId)); });
              render();
            } catch (_) { /* The browser's default camera remains available. */ }
          }
          button.onclick = () => { control = { ...control, running: !control.running }; render(); setStateValue('control', control); };
          select.onchange = () => { control = { ...control, device: select.value }; setStateValue('control', control); };
          navigator.mediaDevices?.addEventListener('devicechange', devices);
          render(); devices();
          return () => { mounted = false; navigator.mediaDevices?.removeEventListener('devicechange', devices); };
        }
        """,
    )


# Keep Settings aligned to the native tab row, with page content at full width.
shield_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 38"><path fill="#10888e" d="M16 0 30 6l-2 17-12 15L4 23 2 6z"/><circle fill="white" cx="11" cy="11" r="2"/><path fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" d="m9 22 5 5 10-13"/></svg>'
st.html(f'<div class="brand"><span class="brand-shield"><img src="{svg_data_uri(shield_svg)}" width="23" height="27" alt="Shield logo"></span>Child Safety Monitor</div>')
navigation = st.container(key="preview_navigation")
with navigation:
    with st.container(key="preview_settings"):
        with st.popover("Settings", icon=":material/settings:", use_container_width=True):
            st.subheader("Monitoring settings")
            if model_problem:
                st.error(model_problem)
            elif is_demo:
                st.caption("Demo model (random scores). Put the notebook's `final_best.pt` in "
                           "`weights/` and restart to use the trained YOLOv8 detector.")
            else:
                st.caption(f"Model: YOLOv8 · {os.path.basename(classifier.detector.weights_path)} · "
                           f"{len(classifier.detector.names)} classes")
            quality = st.radio("Camera quality", ["Smooth (480p)", "Sharp (720p)"], horizontal=True,
                               key="preview_quality",
                               help="480p lags less: every frame is sent to the server and back. "
                                    "Applies the next time the camera starts.")
            max_fps = st.slider("Max model runs / second", 1, 15, 5, key="preview_fps")
            show_overlay = st.checkbox("Draw detection overlay", value=True, disabled=is_demo, key="preview_overlay")
            if is_demo:
                st.caption("Random danger banners are hidden from the demo camera.")
            st.divider()
            st.subheader("Alert settings")
            st.caption("Turn alerts on and pick the room under Caregiver alerts on the Live tab.")
            min_level = DangerLevel[st.selectbox("Alert at level ≥", ["LOW", "MEDIUM", "HIGH"], index=2, key="preview_level")]
            confirm_s = st.slider("Level must last (s)", 0.0, 5.0, 1.5, 0.5, key="preview_confirm")
            cooldown = st.number_input("Cooldown between alerts (s)", 5, 3600, 60, key="preview_cooldown")
            if st.button("Simulate test alert", key="preview_manual_alert"):
                record_preview_alert()
                st.toast("Simulated event recorded. No Telegram message sent.")
    tab_live, tab_test, tab_video, tab_history, tab_telegram = st.tabs(
        ["Live Monitoring", "Test Image", "Test Video", "Alert History", "Telegram"],
        key="main_tabs", on_change="rerun")   # lets "Configure Telegram" switch tabs


def open_telegram_tab():
    ss.main_tabs = "Telegram"


def alert_room():
    """This camera's room. Read from widget state so the live fragment can use it."""
    rid = ss.get("preview_room")
    return rid if rid in rooms else None


def render_alerts(result=None, processor=None, eligible=False):
    with st.container(border=True, key="preview_live_alerts"):
        now = time.monotonic()
        room_id = alert_room()
        active = bool(ss.get("preview_auto_alerts")) and room_id is not None
        st.html(f'<div class="alert-heading"><div class="panel-title">{svg_icon("bell")}Caregiver alerts</div><span class="pill">{"On" if active else "Off"}</span></div>')
        recipients = store.recipients(room_id) if room_id else []
        ready = bool(active and eligible and result is not None and notifier.configured and recipients)
        if ready and result.level >= min_level:
            ss.preview_danger_since = ss.preview_danger_since or now
        else:
            ss.preview_danger_since = None
        if not home:
            message = "Set up your home in the Telegram tab."
        elif not room_id:
            message = "Add a room in the Telegram tab, then pick it below."
        elif not active:
            message = "Telegram alerts are off. Turn them on below."
        elif not notifier.configured:
            message = "Telegram is not configured. See the Telegram tab."
        elif not recipients:
            message = "Nobody gets alerts for this room yet. Share the invite link (Telegram tab)."
        elif not ready:
            message = "Waiting for the camera and a fresh assessment."
        elif ss.preview_danger_since is None:
            message = "On. Current result is below the alert threshold."
        elif now - ss.preview_danger_since < confirm_s:
            message = f"Confirming assessment for {confirm_s:g}s."
        elif not notifier.claim_cooldown(room_id, cooldown):     # shared by all viewers of the room
            message = f"Cooldown. Next alert possible in {notifier.cooldown_left(room_id, cooldown):.0f}s."
        else:
            queue_alert(result, processor.latest_frame, room_id)
            message = "Alert queued. Check Alert History for delivery status."
        place = escape(f"{home.name} · {rooms[room_id]}" if room_id else "Not configured")
        st.html(f'<div class="alert-content"><p>{escape(message)}</p><small>Room: {place}<br>{len(recipients)} recipient(s)</small></div>')


with tab_live:
    # Each live-updating part is its own fragment and only draws inside itself;
    # writing into placeholders outside a fragment can blank the page on reruns.
    initial_control = {"running": False, "device": ""}
    control_state = ss.get("preview_camera_footer", {}).get("control", initial_control)
    running = bool(control_state.get("running"))
    st.html(f'<div class="live-heading"><h1>Live Monitoring</h1><span class="{"pill live" if running else "pill"}">'
            f'● {"Camera on" if running else "Camera stopped"}</span></div>')
    left, right = st.columns([1.85, 1], gap="medium")
    with left:
        with st.container(border=True, key="preview_camera_card"):
            st.html(f'<div class="camera-card-header"><strong>Camera Feed</strong><span class="camera-view-tag">'
                    f'{"Live camera view" if running else "Camera preview"}</span></div>')
            video_area = st.container(key="preview_video_frame")
            controls = load_camera_controls()(
                key="preview_camera_footer", data=control_state,
                default={"control": initial_control}, on_control_change=lambda: None,
            )
            control = controls.control or initial_control
            with video_area:
                if not control["running"]:
                    st.html('<div class="camera-placeholder"><div class="camera-symbol"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/></svg></div>'
                            '<strong>Your camera view will appear here</strong>'
                            '<p>Choose a video input and start monitoring.</p></div>')
                # Smaller frames = less to encode/decode on both ends = less lag.
                w, h = (640, 480) if quality.startswith("Smooth") else (1280, 720)
                video_constraint = {"width": {"ideal": w}, "height": {"ideal": h},
                                    "frameRate": {"ideal": 20, "max": 24}}
                if control["device"]:
                    video_constraint["deviceId"] = {"exact": control["device"]}
                ctx = webrtc_streamer(
                    key="preview-camera", mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                    video_processor_factory=make_processor,
                    media_stream_constraints={"video": video_constraint, "audio": False},
                    desired_playing_state=bool(control["running"]), media_toggle_controls=False,
                    async_processing=True,
                    video_html_attrs={"autoPlay": True, "controls": False, "muted": True, "style": {"width": "100%", "height": "calc(100vh - 16px)", "objectFit": "cover", "objectPosition": "center", "backgroundColor": "#e9eef4", "borderRadius": "8px"}},
                )

    def live_snapshot():
        """(processor, playing, result, error, age) from the camera thread."""
        processor = ctx.video_processor
        playing = bool(ctx.state.playing)
        result, error, age = None, None, None
        if processor is not None and playing:
            result, error = processor.latest_result, processor.error
            timestamp = processor.classifier.last_completed_at
            age = time.monotonic() - timestamp if timestamp else None
        return processor, playing, result, error, age

    @st.fragment(run_every=1.0)
    def performance():
        processor, playing, _, _, age = live_snapshot()
        with st.expander("Performance details"):
            a, b, c = st.columns(3)
            a.metric("Model FPS", f"{processor.infer_fps:.1f}" if processor and playing else "—")
            b.metric("Latency", f"{processor.latency_ms:.0f} ms" if processor and playing else "—")
            c.metric("Result age", f"{age:.1f} s" if age is not None else "—")
            st.caption(f"Classifier: {type(classifier).__name__} · Inference limit: {max_fps}/s")

    @st.fragment(run_every=0.5)
    def live_status():
        processor, playing, result, error, age = live_snapshot()
        stale = age is not None and age > 5.0
        if processor is not None:
            processor.max_infer_fps = max_fps
            processor.draw_overlay = bool(show_overlay and not is_demo and playing and result is not None and not error and not stale)
        render_assessment(result, error=error, stale=stale, idle=not playing)
        render_alerts(result, processor, eligible=playing and not error and not stale)

    with left:
        performance()
    with right:
        live_status()
        # Controls sit outside the auto-refreshing fragment so clicks work normally.
        with st.container(border=True, key="preview_alert_controls"):
            if rooms:
                st.selectbox("This camera's room", list(rooms), format_func=rooms.get, key="preview_room")
            auto_alerts = st.toggle("Send Telegram alerts", value=False, key="preview_auto_alerts",
                                    disabled=not rooms)
            if not home:
                st.caption("Create or open your home in the Telegram tab first.")
            elif not rooms:
                st.caption("Add a room in the Telegram tab first.")
            elif is_demo and auto_alerts:
                st.caption("Demo model: alerts use random scores. To see one quickly, set "
                           "'Level must last' to 0 in Settings.")
            st.button("Configure Telegram", icon=":material/send:", type="tertiary",
                      key="preview_configure", on_click=open_telegram_tab)

with tab_test:
    st.title("Test Image")
    st.caption("Try the same classifier on an uploaded image. Image tests do not send alerts.")
    image_col, assessment_col = st.columns([1.85, 1], gap="medium")
    with image_col:
        with st.container(border=True):
            panel_title("Image input")
            uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], key="preview_upload")
            upload_key = uploaded.file_id if uploaded is not None else None
            if ss.preview_image_key != upload_key:
                ss.preview_image_key = upload_key
                ss.preview_image_result = ss.preview_image_frame = ss.preview_image_ms = ss.preview_image_error = None
            if uploaded is not None:
                frame = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("This file could not be read as an image. Try another file.")
                else:
                    if st.button("Analyse image", type="primary", key="preview_analyse"):
                        started = time.perf_counter()
                        try:
                            ss.preview_image_result = classifier.predict(frame)
                            ss.preview_image_frame = frame
                            ss.preview_image_ms = (time.perf_counter() - started) * 1000
                            ss.preview_image_error = None
                        except Exception as exc:
                            ss.preview_image_result = ss.preview_image_frame = ss.preview_image_ms = None
                            ss.preview_image_error = str(exc)
                    shown = frame
                    if ss.preview_image_result is not None and not is_demo and show_overlay:
                        shown = draw_result(frame, ss.preview_image_result)
                    st.image(shown, channels="BGR", use_container_width=True)
                    st.caption("Uploaded image · Demo scores are unrelated to the image" if is_demo else "Uploaded image")
            else:
                state_box("No image selected", "Upload a JPG, PNG, or WebP to test this layout.")
    with assessment_col:
        render_assessment(ss.preview_image_result, error=ss.preview_image_error, prefix="image")
        if ss.preview_image_ms is not None:
            st.caption(f"Processing time: {ss.preview_image_ms:.0f} ms")

with tab_history:
    st.title("Alert History")
    st.caption("Simulated tests are labelled. Real alerts show their Telegram delivery status.")

    @st.fragment(run_every=2)
    def render_history():
        if not ss.preview_history:
            with st.container(border=True):
                state_box("No alerts yet", "Use Simulate test alert in Settings to try this view.")
        else:
            if st.button("Clear preview history", key="preview_clear"):
                ss.preview_history = []
                st.rerun()
            for event in ss.preview_history:
                with st.container(border=True):
                    if event["thumb"]:
                        photo, details = st.columns([1, 3])
                        photo.image(event["thumb"])
                    else:
                        details = st.container()
                    with details:
                        room = f" · {event['room']}" if event.get("room") else ""
                        st.markdown(f"**{event['level']}{room} · {event['time']}**")
                        st.write(event["reason"])
                        st.caption(event["status"])

    render_history()

with tab_video:
    st.title("Test Video")
    st.caption("Run the same classifier over a recorded video. Video tests do not send alerts.")
    video_col, summary_col = st.columns([1.85, 1], gap="medium")
    with video_col:
        with st.container(border=True):
            panel_title("Video input")
            uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv", "webm"],
                                              key="preview_video_upload")
            video_key = uploaded_video.file_id if uploaded_video is not None else None
            if ss.preview_video_key != video_key:
                ss.preview_video_key, ss.preview_video_report = video_key, None
            if uploaded_video is None:
                state_box("No video selected", "Upload a short clip (up to 2 minutes are analysed).")
            else:
                every_n = st.slider("Analyse every Nth frame", 1, 30, 5, key="preview_video_step",
                                    help="Higher is faster; frames in between reuse the last result.")
                if st.button("Analyse video", type="primary", key="preview_video_analyse"):
                    # delete=False: Windows cannot reopen a file that is still open
                    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_video.name)[1],
                                                     delete=False) as tmp:
                        tmp.write(uploaded_video.getvalue())
                    bar = st.progress(0.0, text="Analysing video...")
                    try:
                        ss.preview_video_report = analyse_video(
                            tmp.name, classifier, every_n=every_n,
                            overlay=bool(show_overlay and not is_demo),
                            on_progress=lambda f: bar.progress(f, text="Analysing video..."))
                    except Exception as exc:
                        ss.preview_video_report = None
                        st.error(f"Could not analyse this video: {exc}")
                    finally:
                        os.unlink(tmp.name)
                        bar.empty()
                report = ss.preview_video_report
                if report and report.annotated_mp4:
                    st.video(report.annotated_mp4)
                    st.download_button("Download annotated video", report.annotated_mp4,
                                       file_name="annotated.mp4", mime="video/mp4")
                elif report is None:
                    st.video(uploaded_video)
    with summary_col:
        report = ss.preview_video_report
        if report and report.timeline:
            with st.container(border=True):
                panel_title("Danger over time", "assessment")
                st.line_chart({"Danger score": {round(t, 1): r.score for t, r in report.timeline}},
                              x_label="Seconds", height=180)
                high = [t for t, r in report.timeline if r.level >= min_level]
                st.caption(f"{len(high)} of {len(report.timeline)} analysed frames at "
                           f"{min_level.name} or above · {report.seconds:.0f}s analysed"
                           + (" (video cut at 2 min)" if report.truncated else ""))
            st.caption(f"Most dangerous moment: {report.worst_t:.1f}s")
            if report.worst_frame is not None:
                st.image(draw_result(report.worst_frame, report.worst_result)
                         if show_overlay and not is_demo else report.worst_frame, channels="BGR")
            render_assessment(report.worst_result, prefix="video")
        else:
            render_assessment(None, prefix="video")

with tab_telegram:
    st.title("Telegram")
    st.caption("One home holds all your rooms. Family members join the home once with "
               "one link, then pick which rooms alert them from inside Telegram (/rooms).")
    if not notifier.configured:
        st.warning("No bot token found. Add it to `.streamlit/secrets.toml`, then restart the app.")
        st.code('TELEGRAM_BOT_TOKEN = "your-bot-token"', language="toml")

    if not home:
        create_col, open_col = st.columns(2, gap="medium")
        with create_col, st.container(border=True):
            panel_title("Create a home")
            new_home = st.text_input("Home name", placeholder="Tan family", key="tg_new_home")
            if st.button("Create home", type="primary", disabled=not new_home.strip()):
                created = store.create_home(new_home.strip())
                store.add_room(created.id, "Living room")     # rename/remove later
                st.query_params["home"] = created.manage_key
                st.rerun()
        with open_col, st.container(border=True):
            panel_title("Open my home")
            typed_key = st.text_input("Home key", type="password", key="tg_home_key",
                                      help="Shown on the home page after you create it.")
            if st.button("Open home", disabled=not typed_key.strip()):
                if store.home_by_key(typed_key.strip()):
                    st.query_params["home"] = typed_key.strip()
                    st.rerun()
                else:
                    st.error("No home with this key.")
    else:
        head, close = st.columns([4, 1])
        head.subheader(f"🏠 {home.name}")
        if close.button("Close home", use_container_width=True):
            del st.query_params["home"]
            st.rerun()
        with st.expander("Home key (keep private)"):
            st.code(home.manage_key)
            st.caption("This key opens the home on any device. It is also in this page's URL, "
                       "so bookmarking the page works too.")

        rooms_col, people_col = st.columns(2, gap="medium")
        with rooms_col, st.container(border=True):
            panel_title("Rooms")
            if not rooms:
                state_box("No rooms yet", "Add one for each camera location.", compact=True)
            for rid, name in rooms.items():
                c1, c2 = st.columns([4, 1])
                c1.html(f'<div class="room-row"><b>{escape(name)}</b>'
                        f'<span>{len(store.recipients(rid))} recipient(s)</span></div>')
                if c2.button("Remove", key=f"tg_del_room_{rid}", type="tertiary"):
                    store.delete_room(rid)
                    st.rerun()
            with st.form("tg_add_room", clear_on_submit=True, border=False):
                room_name = st.text_input("New room", placeholder="Bedroom")
                if st.form_submit_button("Add room") and room_name.strip():
                    if store.add_room(home.id, room_name.strip()) is None:
                        st.error("This home already has a room with that name.")
                    else:
                        st.rerun()

        with people_col, st.container(border=True):
            panel_title("Invite & members")
            link = notifier.deep_link(home.invite_code) if notifier.configured else None
            if link:
                st.caption("Send this link to caregivers. They open it and press Start. "
                           "It also works in a family group chat.")
                st.code(link, language=None)
            else:
                st.caption("The invite link appears once the bot token is configured.")
            if st.button("Replace link", help="The old link stops working. Current members stay."):
                store.new_invite(home.id)
                st.rerun()
            members = store.members(home.id)
            st.markdown(f"**{len(members)} member(s)**")
            for chat_id, display in members:
                c1, c2 = st.columns([4, 1])
                c1.html(f'<div class="member-row"><b>{escape(display or str(chat_id))}</b></div>')
                if c2.button("Remove", key=f"tg_del_member_{chat_id}", type="tertiary"):
                    store.leave(home.id, chat_id)
                    st.rerun()

        with st.container(border=True):
            panel_title("Check delivery", "bell")
            test_room = st.selectbox("Room", list(rooms), format_func=rooms.get,
                                     key="tg_test_room") if rooms else None
            t1, t2 = st.columns(2)
            if t1.button("Send test message", disabled=not (notifier.configured and members),
                         use_container_width=True):
                with st.spinner("Sending..."):
                    n = notifier.send_text([c for c, _ in members],
                                           f"✅ Test message from Child Safety Monitor ({home.name})").result()
                (st.success if n else st.error)(f"Delivered to {n} of {len(members)} member(s).")
            if t2.button("Send test alert (photo)", use_container_width=True,
                         disabled=not (notifier.configured and test_room and store.recipients(test_room))):
                processor = ctx.video_processor
                frame = processor.latest_frame if processor is not None else None
                if frame is None:                     # no live camera: send a placeholder card
                    frame = np.full((360, 640, 3), (238, 241, 246), np.uint8)
                    cv2.putText(frame, "Test alert", (210, 190), cv2.FONT_HERSHEY_SIMPLEX,
                                1.4, (64, 34, 16), 3, cv2.LINE_AA)
                fake = ClassificationResult(DangerLevel.HIGH, 1.0, [], "Manual test alert")
                event = record_preview_alert(fake, frame, simulated=False)
                event["room"] = rooms[test_room]
                with st.spinner("Sending..."):
                    n = notifier.send_alert(fake, frame, test_room, store).result()
                event["status"] = f"Sent to {n} Telegram chat(s)" if n else "Telegram delivery failed"
                (st.success if n else st.error)(f"Test alert delivered to {n} chat(s).")
            st.caption("Members who muted a room in Telegram don't receive its alerts.")

        with st.expander("Delete home"):
            st.caption("Removes all rooms and members. Invite links stop working.")
            if st.button("Delete this home", type="primary"):
                store.delete_home(home.id)
                del st.query_params["home"]
                st.rerun()

st.html(f'<div class="footer">Child Safety Monitor{" · Demo mode: assessments are simulated." if is_demo else ""}</div>')
