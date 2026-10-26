"""Child Safety Monitor: python -m streamlit run app.py.

Uses the shared classifier pipeline. Dummy predictions remain labelled as
simulated, and automatic Telegram alerts require a real classifier.
"""
from __future__ import annotations

from base64 import b64encode
from html import escape
import threading
import time

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from modules.camera import CameraProcessor, draw_result
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
  --nav-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='3' width='14' height='18' rx='2'/%3E%3Cpath d='M9 7h6M9 11h6M9 15h4'/%3E%3C/svg%3E");
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
def load_classifier() -> tuple[BaseClassifier, bool]:
    model = get_classifier()
    return model, isinstance(model, DummyClassifier)


@st.cache_resource
def load_services():
    notifier, store = TelegramNotifier(), SubscriptionStore()
    if notifier.token:
        start_poller(notifier.token, store)
    return notifier, store


classifier, is_demo = load_classifier()
notifier, store = load_services()
ss = st.session_state
for name, default in {
    "preview_history": [], "preview_danger_since": None,
    "preview_last_alert": 0.0, "preview_image_result": None,
    "preview_image_frame": None, "preview_image_ms": None,
    "preview_image_key": None, "preview_image_error": None,
}.items():
    ss.setdefault(name, default)


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


def queue_alert(result, frame, camera_code):
    """Send a real alert in the background and retain its delivery status."""
    event = record_preview_alert(result, frame, simulated=False)

    def send():
        try:
            sent = notifier.send_alert(result, frame, camera_code, store, force=True)
            event["status"] = "Sent to Telegram" if sent else "Telegram delivery failed"
        except Exception:
            event["status"] = "Telegram delivery failed"

    threading.Thread(target=send, daemon=True).start()


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


@st.cache_resource
def load_settings_opener():
    return st.components.v2.component(
        "preview_telegram_link",
        html='<button type="button"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m22 2-7 20-4-9-9-4 20-7ZM22 2 11 13"/></svg><span>Configure Telegram</span></button>',
        css='button { display:flex; align-items:center; gap:10px; border:0; background:transparent; color:#10888e; font:14px Arial,sans-serif; padding:2px 0; cursor:pointer; } span { text-decoration:underline; text-underline-offset:3px; } button:focus-visible { outline:2px solid #10888e; outline-offset:3px; }',
        js="""
        export default function({ parentElement }) {
          parentElement.querySelector('button').onclick = () => {
            const button = document.querySelector('.st-key-preview_settings [data-testid="stPopoverButton"]');
            if (button && button.getAttribute('aria-expanded') === 'false') button.click();
          };
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
            st.caption(f"Classifier: {type(classifier).__name__}")
            max_fps = st.slider("Max model runs / second", 1, 15, 5, key="preview_fps")
            show_overlay = st.checkbox("Draw detection overlay", value=True, disabled=is_demo, key="preview_overlay")
            if is_demo:
                st.caption("Random danger banners are hidden from the demo camera.")
            st.divider()
            st.subheader("Alert settings")
            auto_alerts = st.toggle("Enable automatic alerts", value=False, disabled=is_demo, key="preview_auto_alerts")
            st.caption("Automatic alerts require a real classifier and subscribed Telegram recipients.")
            min_level = DangerLevel[st.selectbox("Alert at level ≥", ["LOW", "MEDIUM", "HIGH"], index=2, key="preview_level")]
            confirm_s = st.slider("Level must last (s)", 0.0, 5.0, 1.5, 0.5, key="preview_confirm")
            cooldown = st.number_input("Cooldown between alerts (s)", 5, 3600, 60, key="preview_cooldown")
            st.divider()
            st.subheader("Telegram configuration")
            st.caption("Bot token found" if notifier.configured else "Bot token not configured")
            cameras = store.cameras()
            camera_code = st.selectbox("Recipient group", list(cameras), format_func=lambda code: cameras[code], key="preview_camera") if cameras else None
            if not cameras:
                st.caption("No recipient groups configured in the existing app.")
            with st.expander("New camera / invite link"):
                new_name = st.text_input("Camera name", key="camera_name", placeholder="Living room")
                if st.button("Generate link", disabled=not (notifier.configured and new_name.strip())):
                    code = store.create_camera(new_name.strip())
                    ss["camera_invite_link"] = notifier.deep_link(code)
                    st.rerun()
                if ss.get("camera_invite_link"):
                    st.caption("Open this link in Telegram and press Start to subscribe.")
                    st.code(ss.camera_invite_link)
            if st.button("Test message", disabled=not (notifier.configured and camera_code)):
                try:
                    sent = notifier.broadcast_text(store, camera_code, "Test message from Child Safety Monitor")
                    st.toast(f"Sent to {sent} subscriber(s)" if sent else "No subscribers, or sending failed")
                except Exception:
                    st.error("Could not send the test message. Check the Telegram configuration.")
            if camera_code and st.button("Revoke camera"):
                store.delete_camera(camera_code)
                ss.pop("camera_invite_link", None)
                st.rerun()
            st.code('TELEGRAM_BOT_TOKEN = "your-bot-token"', language="toml")
            st.caption("Token location: .streamlit/secrets.toml. Never commit this file.")
            if st.button("Simulate test alert", key="preview_manual_alert"):
                record_preview_alert()
                st.toast("Simulated event recorded. No Telegram message sent.")
    tab_live, tab_test, tab_history = st.tabs(["Live Monitoring", "Test Image", "Alert History"])


def render_alerts(result=None, frame=None, eligible=False):
    with st.container(border=True, key="preview_live_alerts"):
        now = time.monotonic()
        active = auto_alerts and not is_demo
        st.html(f'<div class="alert-heading"><div class="panel-title">{svg_icon("bell")}Caregiver alerts</div><span class="pill">{"On" if active else "Off"}</span></div>')
        ready = bool(active and eligible and result is not None and notifier.configured and camera_code and store.chats_for(camera_code))
        if ready and result.level >= min_level:
            ss.preview_danger_since = ss.preview_danger_since or now
        else:
            ss.preview_danger_since = None
        if is_demo:
            message = "Automatic alerts are disabled."
        elif not active:
            message = "Automatic alerts are disabled."
        elif not notifier.configured:
            message = "Telegram is not configured. Open Settings to add the bot token."
        elif not camera_code:
            message = "Select a recipient group in Settings."
        elif not store.chats_for(camera_code):
            message = "No subscribers yet. Open the camera invite link in Telegram."
        elif not ready:
            message = "Paused. Waiting for a fresh assessment."
        elif ss.preview_danger_since is None:
            message = "Enabled. Current result is below the alert threshold."
        elif now - ss.preview_danger_since < confirm_s:
            message = f"Confirming assessment for {confirm_s:g}s."
        elif now - ss.preview_last_alert < cooldown:
            message = "Cooldown. An alert was recently recorded."
        else:
            queue_alert(result, frame, camera_code)
            message = "Alert queued. Check Alert History for delivery status."
        group = escape(cameras.get(camera_code, "Not configured"))
        count = len(store.chats_for(camera_code)) if camera_code else 0
        st.html(f'<div class="alert-content"><p>{escape(message)}</p><small>Recipient group: {group}<br>{count} subscriber(s)</small></div>')
        load_settings_opener()(key="preview_configure", height=28)


with tab_live:
    camera_status = st.empty()
    left, right = st.columns([1.85, 1], gap="medium")
    with left:
        with st.container(border=True, key="preview_camera_card"):
            camera_header = st.empty()
            video_area = st.container(key="preview_video_frame")
            initial_control = {"running": False, "device": ""}
            control_state = ss.get("preview_camera_footer", {}).get("control", initial_control)
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
                video_constraint = {"deviceId": {"exact": control["device"]}} if control["device"] else True
                ctx = webrtc_streamer(
                    key="preview-camera", mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                    video_processor_factory=make_processor,
                    media_stream_constraints={"video": video_constraint, "audio": False},
                    desired_playing_state=bool(control["running"]), media_toggle_controls=False,
                    async_processing=True,
                    video_html_attrs={"autoPlay": True, "controls": False, "muted": True, "style": {"width": "100%", "height": "calc(100vh - 16px)", "objectFit": "cover", "objectPosition": "center", "backgroundColor": "#e9eef4", "borderRadius": "8px"}},
                )
        performance = st.empty()

    @st.fragment(run_every=0.5)
    def live_status():
        processor = ctx.video_processor
        playing = bool(ctx.state.playing)
        result, error, frame, age = None, None, None, None
        if processor is not None:
            processor.max_infer_fps = max_fps
        if processor is not None and playing:
            result, error, frame = processor.latest_result, processor.error, processor.latest_frame
            timestamp = processor.classifier.last_completed_at
            age = time.monotonic() - timestamp if timestamp else None
        # Placeholders outside a fragment are replaced, so updates do not accumulate.
        view = "Live camera view" if playing and frame is not None else "Connecting camera" if control["running"] else "Camera preview"
        camera_header.html(f'<div class="camera-card-header"><strong>Camera Feed</strong><span class="camera-view-tag">{view}</span></div>')
        status = "Camera streaming" if playing and frame is not None else "Camera connecting" if control["running"] else "Camera stopped"
        status_class = "pill live" if playing and frame is not None else "pill"
        camera_status.html(f'<div class="live-heading"><h1>Live Monitoring</h1><span class="{status_class}">● {status}</span></div>')
        stale = age is not None and age > 5.0
        if processor is not None:
            processor.draw_overlay = bool(show_overlay and not is_demo and playing and result is not None and not error and not stale)
        render_assessment(result, error=error, stale=stale, idle=not playing)
        render_alerts(result, frame, eligible=playing and not error and not stale)
        with performance.container():
            with st.expander("Performance details"):
                a, b, c = st.columns(3)
                a.metric("Model FPS", f"{processor.infer_fps:.1f}" if processor and playing else "—")
                b.metric("Latency", f"{processor.latency_ms:.0f} ms" if processor and playing else "—")
                c.metric("Result age", f"{age:.1f} s" if age is not None else "—")
                st.caption(f"Classifier: {type(classifier).__name__} · Inference limit: {max_fps}/s")

    with right:
        live_status()

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
                        st.markdown(f"**{event['level']} · {event['time']}**")
                        st.write(event["reason"])
                        st.caption(event["status"])

    render_history()

st.html(f'<div class="footer">Child Safety Monitor{" · Demo mode: assessments are simulated; automatic alerts are disabled." if is_demo else ""}</div>')
