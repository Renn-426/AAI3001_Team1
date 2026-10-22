# Child Safety Monitor – UI base

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
app.py                  # Streamlit UI (camera, status, sidebar, telegram hook)
modules/
  camera.py             # webrtc live camera + per-frame processing + overlay
  classifier.py         # DangerLevel, BaseClassifier, DummyClassifier  <-- plug models here
  notifier.py           # TelegramNotifier (text + photo, cooldown)
```

## Telegram secrets
Create `.streamlit/secrets.toml` (do NOT commit it; add to .gitignore):
```
TELEGRAM_BOT_TOKEN = "123456:ABC..."
TELEGRAM_CHAT_ID = "123456789"
```
On Streamlit Community Cloud, paste the same content in App settings → Secrets.

## Integration points for teammates
- **Spatial detection / modelling**: subclass `BaseClassifier` in `modules/classifier.py`,
  return a `ClassificationResult` (level, score, list of `Detection` with bbox, reason),
  then return it from `get_classifier()`.
- **Data augmentation**: only affects training, no UI change.
- **Enable alerts**: uncomment the `notifier.send_alert(...)` lines in `app.py`.

## Cloud
Camera runs in the viewer's browser through WebRTC, so it works when deployed.
Streamlit Community Cloud: push to GitHub, point to `app.py`. Heavy models may need
a bigger host (e.g. Hugging Face Spaces, Render, or a VM). If the video doesn't connect
behind strict networks, add a TURN server to `RTC_CONFIG` in `app.py`.
