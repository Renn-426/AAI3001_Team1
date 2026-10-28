# Child Safety Monitor

Streamlit UI for live camera monitoring, image testing, and Telegram alerts.
The app currently uses `DummyClassifier`, which generates random danger scores;
real detection and spatial risk assessment are not connected yet.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The improved UI is in `app.py`: live camera controls, image and video testing, and four
separate status cards for Assessment, Detected objects, Proximity assessment,
and Caregiver alerts. Simulated results are shown directly with severity colours.
The Settings button is on the navigation row. Automatic Telegram alerts are
disabled while `DummyClassifier` is selected; manual test messages are available.

The training notebook currently targets Google Colab and Google Drive. Its
`google.colab` imports and dataset/output paths need adapting for local training.
The single `requirements.txt` includes both UI and training dependencies.

## Files and ownership

| File | Purpose | Owners |
|---|---|---|
| `app.py` | Improved UI, camera controls, four status cards, image & video tester, Telegram setup, Settings, and alert history | Liv & Ren |
| `modules/camera.py` | Browser camera frames, background inference, overlays, and video analysis | Liv & Ren |
| `modules/notifier.py` | Telegram send queue (rate limits, photo reuse) and per-room cooldown | Liv & Ren |
| `modules/subscriptions.py` | Homes, rooms, members (SQLite) and bot commands | Liv & Ren |
| `modules/detector.py` | Detection interface and unfinished YOLO model skeleton | Tze |
| `modules/spatial.py` | Spatial interface and unfinished proximity/risk skeleton | Ayumi & Nic |
| `modules/classifier.py` | Shared result types and pipeline integration | Shared |
| `Child_Safety_Hazard_Detection_YOLOv8.ipynb` | Augmentation, training, and evaluation | Tze |

## Integration

```text
BGR image -> detector.detect() -> spatial.assess() -> ClassificationResult -> UI/alerts
```

**Tze:** implement `YOLODetector` in `modules/detector.py`. Load weights once in
`__init__()`. `detect(frame_bgr)` accepts an OpenCV BGR image and returns a
`list[Detection]`, with labels, confidence in `[0, 1]`, and bounding boxes
`(x1, y1, x2, y2)` in the original image's pixels. Keep augmentation in training.

**Ayumi & Nic:** implement `SpatialAnalyzer.assess(detections, frame_shape)` in
`modules/spatial.py`. `frame_shape` is `(height, width)`. Return
`(danger_score, reason)`, where the score is in `[0, 1]` and the reason explains
the risk. Define proximity calculations and behaviour when children, hazards,
or bounding boxes are missing. Image distance alone is not physical distance.

Agree on class labels and how children are identified before integration.
A person detection alone does not identify a child. Detection confidence and
danger score are separate values.

**Shared:** `SafetyClassifier` already connects these interfaces and converts
the score to `SAFE`, `LOW`, `MEDIUM`, or `HIGH`. The UI continues to call
`predict(frame_bgr) -> ClassificationResult`, containing `level`, `score`,
`detections`, and `reason`.

Once both implementations are ready, replace the body of `get_classifier()`
in `modules/classifier.py` with:

```python
from modules.detector import YOLODetector
from modules.spatial import SpatialAnalyzer

return SafetyClassifier(
    detector=YOLODetector("path/to/final_best.pt"),
    spatial=SpatialAnalyzer(),
)
```

The skeletons intentionally raise `NotImplementedError` until implemented.
Keep `DummyClassifier` selected while building the UI. After switching, use
the **Test image** tab to check boxes, scores, and explanations.

## Telegram alerts

Create `.streamlit/secrets.toml` (already ignored by Git):

```toml
TELEGRAM_BOT_TOKEN = "your-bot-token"
```

How recipients are organised:

```text
Home (one household, private home key)
 ├─ Rooms:   Living room, Bedroom, ...   <- each camera picks its room in Settings
 └─ Members: Telegram chats that joined with the home's invite link
             (a parent, or a family group chat)
```

1. Open the **Telegram** tab and create a home. Its private key is put in the
   page URL, so bookmark the page (or keep the key) to reopen it.
2. Add one room per camera location.
3. Send the invite link to caregivers. One tap joins **all** rooms of the home.
4. On **Live Monitoring**, under Caregiver alerts, pick this camera's room and
   switch on **Send Telegram alerts**.
5. Use **Send test message** / **Send test alert** in the Telegram tab to check delivery.

In Telegram, members can use `/rooms` to mute or unmute single rooms, or leave a
home, and `/stop` to leave everything. One person can join several homes
(e.g. their own and a grandparent's) and gets alerts labelled `Home · Room`.

Built for many users:
- every browser only sees its own home; nobody can see or delete other homes;
- subscriptions are in SQLite (`subscribers.db`), safe with many users and threads;
- alerts go through one queue that stays under Telegram's rate limit and retries
  when Telegram asks to slow down; a photo is uploaded once and re-used for all members;
- chats that blocked the bot are removed automatically;
- the alert cooldown is per room and shared, so two open browsers don't double-alert.

With `DummyClassifier`, alerts can be switched on for testing but use random
scores (set **Level must last** to 0 in Settings to trigger one quickly).
`.streamlit/config.toml` forces the light theme so text stays readable. Do not commit bot secrets.

## Deployment

WebRTC uses the viewer's browser camera. For Streamlit Community Cloud, point
the deployment to `app.py` and enter the bot token in App settings -> Secrets.
Model hosting requirements depend on the selected model. Restrictive networks
may require a TURN server in `RTC_CONFIG` in `app.py`.

Subscriptions are stored in local `subscribers.db` (SQLite). Hosts that wipe the disk
on restart (e.g. Streamlit Community Cloud) need persistent storage or a hosted
database (Postgres/Supabase) behind the same `SubscriptionStore` methods. Only one
running server may poll the bot; with several servers, switch the bot to a webhook.
