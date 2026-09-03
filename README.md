# Motion-Controlled Ball Sort (Hand-Landmark Detection Game)

A quick note on terminology first, since it affects how you should describe this
project: the computer-vision piece here is **hand landmark detection**
(MediaPipe's `HandLandmarker`), not classic bounding-box "object detection"
(like YOLO drawing a box around a "cup" or "car"). It's in the same family —
both are computer-vision models that locate something in a video frame — but
landmark detection returns 21 keypoints per hand (fingertips, knuckles,
wrist) instead of a box + class label. If your assignment specifically
requires bounding-box object detection, say so and I'll adapt this (e.g. add
a YOLO/MediaPipe Object Detector stage that finds the hand first). If
"detecting objects in a video stream with a pretrained model" is close
enough, this project already satisfies that.

## What it does

1. `frontend/ball_sort_game_frontend.py` opens your webcam, runs MediaPipe's
   `HandLandmarker` on each frame (asynchronously, so the game never
   freezes), and tracks your thumb + index fingertip. Pinching them together
   "grabs" a ball; releasing "drops" it into whichever tube your fingertip is
   hovering over. It's a full ball-sort puzzle game (10 levels) with a
   pygame UI, name entry, and a small webcam preview overlay.
2. `backend/backend.py` is a Flask server with two endpoints
   (`/add_score`, `/get_leaderboard`) that stores a top-5 leaderboard in a
   local `leaderboard.json` file. The frontend posts your score to it
   whenever you finish a level.

## 1. Install dependencies

Two separate processes, so it's cleanest to use two terminals (a venv per
folder is optional but recommended):

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt

# Terminal 2 — frontend
cd frontend
pip install -r requirements.txt
```

## 2. Download the hand-landmark model

The frontend needs `hand_landmarker.task` sitting in the **same folder** as
`ball_sort_game_frontend.py` (i.e. `frontend/hand_landmarker.task`). Download
it from the URL you already have:

```bash
cd frontend
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

(No `curl`? Just paste the URL into a browser — it downloads the file
directly — and move it into `frontend/`.)

## 3. Run it

```bash
# Terminal 1
cd backend
python3 backend.py
# -> "Ball Sort leaderboard server starting on port 5050 ..."

# Terminal 2
cd frontend
python3 ball_sort_game_frontend.py
```

A pygame window opens, asks for your name, then shows the level-select menu.
Make sure your webcam has a clear view of your hand.

**Controls**
- Pinch thumb + index finger to select a level, and to grab/drop balls.
- `R` reset level · `M` level menu · `N` next level (after solving) · `Esc` quit.

If the frontend can't reach the backend (e.g. you only ran one terminal),
the game still works — it just shows "Could not reach the leaderboard
server" instead of saving your score.

## Running frontend and backend on separate machines

If the backend runs on another device (e.g. a Jetson Nano) on your network,
change this one line near the top of `ball_sort_game_frontend.py`:

```python
BACKEND_URL = "http://127.0.0.1:5050"   # -> "http://<jetson-ip>:5050"
```

## Troubleshooting

- **"Error: Laptop webcam not found."** — no camera detected at index 0;
  check `CameraStream(index=0, ...)` near the top of the frontend file and
  try `index=1` if you have multiple cameras.
- **"Missing 'hand_landmarker.task' file."** — you skipped step 2, or it's
  in the wrong folder.
- **Backend says "Address already in use" on port 5000 (macOS)** — Control
  Center's AirPlay Receiver uses port 5000 by default. This project now
  runs the backend on port 5050 instead to sidestep that entirely; if you
  see this on an older copy of the files, either disable AirPlay Receiver
  (System Settings → General → AirDrop & Handoff) or change the port.
- **Pinch detection feels off** — tune `PINCH_RATIO` (smaller = tighter
  pinch required) or `CURSOR_SMOOTHING` (higher = snappier cursor) near the
  top of the frontend file.

## Performance notes

Already applied in this version of `ball_sort_game_frontend.py`:

- **Tube & menu-button backgrounds are cached, not redrawn every frame.**
  The old code allocated a new per-pixel-alpha `Surface` for every tube
  (up to 13 of them) and every menu button (10), every single frame at
  60fps, plus re-rendering 3 fonts per menu button every frame. All of that
  is now built once and reused — this was the single biggest render-loop
  cost.
- **Detection is rate-limited to the model's actual speed.** A new frame is
  only handed to `detect_async` once the previous inference has returned.
  Without this, a slower device queues up frames faster than it can
  process them and the hand cursor lags progressively further behind the
  longer you play.
- **GPU delegate is tried first, with automatic CPU fallback.** On a
  Jetson Nano especially, running hand-landmark inference on the onboard
  GPU instead of the CPU is the biggest lever you have — the code now
  attempts it automatically and prints which one it's using.
- **The webcam thumbnail reuses one `Surface`** instead of allocating a new
  one every frame.
- Flask backend now runs with `threaded=True` so a score submit and a
  leaderboard fetch happening close together don't block each other.

Worth trying yourself, since they trade something off (quality, accuracy,
or your own testing time) so I didn't change them for you:

- **Lower `CAM_WIDTH`/`CAM_HEIGHT`** (currently 640x480) if your device
  struggles with camera capture itself, not just detection — e.g. 480x360.
  Detection already runs on a separate, smaller `DET_WIDTH`/`DET_HEIGHT`
  (320x240) copy, so this mainly helps capture/resize cost, not accuracy.
- **Raise `min_hand_detection_confidence` / `min_tracking_confidence`**
  (currently 0.6) if you get flickery false-positive detections in bad
  lighting — fewer re-detections from scratch = smoother tracking, at the
  cost of being pickier about hand visibility.
- **Cap the frame rate lower than 60** (`clock.tick(60)` near the game
  loop) if you're running on a device where the CPU is the bottleneck for
  *rendering*, not detection — pygame's software renderer is single
  threaded, so 30fps can look identical in practice while using half the
  CPU.
- If you ever profile and find `pygame.draw.lines` (tube outlines) or the
  gradient background is a hotspot, those can be cached the same way the
  glass surfaces now are — I left them as-is since they're much cheaper
  operations than what was removed above.

## Note on the extra `ballsort.zip`

Your `ballsort.zip` contained an earlier version of the same frontend file
without the leaderboard/name-entry code — I used the more complete version
from `ballsortbackend.zip` for this project. If you actually wanted the
simpler standalone version (no backend needed at all), let me know.
