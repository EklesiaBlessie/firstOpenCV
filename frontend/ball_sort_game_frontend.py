import cv2
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import sys
import random
import os
import time
import threading
import numpy as np
import requests
import gc

# ============================================================
#  CONFIG
# ============================================================
WIDTH, HEIGHT = 1000, 700
CAM_WIDTH, CAM_HEIGHT = 640, 480
DET_WIDTH, DET_HEIGHT = 320, 240        # downscaled frame fed to the model (speed)
THUMB_WIDTH, THUMB_HEIGHT = 136, 102    # compact webcam preview thumbnail size

TUBE_CAPACITY = 4                       # balls per color / per full tube
EMPTY_TUBES = 2                         # spare empty tubes per level (classic ball-sort difficulty)
PINCH_START_RATIO = 0.35  # Threshold to engage pinch (in isotropic camera space)
PINCH_END_RATIO = 0.48    # Threshold to release pinch (hysteresis deadband: 0.35 - 0.48)
DROP_ANIM_MS = 180
LEVEL_INTRO_ANIM_MS = 420   # staggered ball "drop-in" animation duration when a level loads
CONFETTI_COUNT = 90         # celebratory particles spawned once when a level is won

# Fraction of the camera frame, on each side, treated as an unreliable "dead" margin.
# Hand-tracking confidence drops sharply near the true edge of the webcam's field of
# view (fingers partially out of frame, motion blur, etc). Mapping raw normalized
# coordinates 1:1 onto the screen means reaching the screen's edges requires pushing
# the hand all the way to that unreliable margin - which is exactly why tubes placed
# near the screen edges (the last couple of tubes, once a level's layout widens to
# fill the screen from level 4 onward) were hard or impossible to grab/drop into.
# Remapping compresses the *reliable* portion of the frame to cover the full screen.
CURSOR_EDGE_MARGIN = 0.12

BACKEND_URL = "http://127.0.0.1:5050"   # change to the Jetson's IP if the leaderboard
                                         # server runs on a different device

THEMES = {
    "galaxy": {
        "name": "Galaxy",
        "bg_top": (7, 6, 20),
        "bg_bottom": (18, 14, 40),
        "accent": (45, 230, 255),           # Bright electric cyan
        "accent_secondary": (195, 80, 255), # Electric neon purple
        "tube_glass": (40, 70, 160, 50),
        "tube_border": (65, 140, 230),
        "tube_border_active": (45, 230, 255),
        "shadow_color": (4, 3, 10),
        "text_color": (235, 245, 255),
        "win_color": (50, 245, 190),
        "has_stars": True,
        "ball_colors": {
            "cyan":     (0, 240, 255),
            "magenta":  (255, 60, 180),
            "yellow":   (255, 225, 45),
            "lime":     (120, 255, 80),
            "orange":   (255, 135, 30),
            "purple":   (175, 95, 255),
            "blue":     (55, 150, 255),
            "coral":    (255, 85, 105),
            "emerald":  (35, 225, 150),
            "amber":    (255, 190, 50),
            "pink":     (255, 130, 220),
            "azure":    (80, 200, 255),
        }
    },
    "nebula": {
        "name": "Nebula",
        "bg_top": (16, 13, 34),
        "bg_bottom": (36, 22, 58),
        "accent": (255, 220, 90),          # Warm amber/gold
        "accent_secondary": (232, 64, 148),
        "tube_glass": (140, 165, 235, 55),
        "tube_border": (140, 170, 235),
        "tube_border_active": (255, 220, 90),
        "shadow_color": (6, 5, 14),
        "text_color": (225, 225, 240),
        "win_color": (110, 235, 150),
        "has_stars": False,
        "ball_colors": {
            "magenta": (232, 64, 148),
            "yellow":  (255, 205, 30),
            "blue":    (46, 143, 255),
            "green":   (72, 200, 120),
            "orange":  (255, 130, 40),
            "cyan":    (40, 220, 235),
            "purple":  (150, 80, 220),
            "red":     (230, 60, 60),
            "lime":    (170, 220, 50),
            "pink":    (255, 140, 190),
            "teal":    (30, 160, 150),
            "gold":    (210, 170, 60),
        }
    },
    "cyber": {
        "name": "Cyber",
        "bg_top": (8, 16, 24),
        "bg_bottom": (16, 30, 44),
        "accent": (50, 255, 160),          # Neon mint
        "accent_secondary": (255, 235, 60),
        "tube_glass": (40, 180, 170, 50),
        "tube_border": (50, 200, 180),
        "tube_border_active": (50, 255, 160),
        "shadow_color": (4, 10, 16),
        "text_color": (225, 250, 245),
        "win_color": (50, 255, 160),
        "has_stars": False,
        "ball_colors": {
            "neon_green":  (45, 255, 125),
            "neon_cyan":   (30, 235, 255),
            "neon_yellow": (255, 240, 40),
            "neon_pink":   (255, 55, 165),
            "neon_orange": (255, 125, 30),
            "electric_bl": (50, 140, 255),
            "neon_purple": (185, 75, 255),
            "neon_red":    (255, 65, 80),
            "lime_glow":   (190, 255, 45),
            "ice_blue":    (130, 225, 255),
            "amber_glow":  (255, 175, 45),
            "hot_violet":  (225, 60, 240),
        }
    },
    "sunset": {
        "name": "Sunset",
        "bg_top": (28, 12, 22),
        "bg_bottom": (54, 20, 28),
        "accent": (255, 165, 55),          # Sunset gold
        "accent_secondary": (255, 85, 85),
        "tube_glass": (230, 140, 120, 50),
        "tube_border": (230, 130, 110),
        "tube_border_active": (255, 185, 70),
        "shadow_color": (14, 5, 10),
        "text_color": (255, 235, 230),
        "win_color": (255, 210, 75),
        "has_stars": False,
        "ball_colors": {
            "crimson":    (240, 55, 75),
            "sun_gold":   (255, 195, 35),
            "tangerine":  (255, 120, 30),
            "rose":       (245, 85, 150),
            "peach":      (255, 160, 120),
            "coral":      (255, 95, 95),
            "violet":     (170, 75, 200),
            "sky_amber":  (255, 220, 75),
            "deep_orange":(235, 85, 25),
            "blush":      (255, 140, 175),
            "dusk_blue":  (85, 135, 230),
            "warm_white": (255, 245, 230),
        }
    }
}

current_theme_key = "galaxy"
_t = THEMES[current_theme_key]
BACKGROUND_TOP = _t["bg_top"]
BACKGROUND_BOTTOM = _t["bg_bottom"]
ACCENT = _t["accent"]
ACCENT_SECONDARY = _t["accent_secondary"]
TUBE_GLASS = _t["tube_glass"]
TUBE_BORDER = _t["tube_border"]
TUBE_BORDER_ACTIVE = _t["tube_border_active"]
SHADOW_COLOR = _t["shadow_color"]
TEXT_COLOR = _t["text_color"]
WIN_COLOR = _t["win_color"]
BALL_COLORS = _t["ball_colors"]
COLOR_LIST = list(BALL_COLORS.values())

# ------------------------------------------------------------
#  LEVELS  — level 1 (easiest) -> level 10 (hardest)
#  tubes grows 5 -> 13, colors = tubes - EMPTY_TUBES (capped to palette size)
# ------------------------------------------------------------
LEVEL_TUBE_COUNTS = [5, 7, 9, 10, 11, 11, 12, 12, 13, 13]
LEVELS = []
for _tubes in LEVEL_TUBE_COUNTS:
    _colors = min(_tubes - EMPTY_TUBES, len(COLOR_LIST))
    LEVELS.append({"tubes": _tubes, "colors": _colors})

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
if not os.path.exists(model_path):
    print(f"Error: Missing '{model_path}' file.")
    print("Please download it and place it in the same directory as this script.")
    sys.exit()


# ============================================================
#  THREADED CAMERA CAPTURE  (removes camera-read blocking from the game loop)
# ============================================================
class CameraStream:
    def __init__(self, index=0, width=CAM_WIDTH, height=CAM_HEIGHT):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            print("Error: Laptop webcam not found.")
            sys.exit()
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


# ============================================================
#  ASYNC HAND LANDMARKER  (LIVE_STREAM mode -> never blocks the render loop)
# ============================================================
_result_lock = threading.Lock()
_latest_landmarks = None
_detection_busy = False
_det_buffer = np.zeros((DET_HEIGHT, DET_WIDTH, 4), dtype=np.uint8, order='C')


class _Point:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


_cb_count = 0


def _on_result(result, output_image, timestamp_ms):
    global _latest_landmarks, _detection_busy, _cb_count
    with _result_lock:
        if result and result.hand_landmarks:
            _latest_landmarks = [[_Point(lm.x, lm.y, lm.z) for lm in hand] for hand in result.hand_landmarks]
        else:
            _latest_landmarks = None
        _detection_busy = False
    del result, output_image
    _cb_count += 1
    # Forcing a *full* collection every 100 callbacks (roughly every couple of
    # seconds) was itself a source of visible stutter - it briefly stops the
    # world right in the middle of gameplay. A young-generation-only sweep,
    # done far less often, reclaims the same short-lived detection objects
    # without the full-heap pause.
    if _cb_count % 1000 == 0:
        gc.collect(0)


def _make_hand_options(delegate):
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path, delegate=delegate),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_on_result,
    )


_delegate = python.BaseOptions.Delegate.GPU
_last_detector_refresh = time.time()
_refreshing_detector = False


def refresh_detector():
    """Rebuilds the hand-tracking model. Loading a model from disk can take a
    noticeable chunk of a second (much more on a Jetson Nano's CPU fallback),
    and this used to run directly on the render/game-loop thread - which meant
    the whole game visibly froze for a moment every time a level loaded and
    again every ~75s during play. It now runs on a background thread so the
    game keeps rendering at full speed while the swap happens invisibly."""
    global _refreshing_detector
    if _refreshing_detector:
        return
    _refreshing_detector = True

    def worker():
        global detector, _last_detector_refresh, _refreshing_detector
        try:
            new_detector = vision.HandLandmarker.create_from_options(_make_hand_options(_delegate))
            with _result_lock:
                old_detector = detector
                detector = new_detector
            old_detector.close()
            _last_detector_refresh = time.time()
        except Exception as e:
            print(f"[detector] Refresh error: {e}")
        finally:
            _refreshing_detector = False

    threading.Thread(target=worker, daemon=True).start()


try:
    # GPU delegate is a large speedup where it's supported (e.g. Jetson
    # Nano's onboard GPU) - inference moves off the CPU entirely, which is
    # what actually gates the framerate on that kind of device.
    detector = vision.HandLandmarker.create_from_options(
        _make_hand_options(_delegate))
    print("Hand landmarker: using GPU delegate.")
except Exception as e:
    print(f"GPU delegate unavailable ({e}), falling back to CPU.")
    _delegate = python.BaseOptions.Delegate.CPU
    detector = vision.HandLandmarker.create_from_options(
        _make_hand_options(_delegate))
camera = CameraStream()

# ============================================================
#  PYGAME SETUP
# ============================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Motion Controlled Ball Sort")
clock = pygame.time.Clock()

FONT_TITLE = pygame.font.SysFont("segoeuisemibold,arial", 24, bold=True)
FONT_SUB = pygame.font.SysFont("segoeui,arial", 15)
FONT_BIG = pygame.font.SysFont("segoeuisemibold,arial", 44, bold=True)
FONT_SMALL = pygame.font.SysFont("segoeui,arial", 14)
FONT_BTN = pygame.font.SysFont("segoeuisemibold,arial", 20, bold=True)
FONT_BTN_SUB = pygame.font.SysFont("segoeui,arial", 13)
FONT_NAME = pygame.font.SysFont("segoeuisemibold,arial", 26, bold=True)


def make_vertical_gradient(size, top_color, bottom_color):
    surf = pygame.Surface(size)
    h = size[1]
    for y in range(h):
        t = y / h
        color = tuple(int(top_color[i] + (bottom_color[i] - top_color[i]) * t) for i in range(3))
        pygame.draw.line(surf, color, (0, y), (size[0], y))
    return surf


def make_vertical_gradient_with_stars(size, top_color, bottom_color, seed=42):
    surf = make_vertical_gradient(size, top_color, bottom_color)
    rng = random.Random(seed)
    # 110 static celestial stars of varying brightness and twinkle
    for _ in range(110):
        sx = rng.randint(0, size[0] - 1)
        sy = rng.randint(0, size[1] - 1)
        b = rng.randint(140, 255)
        roll = rng.random()
        tint = (b, b, min(255, b + 25)) if roll > 0.4 else (b, min(255, b + 15), b)
        if roll > 0.88:
            pygame.draw.circle(surf, tint, (sx, sy), 2)
            surf.set_at((max(0, sx - 2), sy), (b // 2, b // 2, b // 2))
            surf.set_at((min(size[0] - 1, sx + 2), sy), (b // 2, b // 2, b // 2))
            surf.set_at((sx, max(0, sy - 2)), (b // 2, b // 2, b // 2))
            surf.set_at((sx, min(size[1] - 1, sy + 2)), (b // 2, b // 2, b // 2))
        elif roll > 0.5:
            pygame.draw.circle(surf, tint, (sx, sy), 1)
        else:
            surf.set_at((sx, sy), tint)
    return surf


BG_SURFACE = make_vertical_gradient_with_stars((WIDTH, HEIGHT), BACKGROUND_TOP, BACKGROUND_BOTTOM)


# ============================================================
#  ONE-EURO ADAPTIVE FILTER & HITBOX GEOMETRY
# ============================================================
class OneEuroFilter:
    """Adaptive 1-Euro filter: heavy smoothing when hand is stationary (eliminates jitter),
    near-zero lag when hand moves fast (preserves responsiveness)."""
    def __init__(self, min_cutoff=1.0, beta=0.018, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, rate, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x, t):
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            self.t_prev = float(t)
            return float(x)

        dt = max(1e-4, float(t - self.t_prev))
        rate = 1.0 / dt
        self.t_prev = float(t)

        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(rate, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(rate, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


def remap_cursor_norm(v, margin=CURSOR_EDGE_MARGIN):
    """Rescales a normalized (0-1) camera coordinate so that the reliable center
    portion of the frame - [margin, 1-margin] - is stretched to cover the full
    [0, 1] range before it gets multiplied out to screen pixels. This lets the
    on-screen cursor actually reach the screen's edges (and therefore the tubes
    placed nearest to them) without requiring the hand to be pushed into the
    unreliable outer edge of the webcam's field of view."""
    span = max(1e-6, 1.0 - 2.0 * margin)
    return min(1.0, max(0.0, (v - margin) / span))


# Sticky hysteresis margin: once a tube is the active drop target, its
# hit zone expands by this many pixels on each side so minor tremor
# doesn't flick the selection to a neighbouring tube.
TUBE_MAGNET_PX = 22


def find_hovered_tube(hx, hy, tube_rects, sticky_idx=None):
    """Finds hovered tube using continuous mid-gap Voronoi boundaries
    and generous vertical tolerance. Eliminates all dead zones between narrow tubes.
    When sticky_idx is provided (the tube the player is currently targeting),
    that tube's boundaries are expanded by TUBE_MAGNET_PX so it 'holds on'
    even through minor hand tremor."""
    if not tube_rects:
        return None
    top_limit = 110
    bottom_limit = tube_rects[0].bottom + 50
    if not (top_limit <= hy <= bottom_limit):
        return None

    # Check the sticky tube first with expanded boundaries
    if sticky_idx is not None and 0 <= sticky_idx < len(tube_rects):
        rect = tube_rects[sticky_idx]
        left_limit = (tube_rects[sticky_idx - 1].right + rect.left) / 2 if sticky_idx > 0 else rect.left - 40
        right_limit = (rect.right + tube_rects[sticky_idx + 1].left) / 2 if sticky_idx < len(tube_rects) - 1 else rect.right + 40
        if (left_limit - TUBE_MAGNET_PX) <= hx <= (right_limit + TUBE_MAGNET_PX):
            return sticky_idx

    for idx, rect in enumerate(tube_rects):
        left_limit = (tube_rects[idx - 1].right + rect.left) / 2 if idx > 0 else rect.left - 40
        right_limit = (rect.right + tube_rects[idx + 1].left) / 2 if idx < len(tube_rects) - 1 else rect.right + 40
        if left_limit <= hx <= right_limit:
            return idx
    return None


# ============================================================
#  DYNAMIC LAYOUT  (fits anywhere from 5 to 13 tubes on screen)
# ============================================================
def compute_layout(num_tubes):
    """Returns (tube_width, gap, ball_radius, tube_height, tube_rects) sized to fit WIDTH."""
    margin = 40
    available_width = WIDTH - margin * 2
    gap_ratio = 0.35
    max_tube_width = 84
    min_tube_width = 30

    denom = num_tubes + gap_ratio * (num_tubes - 1)
    tube_width = available_width / denom
    tube_width = max(min_tube_width, min(max_tube_width, tube_width))
    gap = tube_width * gap_ratio

    ball_radius = max(9, int(tube_width * 0.31))
    tube_height = TUBE_CAPACITY * ball_radius * 2 + 40

    total_width = num_tubes * tube_width + (num_tubes - 1) * gap
    start_x = (WIDTH - total_width) / 2
    tube_y = 300

    rects = []
    for i in range(num_tubes):
        x = start_x + i * (tube_width + gap)
        rects.append(pygame.Rect(int(round(x)), tube_y, int(round(tube_width)), int(round(tube_height))))
    return tube_width, gap, ball_radius, tube_height, rects


def generate_puzzle(level_cfg):
    tubes_count = level_cfg["tubes"]
    colors_count = level_cfg["colors"]
    colors = COLOR_LIST[:colors_count]
    balls = []
    for color in colors:
        balls.extend([color] * TUBE_CAPACITY)
    random.shuffle(balls)
    data = []
    for i in range(tubes_count):
        if i < colors_count:
            data.append(balls[i * TUBE_CAPACITY:(i + 1) * TUBE_CAPACITY])
        else:
            data.append([])
    return data


def check_victory(tubes):
    for tube in tubes:
        if len(tube) == 0:
            continue
        if len(tube) != TUBE_CAPACITY or len(set(tube)) != 1:
            return False
    return True


def lighten(color, amt):
    return tuple(min(255, int(c + (255 - c) * amt)) for c in color)


def darken(color, amt):
    return tuple(max(0, int(c * (1 - amt))) for c in color)


def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1 - pow(1 - t, 3)


def draw_ball(surf, color, x, y, radius, alpha=255):
    x, y = int(x), int(y)
    if alpha >= 255:
        pygame.draw.ellipse(surf, (0, 0, 0, 70),
                             pygame.Rect(x - radius - 2, y + radius - 8, radius * 2 + 4, 10))
        pygame.draw.circle(surf, darken(color, 0.28), (x, y), radius)
        pygame.draw.circle(surf, color, (x, y), radius - 2)
        pygame.draw.circle(surf, lighten(color, 0.55), (x - radius // 3, y - radius // 3), max(3, radius // 3))
        pygame.draw.circle(surf, (255, 255, 255), (x - radius // 2, y - radius // 2), max(2, radius // 6))
    else:
        temp = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        cx, cy = radius + 2, radius + 2
        pygame.draw.circle(temp, darken(color, 0.28), (cx, cy), radius)
        pygame.draw.circle(temp, color, (cx, cy), radius - 2)
        pygame.draw.circle(temp, lighten(color, 0.55), (cx - radius // 3, cy - radius // 3), max(3, radius // 3))
        temp.set_alpha(alpha)
        surf.blit(temp, (x - cx, y - cy))


_tube_glass_cache = {}
_tube_glow_cache = {}


def _get_tube_glass(w, h):
    key = (w, h)
    s = _tube_glass_cache.get(key)
    if s is None:
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, TUBE_GLASS, s.get_rect(),
                          border_bottom_left_radius=20, border_bottom_right_radius=20)
        _tube_glass_cache[key] = s
    return s


def _get_tube_glow(w, h):
    key = (w, h)
    s = _tube_glow_cache.get(key)
    if s is None:
        s = pygame.Surface((w + 30, h + 20), pygame.SRCALPHA)
        pygame.draw.rect(s, (*TUBE_BORDER_ACTIVE, 40), s.get_rect(), border_radius=24)
        _tube_glow_cache[key] = s
    return s


def draw_tube(surf, rect, active):
    # Glass + glow are the same pixels every frame for a given tube size, so they're
    # built once per size and reused instead of allocating+filling a new
    # per-pixel-alpha Surface on every single frame for every tube (was the
    # single biggest render-loop cost at 13 tubes x 60fps).
    surf.blit(_get_tube_glass(rect.width, rect.height), rect.topleft)

    border_color = TUBE_BORDER_ACTIVE if active else TUBE_BORDER
    width = 6 if active else 4
    if active:
        # Gentle breathing pulse via set_alpha directly on the cached surface.
        # Mutating the cached surface's alpha is safe because every tube that
        # draws the glow in this frame uses the same alpha anyway, and the
        # surface is reset each frame. Eliminates the per-frame .copy() alloc.
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 220.0))
        glow = _get_tube_glow(rect.width, rect.height)
        glow.set_alpha(int(160 + 95 * pulse))
        surf.blit(glow, (rect.x - 15, rect.y - 10))

    pygame.draw.lines(surf, border_color, False, [
        (rect.left, rect.top),
        (rect.left, rect.bottom - 15),
        (rect.left + 15, rect.bottom),
        (rect.right - 15, rect.bottom),
        (rect.right, rect.bottom - 15),
        (rect.right, rect.top)
    ], width=width)


def slot_position(rect, slot_idx, radius):
    x = rect.centerx
    y = rect.bottom - (slot_idx * (radius * 2)) - radius - 5
    return x, y


_THUMB_SURF = pygame.Surface((THUMB_WIDTH, THUMB_HEIGHT))


def draw_thumbnail(surf, rgb_small, landmark_pt, pos=None, cursor_near=False):
    if pos is None:
        box_x, box_y = WIDTH - THUMB_WIDTH - 16, 12
    else:
        box_x, box_y = pos
    alpha = 75 if cursor_near else 235

    container = pygame.Surface((THUMB_WIDTH, THUMB_HEIGHT), pygame.SRCALPHA)
    if rgb_small is not None:
        pygame.surfarray.blit_array(_THUMB_SURF, np.transpose(rgb_small, (1, 0, 2)))
        container.blit(_THUMB_SURF, (0, 0))
        if landmark_pt is not None:
            lx = landmark_pt[0] * THUMB_WIDTH
            ly = landmark_pt[1] * THUMB_HEIGHT
            pygame.draw.circle(container, ACCENT, (int(lx), int(ly)), 4)
    else:
        container.fill((25, 20, 42))

    container.set_alpha(alpha)
    surf.blit(container, (box_x, box_y))

    border_surf = pygame.Surface((THUMB_WIDTH, THUMB_HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(border_surf, (*TUBE_BORDER, alpha), border_surf.get_rect(), width=2, border_radius=8)
    pygame.draw.rect(border_surf, (15, 12, 28, min(alpha, 200)), (4, 4, 32, 16), border_radius=4)
    badge = FONT_SMALL.render("CAM", True, (215, 215, 235))
    border_surf.blit(badge, (7, 4))
    surf.blit(border_surf, (box_x, box_y))


# ============================================================
#  LEVEL-SELECT MENU LAYOUT
# ============================================================
def get_menu_buttons():
    buttons = []
    btn_w, btn_h = 175, 100
    gap_x, gap_y = 26, 20
    row1_w = 4 * btn_w + 3 * gap_x
    start_x_4 = (WIDTH - row1_w) // 2

    row3_w = 2 * btn_w + 1 * gap_x
    start_x_2 = (WIDTH - row3_w) // 2

    base_y = 135
    for i in range(10):
        if i < 4:  # Row 1: Lv 1, 2, 3, 4
            x = start_x_4 + i * (btn_w + gap_x)
            y = base_y
        elif i < 8:  # Row 2: Lv 5, 6, 7, 8
            x = start_x_4 + (i - 4) * (btn_w + gap_x)
            y = base_y + btn_h + gap_y  # 255
        else:  # Row 3: Lv 9, 10 centered
            x = start_x_2 + (i - 8) * (btn_w + gap_x)
            y = base_y + 2 * (btn_h + gap_y)  # 375
        buttons.append((i, pygame.Rect(x, y, btn_w, btn_h)))
    return buttons


MENU_BUTTONS = get_menu_buttons()

# Bottom action bar buttons (centered horizontally)
BTN_LEADERBOARD = pygame.Rect(170, 520, 200, 48)
BTN_THEMES = pygame.Rect(395, 520, 175, 48)
INDICATOR_HAND = pygame.Rect(595, 520, 235, 48)

THEME_CARD_RECTS = {
    "galaxy": pygame.Rect(195, 185, 290, 115),
    "nebula": pygame.Rect(515, 185, 290, 115),
    "cyber":  pygame.Rect(195, 315, 290, 115),
    "sunset": pygame.Rect(515, 315, 290, 115),
}


def _build_menu_button_art():
    art = {}
    for idx, rect in MENU_BUTTONS:
        cfg = LEVELS[idx]
        base = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(base, TUBE_GLASS, base.get_rect(), border_radius=16)

        key_char = str(idx + 1 if idx < 9 else 0)
        badge_txt = FONT_SMALL.render(f"[{key_char}]", True, ACCENT)
        base.blit(badge_txt, (rect.width - badge_txt.get_width() - 10, 8))

        lvl_txt = FONT_BTN.render(f"LEVEL {idx + 1}", True, TEXT_COLOR)
        base.blit(lvl_txt, (rect.width // 2 - lvl_txt.get_width() // 2, 18))
        tubes_txt = FONT_BTN_SUB.render(f"{cfg['tubes']} tubes", True, (190, 200, 225))
        base.blit(tubes_txt, (rect.width // 2 - tubes_txt.get_width() // 2, 50))
        colors_txt = FONT_BTN_SUB.render(f"{cfg['colors']} colors", True, (150, 165, 195))
        base.blit(colors_txt, (rect.width // 2 - colors_txt.get_width() // 2, 70))
        art[idx] = base
    return art


def _build_menu_glow(w, h):
    s = pygame.Surface((w + 20, h + 20), pygame.SRCALPHA)
    pygame.draw.rect(s, (*TUBE_BORDER_ACTIVE, 55), s.get_rect(), border_radius=20)
    return s


MENU_BUTTON_ART = _build_menu_button_art()
MENU_GLOW = _build_menu_glow(175, 100)
MENU_FOOTER_SURF = FONT_SMALL.render("ESC  quit    1-9, 0  select level    L  leaderboard    T  themes    C  toggle camera", True, (140, 145, 175))


def apply_theme(theme_key):
    global current_theme_key, BACKGROUND_TOP, BACKGROUND_BOTTOM, ACCENT, ACCENT_SECONDARY
    global TUBE_GLASS, TUBE_BORDER, TUBE_BORDER_ACTIVE, SHADOW_COLOR, TEXT_COLOR, WIN_COLOR
    global BALL_COLORS, COLOR_LIST, BG_SURFACE, MENU_BUTTON_ART, MENU_GLOW
    if theme_key not in THEMES:
        return
    current_theme_key = theme_key
    t = THEMES[theme_key]
    BACKGROUND_TOP = t["bg_top"]
    BACKGROUND_BOTTOM = t["bg_bottom"]
    ACCENT = t["accent"]
    ACCENT_SECONDARY = t["accent_secondary"]
    TUBE_GLASS = t["tube_glass"]
    TUBE_BORDER = t["tube_border"]
    TUBE_BORDER_ACTIVE = t["tube_border_active"]
    SHADOW_COLOR = t["shadow_color"]
    TEXT_COLOR = t["text_color"]
    WIN_COLOR = t["win_color"]
    BALL_COLORS = t["ball_colors"]
    COLOR_LIST = list(BALL_COLORS.values())

    if t.get("has_stars", False):
        BG_SURFACE = make_vertical_gradient_with_stars((WIDTH, HEIGHT), BACKGROUND_TOP, BACKGROUND_BOTTOM)
    else:
        BG_SURFACE = make_vertical_gradient((WIDTH, HEIGHT), BACKGROUND_TOP, BACKGROUND_BOTTOM)

    MENU_BUTTON_ART = _build_menu_button_art()
    MENU_GLOW = _build_menu_glow(175, 100)


def draw_action_button(surf, rect, label, icon, is_hover=False, active=False):
    btn_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg = (*ACCENT, 70) if active else ((50, 70, 120, 140) if is_hover else (30, 25, 55, 160))
    border = ACCENT if (is_hover or active) else TUBE_BORDER
    pygame.draw.rect(btn_surf, bg, btn_surf.get_rect(), border_radius=14)
    pygame.draw.rect(btn_surf, border, btn_surf.get_rect(), width=3 if is_hover else 2, border_radius=14)
    full_lbl = f"{icon}  {label}"
    txt = FONT_BTN.render(full_lbl, True, TEXT_COLOR)
    btn_surf.blit(txt, (rect.width // 2 - txt.get_width() // 2, rect.height // 2 - txt.get_height() // 2))
    surf.blit(btn_surf, rect.topleft)


def draw_hand_indicator(surf, rect, hand_detected):
    pill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg_color = (25, 50, 32, 175) if hand_detected else (45, 25, 30, 175)
    border_color = (60, 230, 120) if hand_detected else (220, 95, 95)
    dot_color = (65, 245, 125) if hand_detected else (240, 90, 90)
    text_color = (215, 255, 230) if hand_detected else (255, 210, 210)
    txt_str = "Hand: Detected" if hand_detected else "Hand: Not Detected"

    pygame.draw.rect(pill, bg_color, pill.get_rect(), border_radius=14)
    pygame.draw.rect(pill, border_color, pill.get_rect(), width=2, border_radius=14)
    pygame.draw.circle(pill, dot_color, (22, rect.height // 2), 6)
    txt = FONT_SUB.render(txt_str, True, text_color)
    pill.blit(txt, (38, rect.height // 2 - txt.get_height() // 2))
    surf.blit(pill, rect.topleft)


def draw_leaderboard_modal(surf, close_hover=False):
    dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    dim.fill((6, 5, 14, 205))
    surf.blit(dim, (0, 0))

    modal_rect = pygame.Rect(240, 115, 520, 445)
    card = pygame.Surface((modal_rect.width, modal_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(card, (18, 15, 32, 245), card.get_rect(), border_radius=20)
    pygame.draw.rect(card, ACCENT, card.get_rect(), width=2, border_radius=20)

    title = FONT_TITLE.render("🏆  GLOBAL LEADERBOARD", True, ACCENT)
    card.blit(title, (modal_rect.width // 2 - title.get_width() // 2, 22))
    sub = FONT_SMALL.render("Top scores from players worldwide", True, (170, 170, 200))
    card.blit(sub, (modal_rect.width // 2 - sub.get_width() // 2, 54))

    with net_lock:
        loaded = leaderboard_data["loaded"]
        entries = list(leaderboard_data["entries"])

    medals = ["🥇", "🥈", "🥉", "4.", "5."]
    if loaded and entries:
        for i, entry in enumerate(entries[:5]):
            row_y = 90 + i * 52
            row_rect = pygame.Rect(25, row_y, modal_rect.width - 50, 42)
            pygame.draw.rect(card, (30, 25, 52, 180), row_rect, border_radius=10)
            medal_str = medals[i] if i < len(medals) else f"{i+1}."
            medal_txt = FONT_BTN.render(medal_str, True, ACCENT if i < 3 else TEXT_COLOR)
            card.blit(medal_txt, (40, row_y + (42 - medal_txt.get_height()) // 2))

            name_str = str(entry.get("name", "Anonymous"))[:18]
            name_txt = FONT_BTN.render(name_str, True, TEXT_COLOR)
            card.blit(name_txt, (95, row_y + (42 - name_txt.get_height()) // 2))

            moves_str = f"{entry.get('moves', '?')} moves"
            moves_txt = FONT_BTN.render(moves_str, True, (180, 220, 255))
            card.blit(moves_txt, (modal_rect.width - 40 - moves_txt.get_width(), row_y + (42 - moves_txt.get_height()) // 2))
    elif loaded:
        no_scores = FONT_SUB.render("No scores yet — be the first to solve a level!", True, (180, 180, 210))
        card.blit(no_scores, (modal_rect.width // 2 - no_scores.get_width() // 2, 180))
    else:
        loading = FONT_SUB.render("Connecting to leaderboard server...", True, (180, 180, 210))
        card.blit(loading, (modal_rect.width // 2 - loading.get_width() // 2, 180))

    surf.blit(card, modal_rect.topleft)

    close_rect = pygame.Rect(WIDTH // 2 - 60, modal_rect.bottom - 55, 120, 38)
    close_surf = pygame.Surface((close_rect.width, close_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(close_surf, (*ACCENT, 80) if close_hover else (40, 35, 70, 200), close_surf.get_rect(), border_radius=12)
    pygame.draw.rect(close_surf, ACCENT if close_hover else TUBE_BORDER, close_surf.get_rect(), width=2, border_radius=12)
    cl_txt = FONT_BTN.render("Close", True, TEXT_COLOR)
    close_surf.blit(cl_txt, (close_rect.width // 2 - cl_txt.get_width() // 2, close_rect.height // 2 - cl_txt.get_height() // 2))
    surf.blit(close_surf, close_rect.topleft)


def draw_themes_modal(surf, hovered_theme=None, close_hover=False):
    dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    dim.fill((6, 5, 14, 205))
    surf.blit(dim, (0, 0))

    modal_rect = pygame.Rect(170, 115, 660, 465)
    card = pygame.Surface((modal_rect.width, modal_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(card, (18, 15, 32, 245), card.get_rect(), border_radius=20)
    pygame.draw.rect(card, ACCENT, card.get_rect(), width=2, border_radius=20)

    title = FONT_TITLE.render("🎨  SELECT VISUAL THEME", True, ACCENT)
    card.blit(title, (modal_rect.width // 2 - title.get_width() // 2, 22))
    sub = FONT_SMALL.render("Pinch or click any theme to apply immediately", True, (170, 170, 200))
    card.blit(sub, (modal_rect.width // 2 - sub.get_width() // 2, 54))

    surf.blit(card, modal_rect.topleft)

    for t_key, t_rect in THEME_CARD_RECTS.items():
        t = THEMES[t_key]
        is_cur = (t_key == current_theme_key)
        is_hov = (t_key == hovered_theme)

        c_surf = pygame.Surface((t_rect.width, t_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(c_surf, (*t["bg_top"], 230), c_surf.get_rect(), border_radius=16)
        border_col = t["accent"] if (is_cur or is_hov) else t["tube_border"]
        border_w = 3 if (is_cur or is_hov) else 2
        pygame.draw.rect(c_surf, border_col, c_surf.get_rect(), width=border_w, border_radius=16)

        name_txt = FONT_BTN.render(t["name"], True, t["text_color"])
        c_surf.blit(name_txt, (18, 16))

        if is_cur:
            badge = FONT_SMALL.render("✓ ACTIVE", True, t["accent"])
            c_surf.blit(badge, (t_rect.width - badge.get_width() - 16, 18))
        elif is_hov:
            badge = FONT_SMALL.render("PINCH TO SELECT", True, ACCENT)
            c_surf.blit(badge, (t_rect.width - badge.get_width() - 16, 18))

        sub_desc = "Deep space with static stars" if t.get("has_stars") else f"Vibrant {t['name']} palette"
        desc_txt = FONT_BTN_SUB.render(sub_desc, True, (160, 170, 200))
        c_surf.blit(desc_txt, (18, 44))

        ball_colors_sample = list(t["ball_colors"].values())[:6]
        for c_idx, c_rgb in enumerate(ball_colors_sample):
            dot_x = 24 + c_idx * 28
            dot_y = 86
            pygame.draw.circle(c_surf, c_rgb, (dot_x, dot_y), 10)
            pygame.draw.circle(c_surf, (255, 255, 255, 100), (dot_x - 3, dot_y - 3), 3)

        surf.blit(c_surf, t_rect.topleft)

    close_rect = pygame.Rect(WIDTH // 2 - 60, modal_rect.bottom - 55, 120, 38)
    close_surf = pygame.Surface((close_rect.width, close_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(close_surf, (*ACCENT, 80) if close_hover else (40, 35, 70, 200), close_surf.get_rect(), border_radius=12)
    pygame.draw.rect(close_surf, ACCENT if close_hover else TUBE_BORDER, close_surf.get_rect(), width=2, border_radius=12)
    cl_txt = FONT_BTN.render("Close", True, TEXT_COLOR)
    close_surf.blit(cl_txt, (close_rect.width // 2 - cl_txt.get_width() // 2, close_rect.height // 2 - cl_txt.get_height() // 2))
    surf.blit(close_surf, close_rect.topleft)


def draw_menu(surf, hovered_idx, hovered_action, cursor_active, player_name, menu_overlay=None, hovered_theme_key=None, hovered_modal_close=False):
    surf.blit(BG_SURFACE, (0, 0))

    # Top Center Title
    title = FONT_TITLE.render("BALL SORT PUZZLE", True, TEXT_COLOR)
    surf.blit(title, (WIDTH // 2 - title.get_width() // 2, 28))
    sub = FONT_SUB.render("Select a level  ·  Pinch or press 1-9, 0", True, (170, 175, 205))
    surf.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 64))

    # Top Right Player Badge
    player_lbl = FONT_SUB.render(f"Player: {player_name}", True, TEXT_COLOR)
    pill_w = player_lbl.get_width() + 36
    pill_h = 36
    pill_x = WIDTH - pill_w - 24
    pill_y = 18
    pill_surf = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
    pygame.draw.rect(pill_surf, (*BACKGROUND_TOP, 190), pill_surf.get_rect(), border_radius=18)
    pygame.draw.rect(pill_surf, (*ACCENT, 200), pill_surf.get_rect(), width=2, border_radius=18)
    pygame.draw.circle(pill_surf, ACCENT, (16, pill_h // 2), 6)
    surf.blit(pill_surf, (pill_x, pill_y))
    surf.blit(player_lbl, (pill_x + 28, pill_y + (pill_h - player_lbl.get_height()) // 2))

    # Level Grid (Centerpiece)
    for idx, rect in MENU_BUTTONS:
        is_hover = (idx == hovered_idx and menu_overlay is None)
        if is_hover:
            surf.blit(MENU_GLOW, (rect.x - 10, rect.y - 10))
        surf.blit(MENU_BUTTON_ART[idx], rect.topleft)
        border_color = TUBE_BORDER_ACTIVE if is_hover else TUBE_BORDER
        pygame.draw.rect(surf, border_color, rect, width=3 if is_hover else 2, border_radius=16)

    # Bottom Action Bar
    draw_action_button(surf, BTN_LEADERBOARD, "Leaderboard", "🏆", is_hover=(hovered_action == "leaderboard" and menu_overlay is None), active=(menu_overlay == "leaderboard"))
    draw_action_button(surf, BTN_THEMES, "Themes", "🎨", is_hover=(hovered_action == "themes" and menu_overlay is None), active=(menu_overlay == "themes"))
    draw_hand_indicator(surf, INDICATOR_HAND, cursor_active)

    # Footer navigation hints
    surf.blit(MENU_FOOTER_SURF, (WIDTH // 2 - MENU_FOOTER_SURF.get_width() // 2, 658))

    # Modals (if open)
    if menu_overlay == "leaderboard":
        draw_leaderboard_modal(surf, close_hover=hovered_modal_close)
    elif menu_overlay == "themes":
        draw_themes_modal(surf, hovered_theme=hovered_theme_key, close_hover=hovered_modal_close)


# ============================================================
#  BACKEND NETWORKING  (runs in background threads so the game never freezes)
# ============================================================
net_lock = threading.Lock()
score_submit_status = {"sent": False, "ok": False}
leaderboard_data = {"loaded": False, "entries": []}

_net_session = requests.Session()
_net_session.trust_env = False  # Ignore system proxies and VPN configurations for localhost


def ensure_backend_running():
    """Checks if the Flask leaderboard backend is listening on port 5050; if not, launches it."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    is_open = False
    try:
        s.connect(("127.0.0.1", 5050))
        is_open = True
    except (socket.error, ConnectionRefusedError, OSError):
        is_open = False
    finally:
        s.close()

    if not is_open:
        backend_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "backend.py"))
        if os.path.exists(backend_script):
            import subprocess
            try:
                subprocess.Popen(
                    [sys.executable, backend_script],
                    cwd=os.path.dirname(backend_script),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(0.5)
            except Exception as e:
                print(f"[network] Auto-launch backend error: {e}")


def submit_score_async(name, moves):
    """Sends finished score to the backend with retry and localhost fallbacks."""
    with net_lock:
        score_submit_status["sent"] = False
        score_submit_status["ok"] = False

    def worker():
        ensure_backend_running()
        ok = False
        endpoints = [BACKEND_URL]
        if "127.0.0.1" in BACKEND_URL:
            endpoints.append("http://localhost:5050")
        elif "localhost" in BACKEND_URL:
            endpoints.append("http://127.0.0.1:5050")

        for base_url in endpoints:
            for _ in range(2):
                try:
                    resp = _net_session.post(
                        f"{base_url}/add_score",
                        json={"name": name, "moves": moves},
                        timeout=3,
                    )
                    if resp.status_code == 200:
                        ok = True
                        break
                except Exception as e:
                    time.sleep(0.2)
            if ok:
                break

        with net_lock:
            score_submit_status["sent"] = True
            score_submit_status["ok"] = ok

        fetch_leaderboard_async()

    threading.Thread(target=worker, daemon=True).start()


def fetch_leaderboard_async():
    """Fetches top scores from the backend in the background."""
    def worker():
        ensure_backend_running()
        entries = []
        endpoints = [BACKEND_URL]
        if "127.0.0.1" in BACKEND_URL:
            endpoints.append("http://localhost:5050")
        elif "localhost" in BACKEND_URL:
            endpoints.append("http://127.0.0.1:5050")

        for base_url in endpoints:
            for _ in range(2):
                try:
                    resp = _net_session.get(f"{base_url}/get_leaderboard", timeout=3)
                    if resp.status_code == 200:
                        entries = resp.json().get("leaderboard", [])
                        break
                except Exception:
                    time.sleep(0.2)
            if entries:
                break

        with net_lock:
            leaderboard_data["loaded"] = True
            if entries:
                leaderboard_data["entries"] = entries

    threading.Thread(target=worker, daemon=True).start()


# ============================================================
#  NAME ENTRY SCREEN  (simple keyboard text box shown once at startup)
# ============================================================
def run_name_entry():
    pygame.key.start_text_input()
    name = ""
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.release()
                detector.close()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.TEXTINPUT:
                if len(name) < 16:
                    name += event.text
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pygame.key.stop_text_input()
                    pygame.event.clear()
                    return name.strip() if name.strip() else "Player"
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                elif event.key == pygame.K_ESCAPE:
                    camera.release()
                    detector.close()
                    pygame.quit()
                    sys.exit()

        screen.blit(BG_SURFACE, (0, 0))
        title = FONT_TITLE.render("BALL SORT — ENTER YOUR NAME", True, TEXT_COLOR)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 260))

        box_rect = pygame.Rect(WIDTH // 2 - 180, 310, 360, 50)
        pygame.draw.rect(screen, (30, 26, 50), box_rect, border_radius=10)
        pygame.draw.rect(screen, TUBE_BORDER, box_rect, width=2, border_radius=10)

        display_name = name if name else "Type your name..."
        color = TEXT_COLOR if name else (130, 130, 155)
        name_surf = FONT_NAME.render(display_name, True, color)
        screen.blit(name_surf, (box_rect.x + 14, box_rect.y + 10))

        hint = FONT_SUB.render("Press ENTER to continue to the level menu", True, (170, 170, 195))
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, 380))

        pygame.display.flip()


# ============================================================
#  GAME STATE
# ============================================================
player_name = run_name_entry()
pygame.event.clear()

game_state = "menu"          # "menu" or "playing"
menu_overlay = None          # None, "leaderboard", or "themes"
current_level_idx = 0
tubes_data = []
tube_rects = []
cur_ball_radius = 26
cur_tube_height = 260

selected_ball_color = None
source_tube_idx = None

hand_x, hand_y = WIDTH // 2, HEIGHT // 2
smooth_x, smooth_y = float(hand_x), float(hand_y)
is_pinching = False
cursor_found = False
prev_cursor_found = False
lost_hand_grace = 0
LOST_HAND_MAX_GRACE = 3
current_pinch_ratio = 1.0
last_pinch_hx, last_pinch_hy = WIDTH // 2, HEIGHT // 2  # release-anchor coords
last_pinch_tube = None  # tube idx the cursor was over while pinching

FILTER_BETA_IDLE = 0.018   # heavy smoothing when browsing
FILTER_BETA_HOLD = 0.06    # low-lag mode while carrying a ball
cursor_filter_x = OneEuroFilter(min_cutoff=1.0, beta=FILTER_BETA_IDLE)
cursor_filter_y = OneEuroFilter(min_cutoff=1.0, beta=FILTER_BETA_IDLE)

drop_animations = {}   # tube_idx -> animation dict
moves = 0
game_won = False
all_levels_complete = False
show_camera_preview = True

level_intro_start = 0        # ticks when the current level finished loading
level_intro_active = False   # becomes False once every ball has finished popping in
confetti = []                # active celebration particles, spawned once per win


def spawn_confetti():
    confetti.clear()
    palette = COLOR_LIST if COLOR_LIST else [ACCENT, ACCENT_SECONDARY]
    for _ in range(CONFETTI_COUNT):
        confetti.append({
            "x": random.uniform(0, WIDTH),
            "y": random.uniform(-HEIGHT * 0.4, 0),
            "vx": random.uniform(-40, 40),
            "vy": random.uniform(120, 260),
            "size": random.uniform(3, 7),
            "color": random.choice(palette),
            "spin": random.uniform(-6, 6),
            "angle": random.uniform(0, 360),
        })


def update_and_draw_confetti(surf, dt):
    for p in confetti:
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt
        p["vy"] += 260 * dt  # gravity
        p["angle"] += p["spin"]
        if p["y"] > HEIGHT + 20:
            p["y"] = random.uniform(-40, -10)
            p["x"] = random.uniform(0, WIDTH)
            p["vy"] = random.uniform(120, 260)
        s = p["size"]
        rect_surf = pygame.Surface((int(s * 2), int(s)), pygame.SRCALPHA)
        rect_surf.fill(p["color"])
        rotated = pygame.transform.rotate(rect_surf, p["angle"])
        surf.blit(rotated, (p["x"], p["y"]))


def load_level(idx):
    global current_level_idx, tubes_data, tube_rects, cur_ball_radius, cur_tube_height
    global selected_ball_color, source_tube_idx, drop_animations, moves, game_won, all_levels_complete
    global level_intro_start, level_intro_active
    refresh_detector()
    current_level_idx = idx
    cfg = LEVELS[idx]
    _, _, cur_ball_radius, cur_tube_height, tube_rects = compute_layout(cfg["tubes"])
    tubes_data = generate_puzzle(cfg)
    selected_ball_color = None
    source_tube_idx = None
    drop_animations = {}
    moves = 0
    game_won = False
    all_levels_complete = False
    confetti.clear()
    level_intro_start = pygame.time.get_ticks()
    level_intro_active = True
    with net_lock:
        score_submit_status["sent"] = False
        score_submit_status["ok"] = False


start_ts = time.time()
last_ts_ms = 0

fetch_leaderboard_async()
print("Game ready. Look into the webcam and pinch thumb + index finger to select a level and to grab/drop balls.")

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if menu_overlay is not None:
                    menu_overlay = None
                else:
                    running = False
            elif event.key == pygame.K_c:
                show_camera_preview = not show_camera_preview
            elif game_state == "menu":
                if event.key == pygame.K_l:
                    menu_overlay = "leaderboard" if menu_overlay != "leaderboard" else None
                    if menu_overlay == "leaderboard":
                        fetch_leaderboard_async()
                elif event.key == pygame.K_t:
                    menu_overlay = "themes" if menu_overlay != "themes" else None
                elif menu_overlay is None:
                    if pygame.K_1 <= event.key <= pygame.K_9:
                        idx = event.key - pygame.K_1
                        if idx < len(LEVELS):
                            load_level(idx)
                            game_state = "playing"
                    elif event.key == pygame.K_0:
                        idx = 9
                        if idx < len(LEVELS):
                            load_level(idx)
                            game_state = "playing"
            elif game_state == "playing":
                if event.key == pygame.K_r:
                    load_level(current_level_idx)
                elif event.key == pygame.K_m:
                    game_state = "menu"
                    is_pinching = False
                    prev_cursor_found = False
                    pygame.event.clear()
                    refresh_detector()
                elif event.key == pygame.K_n and game_won:
                    if current_level_idx + 1 < len(LEVELS):
                        load_level(current_level_idx + 1)
                        game_state = "playing"
                    else:
                        all_levels_complete = True

    # ---------------- Camera + async detection ----------------
    frame = camera.read()
    rgb_thumb = None

    if frame is not None:
        frame = cv2.flip(frame, 1)

        thumb_bgr = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT))
        rgb_thumb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)

        # Only hand a new frame to the model once it has actually finished
        # with the last one. Without this, a slower device (e.g. a Jetson
        # Nano on CPU) queues up frames faster than it can process them,
        # so the hand cursor visibly lags further and further behind —
        # skipping redundant frames keeps latency bounded instead.
        with _result_lock:
            busy = _detection_busy
        if not busy:
            if time.time() - _last_detector_refresh > 75 and selected_ball_color is None:
                refresh_detector()
            det_bgr = cv2.resize(frame, (DET_WIDTH, DET_HEIGHT))
            cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGBA, dst=_det_buffer)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=_det_buffer)

            ts_ms = max(last_ts_ms + 1, int((time.time() - start_ts) * 1000))
            last_ts_ms = ts_ms
            try:
                with _result_lock:
                    _detection_busy = True
                detector.detect_async(mp_image, ts_ms)
            except Exception:
                with _result_lock:
                    _detection_busy = False

    with _result_lock:
        hand_landmarks_list = _latest_landmarks

    cursor_found = False
    pinch_detected = False
    thumb_landmark_norm = None
    now_t = time.time()

    if hand_landmarks_list:
        lost_hand_grace = 0
        hl = hand_landmarks_list[0]
        thumb_tip = hl[4]
        index_tip = hl[8]
        wrist = hl[0]
        middle_mcp = hl[9]

        # Remap through the edge-compensation curve BEFORE scaling to screen pixels,
        # so the cursor can reliably reach tubes placed near the screen edges.
        ix = remap_cursor_norm(index_tip.x) * WIDTH
        iy = remap_cursor_norm(index_tip.y) * HEIGHT
        tx = remap_cursor_norm(thumb_tip.x) * WIDTH
        ty = remap_cursor_norm(thumb_tip.y) * HEIGHT

        # Isotropic metric distances in camera frame to eliminate aspect ratio / rotation distortion
        f_dx = (index_tip.x - thumb_tip.x) * CAM_WIDTH
        f_dy = (index_tip.y - thumb_tip.y) * CAM_HEIGHT
        finger_dist = math.hypot(f_dx, f_dy)

        p_dx = (middle_mcp.x - wrist.x) * CAM_WIDTH
        p_dy = (middle_mcp.y - wrist.y) * CAM_HEIGHT
        palm_size = max(25.0, math.hypot(p_dx, p_dy))

        current_pinch_ratio = finger_dist / palm_size

        # Dual-threshold hysteresis prevents boundary flicker
        if not is_pinching:
            pinch_detected = (current_pinch_ratio < PINCH_START_RATIO)
        else:
            pinch_detected = (current_pinch_ratio < PINCH_END_RATIO)

        cursor_found = True
        # Natural cursor targeting: fingertip when open, pinch contact midpoint when pinched
        if pinch_detected:
            hand_x, hand_y = (ix + tx) / 2.0, (iy + ty) / 2.0
        else:
            hand_x, hand_y = ix, iy
        thumb_landmark_norm = (index_tip.x, index_tip.y)
    else:
        # Hand lost grace: keep holding state for up to 3 frames (~50ms) to bridge momentary occlusions
        if selected_ball_color is not None and lost_hand_grace < LOST_HAND_MAX_GRACE:
            lost_hand_grace += 1
            cursor_found = True
            pinch_detected = True
        else:
            cursor_found = False
            pinch_detected = False

    if cursor_found:
        # Low-lag mode: reduce smoothing while carrying a ball so the cursor
        # tightly tracks the hand instead of trailing behind by 30-50px.
        desired_beta = FILTER_BETA_HOLD if selected_ball_color is not None else FILTER_BETA_IDLE
        cursor_filter_x.beta = desired_beta
        cursor_filter_y.beta = desired_beta

        if not prev_cursor_found:
            cursor_filter_x.reset()
            cursor_filter_y.reset()
            smooth_x = cursor_filter_x.filter(hand_x, now_t)
            smooth_y = cursor_filter_y.filter(hand_y, now_t)
        else:
            smooth_x = cursor_filter_x.filter(hand_x, now_t)
            smooth_y = cursor_filter_y.filter(hand_y, now_t)
    hx, hy = int(round(smooth_x)), int(round(smooth_y))

    pinch_rising_edge = cursor_found and prev_cursor_found and pinch_detected and not is_pinching

    # ================================================================
    #  MENU STATE
    # ================================================================
    if game_state == "menu":
        hovered_idx = None
        hovered_action = None
        hovered_theme_key = None
        hovered_modal_close = False

        if menu_overlay == "leaderboard":
            close_btn = pygame.Rect(WIDTH // 2 - 60, 505, 120, 38)
            if cursor_found and close_btn.collidepoint(hx, hy):
                hovered_modal_close = True
            if pinch_rising_edge:
                if hovered_modal_close or not pygame.Rect(240, 115, 520, 445).collidepoint(hx, hy):
                    menu_overlay = None
        elif menu_overlay == "themes":
            for t_key, t_rect in THEME_CARD_RECTS.items():
                if cursor_found and t_rect.collidepoint(hx, hy):
                    hovered_theme_key = t_key
                    break
            close_btn = pygame.Rect(WIDTH // 2 - 60, 525, 120, 38)
            if cursor_found and close_btn.collidepoint(hx, hy):
                hovered_modal_close = True
            if pinch_rising_edge:
                if hovered_theme_key:
                    apply_theme(hovered_theme_key)
                elif hovered_modal_close or not pygame.Rect(170, 115, 660, 465).collidepoint(hx, hy):
                    menu_overlay = None
        else:
            if cursor_found:
                for idx, rect in MENU_BUTTONS:
                    if rect.collidepoint(hx, hy):
                        hovered_idx = idx
                        break
                if hovered_idx is None:
                    if BTN_LEADERBOARD.collidepoint(hx, hy):
                        hovered_action = "leaderboard"
                    elif BTN_THEMES.collidepoint(hx, hy):
                        hovered_action = "themes"

            if pinch_rising_edge:
                if hovered_idx is not None:
                    load_level(hovered_idx)
                    game_state = "playing"
                elif hovered_action == "leaderboard":
                    menu_overlay = "leaderboard"
                    fetch_leaderboard_async()
                elif hovered_action == "themes":
                    menu_overlay = "themes"

        is_pinching = cursor_found and pinch_detected
        prev_cursor_found = cursor_found

        draw_menu(screen, hovered_idx, hovered_action, cursor_found, player_name,
                  menu_overlay=menu_overlay, hovered_theme_key=hovered_theme_key,
                  hovered_modal_close=hovered_modal_close)

        # Draw Face Feed Window in top-left
        if show_camera_preview:
            cursor_near = (hx <= 24 + THUMB_WIDTH + 25 and hy <= 14 + THUMB_HEIGHT + 25)
            draw_thumbnail(screen, rgb_thumb, thumb_landmark_norm, pos=(24, 14), cursor_near=cursor_near)

        if cursor_found:
            ring_color = ACCENT if is_pinching else (255, 255, 255)
            pygame.draw.circle(screen, ring_color, (hx, hy), 10, width=2)
            pygame.draw.circle(screen, ring_color, (hx, hy), 3)

        pygame.display.flip()
        continue

    # ================================================================
    #  PLAYING STATE
    # ================================================================
    # Use magnetism: when holding a ball, the last-hovered tube is sticky.
    sticky = last_pinch_tube if selected_ball_color is not None else None
    hovered_tube = find_hovered_tube(hx, hy, tube_rects, sticky_idx=sticky)

    # ---------------- Interaction ----------------
    if not game_won:
        if cursor_found:
            if pinch_detected:
                if not is_pinching and selected_ball_color is None:
                    if hovered_tube is not None and tubes_data[hovered_tube]:
                        selected_ball_color = tubes_data[hovered_tube].pop()
                        source_tube_idx = hovered_tube
                is_pinching = True
                # Track the pinch position and target tube continuously so that
                # when the player opens their fingers the drop uses the LAST
                # confirmed pinch coordinate, not the post-release flicked one.
                last_pinch_hx, last_pinch_hy = hx, hy
                last_pinch_tube = hovered_tube
            else:
                if is_pinching and selected_ball_color is not None:
                    # --- RELEASE ANCHOR ---
                    # Use the anchored pre-release position and tube, not the
                    # current (post-finger-spread) coords which may have jumped
                    # 30-45px into a neighbouring tube.
                    drop_hx, drop_hy = last_pinch_hx, last_pinch_hy
                    drop_tube = last_pinch_tube

                    dropped = False
                    if drop_tube is not None and len(tubes_data[drop_tube]) < TUBE_CAPACITY:
                        top = tubes_data[drop_tube]
                        if not top or top[-1] == selected_ball_color:
                            tubes_data[drop_tube].append(selected_ball_color)
                            rect = tube_rects[drop_tube]
                            slot = len(tubes_data[drop_tube]) - 1
                            end_x, end_y = slot_position(rect, slot, cur_ball_radius)
                            drop_animations[drop_tube] = {
                                "color": selected_ball_color,
                                "slot": slot,
                                "start_x": drop_hx, "start_y": drop_hy,
                                "end_x": end_x, "end_y": end_y,
                                "t0": pygame.time.get_ticks(),
                            }
                            dropped = True
                            moves += 1
                    if not dropped:
                        # Smooth snap-back to source tube on invalid/canceled drop
                        rect = tube_rects[source_tube_idx]
                        tubes_data[source_tube_idx].append(selected_ball_color)
                        slot = len(tubes_data[source_tube_idx]) - 1
                        end_x, end_y = slot_position(rect, slot, cur_ball_radius)
                        drop_animations[source_tube_idx] = {
                            "color": selected_ball_color,
                            "slot": slot,
                            "start_x": drop_hx, "start_y": drop_hy,
                            "end_x": end_x, "end_y": end_y,
                            "t0": pygame.time.get_ticks(),
                        }
                    selected_ball_color = None
                    source_tube_idx = None
                    last_pinch_tube = None
                is_pinching = False
        else:
            if selected_ball_color is not None:
                rect = tube_rects[source_tube_idx]
                tubes_data[source_tube_idx].append(selected_ball_color)
                slot = len(tubes_data[source_tube_idx]) - 1
                end_x, end_y = slot_position(rect, slot, cur_ball_radius)
                drop_animations[source_tube_idx] = {
                    "color": selected_ball_color,
                    "slot": slot,
                    "start_x": hx, "start_y": hy,
                    "end_x": end_x, "end_y": end_y,
                    "t0": pygame.time.get_ticks(),
                }
                selected_ball_color = None
                source_tube_idx = None
                last_pinch_tube = None
            is_pinching = False
    else:
        # Freeze board when level is won, but keep pinch indicator responsive
        if selected_ball_color is not None:
            tubes_data[source_tube_idx].append(selected_ball_color)
            selected_ball_color = None
            source_tube_idx = None
        is_pinching = cursor_found and pinch_detected

    if not game_won and check_victory(tubes_data):
        game_won = True
        spawn_confetti()
        submit_score_async(f"{player_name} (Lv{current_level_idx + 1})", moves)

    # ---------------- Render ----------------
    screen.blit(BG_SURFACE, (0, 0))

    lvl_cfg = LEVELS[current_level_idx]
    title = FONT_TITLE.render(
        f"BALL SORT — {player_name} — LEVEL {current_level_idx + 1} ({lvl_cfg['tubes']} tubes)", True, TEXT_COLOR)
    screen.blit(title, (40, 30))
    sub = FONT_SUB.render("Pinch thumb + index to grab \u00b7 hover a tube \u00b7 release to drop", True, (170, 170, 195))
    screen.blit(sub, (40, 62))

    # Moves counter as a small rounded badge instead of plain text - a bit more
    # like a real HUD element and easier to spot at a glance mid-game.
    moves_txt = FONT_SUB.render(f"\u25cf {moves} moves", True, TEXT_COLOR)
    badge_w, badge_h = moves_txt.get_width() + 28, 30
    badge = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
    pygame.draw.rect(badge, (*ACCENT, 40), badge.get_rect(), border_radius=15)
    pygame.draw.rect(badge, (*ACCENT, 160), badge.get_rect(), width=1, border_radius=15)
    badge.blit(moves_txt, (14, (badge_h - moves_txt.get_height()) // 2))
    screen.blit(badge, (40, 90))

    reset_txt = FONT_SMALL.render("R  reset level    M  level menu    C  toggle camera    ESC  quit", True, (130, 130, 155))
    screen.blit(reset_txt, (40, HEIGHT - 34))

    now_ms = pygame.time.get_ticks()
    still_animating_intro = False

    for idx, rect in enumerate(tube_rects):
        active = (hovered_tube == idx) and (selected_ball_color is not None) and (len(tubes_data[idx]) < TUBE_CAPACITY)
        pygame.draw.rect(screen, SHADOW_COLOR, rect, border_bottom_left_radius=20, border_bottom_right_radius=20)

        anim = drop_animations.get(idx)
        skip_slot = anim["slot"] if anim else -1

        for b_idx, b_color in enumerate(tubes_data[idx]):
            if b_idx == skip_slot:
                continue
            bx, by = slot_position(rect, b_idx, cur_ball_radius)

            # Staggered "pop-in" entrance: each ball falls into place a little
            # after the one before it, tube by tube, so a freshly loaded level
            # feels alive instead of just appearing fully formed.
            if level_intro_active:
                delay = idx * 35 + b_idx * 45
                elapsed = now_ms - level_intro_start - delay
                if elapsed < 0:
                    still_animating_intro = True
                    continue
                if elapsed < LEVEL_INTRO_ANIM_MS:
                    still_animating_intro = True
                    t = ease_out_cubic(elapsed / LEVEL_INTRO_ANIM_MS)
                    fall_from_y = rect.top - cur_ball_radius * 3
                    draw_by = fall_from_y + (by - fall_from_y) * t
                    draw_ball(screen, b_color, bx, draw_by, cur_ball_radius, alpha=int(255 * min(1.0, t + 0.25)))
                    continue

            draw_ball(screen, b_color, bx, by, cur_ball_radius)

        draw_tube(screen, rect, active)

    if level_intro_active and not still_animating_intro:
        level_intro_active = False

    # animate the ball currently dropping into place
    finished = []
    for idx, anim in drop_animations.items():
        elapsed = pygame.time.get_ticks() - anim["t0"]
        t = ease_out_cubic(elapsed / DROP_ANIM_MS)
        cx = anim["start_x"] + (anim["end_x"] - anim["start_x"]) * t
        cy = anim["start_y"] + (anim["end_y"] - anim["start_y"]) * t
        draw_ball(screen, anim["color"], cx, cy, cur_ball_radius)
        if elapsed >= DROP_ANIM_MS:
            finished.append(idx)
    for idx in finished:
        del drop_animations[idx]

    if show_camera_preview:
        cursor_near = (hx >= WIDTH - THUMB_WIDTH - 35 and hy <= 12 + THUMB_HEIGHT + 35)
        draw_thumbnail(screen, rgb_thumb, thumb_landmark_norm, cursor_near=cursor_near)

    if selected_ball_color is not None:
        draw_ball(screen, selected_ball_color, hx, hy, cur_ball_radius, alpha=235)

    if cursor_found:
        ring_color = ACCENT if is_pinching else (255, 255, 255)
        pygame.draw.circle(screen, ring_color, (hx, hy), 10, width=2)
        pygame.draw.circle(screen, ring_color, (hx, hy), 3)

    if game_won:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 8, 20, 170))
        screen.blit(overlay, (0, 0))

        if confetti:
            update_and_draw_confetti(screen, clock.get_time() / 1000.0)

        if all_levels_complete:
            win_text = FONT_BIG.render("ALL LEVELS COMPLETE!", True, WIN_COLOR)
            screen.blit(win_text, (WIDTH // 2 - win_text.get_width() // 2, HEIGHT // 2 - 50))
            info = FONT_SUB.render("Press R to replay this level \u00b7 M for level menu", True, TEXT_COLOR)
            screen.blit(info, (WIDTH // 2 - info.get_width() // 2, HEIGHT // 2 + 10))
        else:
            win_text = FONT_BIG.render("PUZZLE SOLVED!", True, WIN_COLOR)
            screen.blit(win_text, (WIDTH // 2 - win_text.get_width() // 2, HEIGHT // 2 - 50))
            if current_level_idx + 1 < len(LEVELS):
                info = FONT_SUB.render(
                    f"Solved in {moves} moves \u00b7 press N for level {current_level_idx + 2} \u00b7 R to retry \u00b7 M for menu",
                    True, TEXT_COLOR)
            else:
                info = FONT_SUB.render(
                    f"Solved in {moves} moves \u00b7 press N to finish \u00b7 R to retry \u00b7 M for menu",
                    True, TEXT_COLOR)
            screen.blit(info, (WIDTH // 2 - info.get_width() // 2, HEIGHT // 2 + 10))

        # ---- Leaderboard status + top 5, read from the background network threads ----
        with net_lock:
            sent, ok = score_submit_status["sent"], score_submit_status["ok"]
            loaded, entries = leaderboard_data["loaded"], list(leaderboard_data["entries"])

        if not sent:
            status_txt = FONT_SMALL.render("Saving your score...", True, (170, 170, 195))
        elif ok:
            status_txt = FONT_SMALL.render("Score saved to leaderboard!", True, WIN_COLOR)
        else:
            status_txt = FONT_SMALL.render("Could not reach the leaderboard server.", True, (230, 120, 120))
        screen.blit(status_txt, (WIDTH // 2 - status_txt.get_width() // 2, HEIGHT // 2 + 40))

        board_title = FONT_SUB.render("TOP 5 SCORES", True, ACCENT)
        screen.blit(board_title, (WIDTH // 2 - board_title.get_width() // 2, HEIGHT // 2 + 68))

        if loaded:
            if entries:
                for i, entry in enumerate(entries):
                    line = f"{i + 1}. {entry.get('name', '???')} \u2014 {entry.get('moves', '?')} moves"
                    line_surf = FONT_SUB.render(line, True, TEXT_COLOR)
                    screen.blit(line_surf, (WIDTH // 2 - line_surf.get_width() // 2, HEIGHT // 2 + 94 + i * 24))
            else:
                empty_txt = FONT_SUB.render("No scores yet \u2014 be the first!", True, (170, 170, 195))
                screen.blit(empty_txt, (WIDTH // 2 - empty_txt.get_width() // 2, HEIGHT // 2 + 94))
        else:
            loading_txt = FONT_SUB.render("Loading leaderboard...", True, (170, 170, 195))
            screen.blit(loading_txt, (WIDTH // 2 - loading_txt.get_width() // 2, HEIGHT // 2 + 94))

    prev_cursor_found = cursor_found
    pygame.display.flip()

camera.release()
detector.close()
pygame.quit()
sys.exit()