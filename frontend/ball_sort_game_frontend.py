import os
import sys

# ============================================================
#  HIGH-DPI AWARENESS (Must be called before any GUI / Pygame init)
# ============================================================
# Prevents Windows Desktop Window Manager (DWM) from bilinearly stretching
# the window on 125%, 150%, or 200% scaling laptop/desktop screens, which
# causes severe blurriness. Renders 1:1 hardware pixel-perfect crisp output.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import cv2
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import random
import time
import threading
import numpy as np
import requests
import gc

# ============================================================
#  CONFIG & DESIGN SYSTEM (UI/UX Pro Max)
# ============================================================
WIDTH, HEIGHT = 1100, 740               # Crisp, spacious viewport designed for modern screens
CAM_WIDTH, CAM_HEIGHT = 640, 480
DET_WIDTH, DET_HEIGHT = 320, 240        # downscaled frame fed to the model (speed)
THUMB_WIDTH, THUMB_HEIGHT = 144, 108    # crisp 4:3 webcam preview thumbnail size

TUBE_CAPACITY = 4                       # balls per color / per full tube
EMPTY_TUBES = 2                         # spare empty tubes per level
PINCH_START_RATIO = 0.35                # Threshold to engage pinch (camera space)
PINCH_END_RATIO = 0.48                  # Threshold to release pinch (hysteresis deadband)
DROP_ANIM_MS = 220
RETURN_ANIM_MS = 360
LEVEL_INTRO_ANIM_MS = 420               # staggered ball drop-in duration
CONFETTI_COUNT = 100                    # celebratory particles

CURSOR_EDGE_MARGIN = 0.12

BACKEND_URL = "http://127.0.0.1:5050"

# ============================================================
#  THEMES SYSTEM (Calibrated via UI/UX Pro Max)
# ============================================================
THEMES = {
    "galaxy": {
        "name": "Midnight Slate",
        "bg_top": (11, 15, 25),             # Slate 950
        "bg_bottom": (17, 24, 39),          # Slate 900
        "surface": (22, 30, 46),            # Elevated Slate surface
        "surface_border": (51, 65, 85),     # Slate 700 1px border
        "accent": (6, 182, 212),            # Cyan 500
        "accent_secondary": (139, 92, 246), # Violet 500
        "tube_glass": (30, 41, 59, 90),     # Crisp translucent glass
        "tube_border": (71, 85, 105),       # Slate 600
        "tube_border_active": (56, 189, 248),# Sky 400
        "shadow_color": (5, 8, 15),
        "text_color": (241, 245, 249),      # Slate 100
        "text_muted": (148, 163, 184),      # Slate 400
        "win_color": (16, 185, 129),        # Emerald 500
        "has_stars": True,
        "ball_colors": {
            "cyan":     (6, 182, 212),
            "rose":     (244, 63, 94),
            "amber":    (245, 158, 11),
            "emerald":  (16, 185, 129),
            "violet":   (139, 92, 246),
            "sky":      (14, 165, 233),
            "orange":   (249, 115, 22),
            "fuchsia":  (217, 70, 239),
            "lime":     (132, 204, 22),
            "indigo":   (99, 102, 241),
            "teal":     (20, 184, 166),
            "gold":     (234, 179, 8),
        }
    },
    "cyber": {
        "name": "Cyberpunk Neon",
        "bg_top": (8, 14, 22),
        "bg_bottom": (15, 23, 42),
        "surface": (19, 31, 54),
        "surface_border": (45, 212, 191),
        "accent": (52, 211, 153),           # Emerald 400
        "accent_secondary": (250, 204, 21), # Yellow 400
        "tube_glass": (20, 60, 65, 90),
        "tube_border": (45, 212, 191),
        "tube_border_active": (52, 211, 153),
        "shadow_color": (4, 8, 14),
        "text_color": (240, 253, 250),
        "text_muted": (153, 246, 228),
        "win_color": (52, 211, 153),
        "has_stars": False,
        "ball_colors": {
            "mint":     (52, 211, 153),
            "cyan":     (34, 211, 238),
            "yellow":   (250, 204, 21),
            "pink":     (244, 114, 182),
            "orange":   (251, 146, 60),
            "electric": (96, 165, 250),
            "purple":   (192, 132, 252),
            "crimson":  (251, 113, 133),
            "lime":     (163, 230, 53),
            "ice":      (125, 211, 252),
            "amber":    (251, 191, 36),
            "violet":   (232, 121, 249),
        }
    },
    "nebula": {
        "name": "Deep Nebula",
        "bg_top": (19, 13, 33),
        "bg_bottom": (30, 27, 75),
        "surface": (39, 29, 68),
        "surface_border": (109, 40, 217),
        "accent": (244, 114, 182),          # Pink 400
        "accent_secondary": (96, 165, 250), # Blue 400
        "tube_glass": (60, 45, 95, 90),
        "tube_border": (139, 92, 246),
        "tube_border_active": (244, 114, 182),
        "shadow_color": (10, 6, 18),
        "text_color": (250, 245, 255),
        "text_muted": (216, 180, 254),
        "win_color": (74, 222, 128),
        "has_stars": False,
        "ball_colors": {
            "pink":     (244, 114, 182),
            "amber":    (251, 191, 36),
            "blue":     (96, 165, 250),
            "emerald":  (74, 222, 128),
            "orange":   (251, 146, 60),
            "cyan":     (34, 211, 238),
            "purple":   (192, 132, 252),
            "rose":     (251, 113, 133),
            "lime":     (163, 230, 53),
            "fuchsia":  (232, 121, 249),
            "indigo":   (129, 140, 248),
            "gold":     (250, 204, 21),
        }
    },
    "sunset": {
        "name": "Warm Sunset",
        "bg_top": (28, 16, 24),
        "bg_bottom": (49, 18, 25),
        "surface": (60, 25, 34),
        "surface_border": (159, 18, 57),
        "accent": (251, 146, 60),           # Orange 400
        "accent_secondary": (244, 63, 94),  # Rose 500
        "tube_glass": (90, 40, 45, 90),
        "tube_border": (225, 29, 72),
        "tube_border_active": (251, 146, 60),
        "shadow_color": (16, 6, 10),
        "text_color": (255, 241, 242),
        "text_muted": (254, 205, 211),
        "win_color": (250, 204, 21),
        "has_stars": False,
        "ball_colors": {
            "crimson":  (225, 29, 72),
            "gold":     (234, 179, 8),
            "tangerine":(249, 115, 22),
            "rose":     (244, 63, 94),
            "amber":    (245, 158, 11),
            "coral":    (251, 113, 133),
            "purple":   (168, 85, 247),
            "apricot":  (253, 186, 116),
            "brick":    (194, 65, 12),
            "ruby":     (190, 18, 60),
            "sky":      (56, 189, 248),
            "cream":    (254, 243, 199),
        }
    }
}

current_theme_key = "galaxy"
_t = THEMES[current_theme_key]
BACKGROUND_TOP = _t["bg_top"]
BACKGROUND_BOTTOM = _t["bg_bottom"]
SURFACE = _t["surface"]
SURFACE_BORDER = _t["surface_border"]
ACCENT = _t["accent"]
ACCENT_SECONDARY = _t["accent_secondary"]
TUBE_GLASS = _t["tube_glass"]
TUBE_BORDER = _t["tube_border"]
TUBE_BORDER_ACTIVE = _t["tube_border_active"]
SHADOW_COLOR = _t["shadow_color"]
TEXT_COLOR = _t["text_color"]
TEXT_MUTED = _t["text_muted"]
WIN_COLOR = _t["win_color"]
BALL_COLORS = _t["ball_colors"]
COLOR_LIST = list(BALL_COLORS.values())

# ------------------------------------------------------------
#  LEVEL CONFIGURATIONS
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
#  THREADED CAMERA CAPTURE
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
            else:
                time.sleep(0.01)
            time.sleep(0.002)

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()

# ============================================================
#  ASYNC HAND LANDMARKER (LIVE_STREAM)
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

try:
    detector = vision.HandLandmarker.create_from_options(_make_hand_options(_delegate))
    print("Hand landmarker: using GPU delegate.")
except Exception as e:
    print(f"GPU delegate unavailable ({e}), falling back to CPU.")
    _delegate = python.BaseOptions.Delegate.CPU
    detector = vision.HandLandmarker.create_from_options(_make_hand_options(_delegate))

camera = CameraStream()

# ============================================================
#  PYGAME SETUP & RAZOR-SHARP TYPOGRAPHY
# ============================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Motion Controlled Ball Sort")
clock = pygame.time.Clock()

FONT_TITLE = pygame.font.SysFont("segoeuisemibold,segoeui,arial", 24, bold=True)
FONT_SUB = pygame.font.SysFont("segoeui,arial", 15)
FONT_BIG = pygame.font.SysFont("segoeuisemibold,segoeui,arial", 40, bold=True)
FONT_SMALL = pygame.font.SysFont("segoeui,arial", 13)
FONT_BTN = pygame.font.SysFont("segoeuisemibold,segoeui,arial", 19, bold=True)
FONT_BTN_SUB = pygame.font.SysFont("segoeui,arial", 13)
FONT_NAME = pygame.font.SysFont("segoeuisemibold,segoeui,arial", 24, bold=True)
FONT_BADGE = pygame.font.SysFont("segoeuisemibold,segoeui,arial", 12, bold=True)
FONT_NUM = pygame.font.SysFont("segoeuisemibold,segoeui,arial", 30, bold=True)

def make_vertical_gradient(size, top_color, bottom_color):
    surf = pygame.Surface(size)
    h = size[1]
    for y in range(h):
        t = y / h
        color = tuple(int(top_color[i] + (bottom_color[i] - top_color[i]) * t) for i in range(3))
        pygame.draw.line(surf, color, (0, y), (size[0], y))
    return surf

class AmbientCosmicField:
    """
    Renders a living, dynamic ambient cosmic background with drifting stars,
    pulsing luminosities, interactive pointer constellation web, magnetic
    deflection, and floating stardust trails.
    Reuses an internal SRCALPHA surface to achieve 0 allocations per frame.
    """
    def __init__(self, width=WIDTH, height=HEIGHT, count=100):
        self.width = width
        self.height = height
        self.stars = []
        self.stardust = []
        self.last_cursor_pos = None
        self._fx_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._glow_aura = self._build_cursor_aura(48)
        self.time_accum = 0.0

        rng = random.Random(777)
        for _ in range(count):
            layer = rng.choices([0, 1, 2], weights=[0.55, 0.35, 0.10])[0]
            if layer == 0:
                base_sz = 1
                base_alpha = rng.randint(60, 110)
                speed_scale = 0.35
            elif layer == 1:
                base_sz = 2
                base_alpha = rng.randint(120, 190)
                speed_scale = 0.70
            else:
                base_sz = 3
                base_alpha = rng.randint(190, 255)
                speed_scale = 1.05

            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(4.0, 14.0) * speed_scale
            self.stars.append({
                "x": rng.uniform(0, width),
                "y": rng.uniform(0, height),
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "layer": layer,
                "size": base_sz,
                "base_alpha": base_alpha,
                "phase": rng.uniform(0, 6.28),
                "twinkle_speed": rng.uniform(1.2, 2.8),
                "is_accent": (layer == 2 and rng.random() > 0.35),
            })

    def _build_cursor_aura(self, radius):
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -2):
            alpha = int(32 * ((1.0 - r / radius) ** 1.6))
            pygame.draw.circle(s, (*ACCENT, alpha), (radius, radius), r)
        return s

    def retheme(self):
        self._glow_aura = self._build_cursor_aura(48)

    def update_and_draw(self, surf, dt, bg_surf, cursor_pos=None, interactive=True):
        self.time_accum += dt
        surf.blit(bg_surf, (0, 0))
        self._fx_surf.fill((0, 0, 0, 0))

        # Update stardust spawned from pointer motion
        if interactive and cursor_pos is not None:
            cx, cy = cursor_pos
            if self.last_cursor_pos is not None:
                dx = cx - self.last_cursor_pos[0]
                dy = cy - self.last_cursor_pos[1]
                dist = math.hypot(dx, dy)
                if dist > 3.0:
                    spawn_n = min(3, int(dist // 6) + 1)
                    for _ in range(spawn_n):
                        if len(self.stardust) < 45:
                            ang = random.uniform(0, 2 * math.pi)
                            spd = random.uniform(10.0, 32.0)
                            self.stardust.append({
                                "x": cx + random.uniform(-4, 4),
                                "y": cy + random.uniform(-4, 4),
                                "vx": math.cos(ang) * spd,
                                "vy": math.sin(ang) * spd,
                                "life": 1.0,
                                "decay": random.uniform(2.2, 3.5),
                                "size": random.uniform(2.0, 3.5),
                                "color": ACCENT if random.random() > 0.3 else (255, 255, 255)
                            })
            self.last_cursor_pos = (cx, cy)
        else:
            self.last_cursor_pos = None

        # Draw surviving stardust particles onto _fx_surf
        surviving_stardust = []
        for p in self.stardust:
            p["life"] -= p["decay"] * dt
            if p["life"] > 0:
                p["x"] += p["vx"] * dt
                p["y"] += p["vy"] * dt
                p["size"] *= max(0.2, (1.0 - dt * 0.6))
                alpha = max(0, min(255, int(220 * p["life"])))
                r, g, b = p["color"][:3]
                s_sz = max(1, int(round(p["size"])))
                pygame.draw.circle(self._fx_surf, (r, g, b, alpha), (int(p["x"]), int(p["y"])), s_sz)
                surviving_stardust.append(p)
        self.stardust = surviving_stardust

        # Interactive Constellation Web & Magnetic Deflection
        if interactive and cursor_pos is not None:
            cx, cy = cursor_pos
            self._fx_surf.blit(self._glow_aura, (cx - 48, cy - 48))

            for s in self.stars:
                dx = s["x"] - cx
                dy = s["y"] - cy
                dist = math.hypot(dx, dy)
                if dist < 125.0:
                    alpha = max(0, min(255, int((1.0 - dist / 125.0) * 155)))
                    pygame.draw.line(self._fx_surf, (*ACCENT, alpha), (cx, cy), (int(s["x"]), int(s["y"])), 1)
                    if 2.0 < dist < 85.0:
                        repel = (1.0 - dist / 85.0) * 45.0
                        s["vx"] += (dx / dist) * repel * dt
                        s["vy"] += (dy / dist) * repel * dt

        # Animate & Draw Living Stars
        now_sec = self.time_accum
        for s in self.stars:
            s["vx"] *= (1.0 - dt * 0.6)
            s["vy"] *= (1.0 - dt * 0.6)
            s["x"] += s["vx"] * dt
            s["y"] += s["vy"] * dt

            # Screen wrap
            if s["x"] < -10: s["x"] = self.width + 10
            elif s["x"] > self.width + 10: s["x"] = -10
            if s["y"] < -10: s["y"] = self.height + 10
            elif s["y"] > self.height + 10: s["y"] = -10

            twinkle = math.sin(now_sec * s["twinkle_speed"] + s["phase"])
            cur_alpha = max(15, min(255, int(s["base_alpha"] + twinkle * 55)))
            color = ACCENT if s["is_accent"] else (235, 245, 255)
            sx, sy = int(s["x"]), int(s["y"])

            if s["layer"] == 2:
                pygame.draw.circle(self._fx_surf, (*color, max(10, cur_alpha // 5)), (sx, sy), 5)
                pygame.draw.circle(self._fx_surf, (*color, cur_alpha), (sx, sy), 2)
            elif s["layer"] == 1:
                pygame.draw.circle(self._fx_surf, (*color, cur_alpha), (sx, sy), 1)
            else:
                self._fx_surf.set_at((sx, sy), (*color, cur_alpha))

        surf.blit(self._fx_surf, (0, 0))

BG_SURFACE = make_vertical_gradient((WIDTH, HEIGHT), BACKGROUND_TOP, BACKGROUND_BOTTOM)
ambient_cosmic_field = AmbientCosmicField(WIDTH, HEIGHT, count=100)


# ============================================================
#  CRISP VECTOR ICONS (Zero OS-Dependent Blurry Emoji Fonts)
# ============================================================
def draw_vector_trophy(surf, cx, cy, size=20, color=(245, 158, 11)):
    """Draws a crisp geometric trophy icon centered at (cx, cy)."""
    s = size
    cup_pts = [
        (int(cx - s * 0.42), int(cy - s * 0.42)),
        (int(cx + s * 0.42), int(cy - s * 0.42)),
        (int(cx + s * 0.30), int(cy + s * 0.06)),
        (int(cx - s * 0.30), int(cy + s * 0.06))
    ]
    pygame.draw.polygon(surf, color, cup_pts)
    pygame.draw.arc(surf, color, pygame.Rect(int(cx - s * 0.58), int(cy - s * 0.38), int(s * 0.32), int(s * 0.32)), 1.57, 4.71, width=2)
    pygame.draw.arc(surf, color, pygame.Rect(int(cx + s * 0.26), int(cy - s * 0.38), int(s * 0.32), int(s * 0.32)), 4.71, 1.57, width=2)
    pygame.draw.rect(surf, color, (int(cx - s * 0.08), int(cy + s * 0.06), int(s * 0.16), int(s * 0.26)))
    pygame.draw.rect(surf, color, (int(cx - s * 0.38), int(cy + s * 0.32), int(s * 0.76), int(s * 0.16)), border_radius=2)

def draw_vector_palette(surf, cx, cy, size=20, color=(147, 51, 234)):
    """Draws a crisp artist palette icon centered at (cx, cy)."""
    s = size
    pygame.draw.circle(surf, color, (cx, cy), int(s * 0.46), width=2)
    pygame.draw.circle(surf, (244, 63, 94), (int(cx - s * 0.2), int(cy - s * 0.16)), 3)
    pygame.draw.circle(surf, (245, 158, 11), (int(cx), int(cy - s * 0.26)), 3)
    pygame.draw.circle(surf, (6, 182, 212), (int(cx + s * 0.2), int(cy - s * 0.16)), 3)
    pygame.draw.circle(surf, (15, 23, 42), (int(cx + s * 0.1), int(cy + s * 0.16)), 4)

def draw_vector_medal(surf, cx, cy, rank=1, size=24):
    """Draws an ultra-crisp rank medal chip with rank number."""
    if rank == 1:
        medal_col = (245, 158, 11)   # Gold
        ribbon_col = (225, 29, 72)
    elif rank == 2:
        medal_col = (148, 163, 184)  # Silver
        ribbon_col = (59, 130, 246)
    elif rank == 3:
        medal_col = (180, 83, 9)     # Bronze
        ribbon_col = (16, 185, 129)
    else:
        medal_col = (100, 116, 139)
        ribbon_col = (71, 85, 105)

    pygame.draw.polygon(surf, ribbon_col, [(cx - 7, cy - 14), (cx, cy - 6), (cx - 10, cy + 2)])
    pygame.draw.polygon(surf, ribbon_col, [(cx + 7, cy - 14), (cx, cy - 6), (cx + 10, cy + 2)])
    pygame.draw.circle(surf, medal_col, (cx, cy), size // 2)
    pygame.draw.circle(surf, (255, 255, 255, 160), (cx, cy), size // 2, width=1)
    txt = FONT_BADGE.render(str(rank), True, (15, 23, 42))
    surf.blit(txt, (cx - txt.get_width() // 2, cy - txt.get_height() // 2))

def draw_vector_check(surf, cx, cy, size=14, color=(16, 185, 129)):
    pts = [
        (int(cx - size * 0.4), int(cy)),
        (int(cx - size * 0.1), int(cy + size * 0.35)),
        (int(cx + size * 0.45), int(cy - size * 0.35))
    ]
    pygame.draw.lines(surf, color, False, pts, width=2)

def draw_vector_cross(surf, cx, cy, size=14, color=(244, 63, 94)):
    d = int(size * 0.35)
    pygame.draw.line(surf, color, (cx - d, cy - d), (cx + d, cy + d), width=2)
    pygame.draw.line(surf, color, (cx + d, cy - d), (cx - d, cy + d), width=2)

def draw_vector_return(surf, cx, cy, size=14, color=(245, 158, 11)):
    pygame.draw.arc(surf, color, pygame.Rect(int(cx - size * 0.4), int(cy - size * 0.4), int(size * 0.8), int(size * 0.8)), 0.8, 5.5, width=2)
    pygame.draw.polygon(surf, color, [(int(cx - size * 0.4), int(cy - size * 0.2)), (int(cx - size * 0.4), int(cy + size * 0.15)), (int(cx - size * 0.15), int(cy - size * 0.05))])

# ============================================================
#  ONE-EURO ADAPTIVE FILTER & HITBOX GEOMETRY
# ============================================================
class OneEuroFilter:
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
    span = max(1e-6, 1.0 - 2.0 * margin)
    return min(1.0, max(0.0, (v - margin) / span))

def find_hovered_tube(hx, hy, tube_rects, sticky_idx=None):
    if not tube_rects:
        return None
    top_limit = 60
    bottom_limit = HEIGHT - 20
    if not (top_limit <= hy <= bottom_limit):
        return None

    if len(tube_rects) > 1:
        actual_gap = tube_rects[1].left - tube_rects[0].right
        magnet_px = max(6, min(16, int(actual_gap * 0.45)))
    else:
        magnet_px = 18

    if sticky_idx is not None and 0 <= sticky_idx < len(tube_rects):
        rect = tube_rects[sticky_idx]
        left_limit = (tube_rects[sticky_idx - 1].right + rect.left) / 2 if sticky_idx > 0 else rect.left - int(rect.width * 0.7)
        right_limit = (rect.right + tube_rects[sticky_idx + 1].left) / 2 if sticky_idx < len(tube_rects) - 1 else rect.right + int(rect.width * 0.7)
        if (left_limit - magnet_px) <= hx <= (right_limit + magnet_px):
            return sticky_idx

    for idx, rect in enumerate(tube_rects):
        left_limit = (tube_rects[idx - 1].right + rect.left) / 2 if idx > 0 else rect.left - int(rect.width * 0.7)
        right_limit = (rect.right + tube_rects[idx + 1].left) / 2 if idx < len(tube_rects) - 1 else rect.right + int(rect.width * 0.7)
        if left_limit <= hx <= right_limit:
            return idx
    return None

def find_nearest_tube(hx, hy, tube_rects):
    if not tube_rects:
        return None
    best_idx = None
    min_dist = float('inf')
    for idx, rect in enumerate(tube_rects):
        dist = abs(hx - rect.centerx)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
    if best_idx is not None:
        rect = tube_rects[best_idx]
        reach = rect.width * 1.3
        if rect.left - reach <= hx <= rect.right + reach:
            return best_idx
    return None

def get_drop_validation(tube_idx, tubes_data, carried_color, source_idx=None):
    if tube_idx is None or tube_idx < 0 or tube_idx >= len(tubes_data):
        return 'none', "No tube selected"
    if tube_idx == source_idx:
        return 'same_source', "Return ball"
    t_balls = tubes_data[tube_idx]
    if len(t_balls) >= TUBE_CAPACITY:
        return 'full', "Tube is full (max 4)"
    if t_balls and t_balls[-1] != carried_color:
        return 'mismatch', "Color mismatch"
    return 'valid', "Drop here"

# ============================================================
#  DYNAMIC LAYOUT
# ============================================================
def compute_layout(num_tubes):
    margin = 48
    available_width = WIDTH - margin * 2
    gap_ratio = 0.36
    max_tube_width = 88
    min_tube_width = 36

    denom = num_tubes + gap_ratio * (num_tubes - 1)
    tube_width = available_width / denom
    tube_width = max(min_tube_width, min(max_tube_width, tube_width))
    gap = tube_width * gap_ratio

    ball_radius = max(12, int(tube_width * 0.33))
    tube_height = TUBE_CAPACITY * ball_radius * 2 + 46

    total_width = num_tubes * tube_width + (num_tubes - 1) * gap
    start_x = (WIDTH - total_width) / 2
    tube_y = 310

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

def ease_out_back(t, s=1.70158):
    t = max(0.0, min(1.0, t))
    t = t - 1
    return t * t * ((s + 1) * t + s) + 1

_ball_surface_cache = {}

def get_ball_surface(color, radius):
    """Generates ultra-sharp, anti-aliased 3D sphere graphic with specular sheen."""
    key = (color, radius)
    s = _ball_surface_cache.get(key)
    if s is None:
        pad = 4
        size = radius * 2 + pad * 2
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2
        # Base shadow ring
        pygame.draw.circle(s, darken(color, 0.40), (cx, cy), radius)
        # Main body
        pygame.draw.circle(s, color, (cx, cy), radius - 1)
        # Specular curvature illumination
        hl_col = lighten(color, 0.45)
        pygame.draw.circle(s, hl_col, (cx - radius // 3, cy - radius // 3), max(3, int(radius * 0.42)))
        # Direct crisp highlight
        pygame.draw.circle(s, (255, 255, 255), (cx - radius // 2, cy - radius // 2), max(2, int(radius * 0.18)))
        _ball_surface_cache[key] = s
    return s

def draw_ball(surf, color, x, y, radius, alpha=255):
    x, y = int(round(x)), int(round(y))
    ball_surf = get_ball_surface(color, radius)
    offset = ball_surf.get_width() // 2
    if alpha >= 255:
        pygame.draw.ellipse(surf, (0, 0, 0, 70), pygame.Rect(x - radius, y + radius - 6, radius * 2, 9))
        surf.blit(ball_surf, (x - offset, y - offset))
    else:
        temp = ball_surf.copy()
        temp.set_alpha(alpha)
        surf.blit(temp, (x - offset, y - offset))

_tube_glass_cache = {}
_tube_status_glow_cache = {}

def _get_tube_glass(w, h):
    key = (w, h)
    s = _tube_glass_cache.get(key)
    if s is None:
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, TUBE_GLASS, s.get_rect(), border_bottom_left_radius=18, border_bottom_right_radius=18)
        # Crisp vertical reflection highlight
        pygame.draw.line(s, (255, 255, 255, 45), (4, 6), (4, h - 16), 2)
        _tube_glass_cache[key] = s
    return s

def _get_tube_glow_surf(w, h, glow_color):
    key = (w, h, glow_color)
    s = _tube_status_glow_cache.get(key)
    if s is None:
        s = pygame.Surface((w + 20, h + 16), pygame.SRCALPHA)
        pygame.draw.rect(s, (*glow_color, 45), s.get_rect(), border_radius=18)
        pygame.draw.rect(s, (*glow_color, 95), s.get_rect(), width=2, border_radius=18)
        _tube_status_glow_cache[key] = s
    return s

def draw_tube(surf, rect, status=None, ball_radius=22):
    """Draws crisp laboratory tube with capacity notches and status-aware glow."""
    surf.blit(_get_tube_glass(rect.width, rect.height), rect.topleft)

    # Tube interior graduation marks
    for slot in range(1, TUBE_CAPACITY):
        notch_y = rect.bottom - (slot * (ball_radius * 2)) - 5
        pygame.draw.line(surf, (*TUBE_BORDER, 70), (rect.left + 8, notch_y), (rect.right - 8, notch_y), 1)

    if status == 'valid':
        border_color = (16, 185, 129)   # Emerald
        glow_color = (16, 185, 129)
        width = 4
    elif status == 'invalid':
        border_color = (244, 63, 94)    # Rose
        glow_color = (244, 63, 94)
        width = 4
    elif status == 'active':
        border_color = TUBE_BORDER_ACTIVE
        glow_color = TUBE_BORDER_ACTIVE
        width = 4
    else:
        border_color = TUBE_BORDER
        glow_color = None
        width = 2

    if glow_color is not None:
        pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 150.0))
        glow = _get_tube_glow_surf(rect.width, rect.height, glow_color)
        glow.set_alpha(int(140 + 100 * pulse))
        surf.blit(glow, (rect.x - 10, rect.y - 8))

    # Curved test-tube rim lip
    pygame.draw.ellipse(surf, border_color, (rect.left - 2, rect.top - 4, rect.width + 4, 8), width=width)

    # Main tube frame
    pygame.draw.lines(surf, border_color, False, [
        (rect.left, rect.top),
        (rect.left, rect.bottom - 16),
        (rect.left + 16, rect.bottom),
        (rect.right - 16, rect.bottom),
        (rect.right, rect.bottom - 16),
        (rect.right, rect.top)
    ], width=width)

def draw_tube_badge(surf, rect, text, bg_color, border_color, badge_type="check"):
    """Draws pill validation badge with vector icon above the hovered tube."""
    txt_surf = FONT_BADGE.render(text.upper(), True, (255, 255, 255))
    icon_w = 16
    bw, bh = txt_surf.get_width() + icon_w + 22, 26
    bx = rect.centerx - bw // 2
    by = rect.top - bh - 8

    bg_s = pygame.Surface((bw, bh), pygame.SRCALPHA)
    pygame.draw.rect(bg_s, (*bg_color, 240), bg_s.get_rect(), border_radius=13)
    pygame.draw.rect(bg_s, (*border_color, 255), bg_s.get_rect(), width=1, border_radius=13)

    icon_cx = 14
    icon_cy = bh // 2
    if badge_type == "check":
        draw_vector_check(bg_s, icon_cx, icon_cy, size=11, color=(255, 255, 255))
    elif badge_type == "cross":
        draw_vector_cross(bg_s, icon_cx, icon_cy, size=11, color=(255, 255, 255))
    elif badge_type == "return":
        draw_vector_return(bg_s, icon_cx, icon_cy, size=11, color=(255, 255, 255))

    bg_s.blit(txt_surf, (icon_cx + 10, (bh - txt_surf.get_height()) // 2))
    surf.blit(bg_s, (bx, by))

shaking_tubes = {}
floating_toasts = []

def trigger_tube_shake(tube_idx):
    if tube_idx is not None:
        shaking_tubes[tube_idx] = {
            "t0": pygame.time.get_ticks(),
            "duration_ms": 240
        }

def add_toast(text, x, y, color=(244, 63, 94)):
    x = max(150, min(WIDTH - 150, int(x)))
    y = max(100, min(HEIGHT - 60, int(y)))
    floating_toasts.append({
        "text": text,
        "x": x,
        "y": y,
        "t0": pygame.time.get_ticks(),
        "duration_ms": 1200,
        "color": color
    })

def update_and_draw_toasts(surf):
    now = pygame.time.get_ticks()
    alive = []
    for toast in floating_toasts:
        elapsed = now - toast["t0"]
        if elapsed < toast["duration_ms"]:
            alive.append(toast)
            progress = elapsed / toast["duration_ms"]
            cur_y = toast["y"] - int(progress * 28)
            alpha = int(255 * (1.0 - max(0.0, (progress - 0.4) / 0.6)))

            txt_s = FONT_BTN_SUB.render(toast["text"], True, (255, 255, 255))
            pad_x, pad_y = 16, 6
            tw, th = txt_s.get_width() + pad_x * 2, txt_s.get_height() + pad_y * 2
            toast_surf = pygame.Surface((tw, th), pygame.SRCALPHA)
            pygame.draw.rect(toast_surf, (*toast["color"], min(240, alpha)), toast_surf.get_rect(), border_radius=12)
            pygame.draw.rect(toast_surf, (255, 255, 255, min(210, alpha)), toast_surf.get_rect(), width=1, border_radius=12)
            txt_s.set_alpha(alpha)
            toast_surf.blit(txt_s, (pad_x, pad_y))
            surf.blit(toast_surf, (toast["x"] - tw // 2, cur_y - th // 2))
    floating_toasts[:] = alive

def slot_position(rect, slot_idx, radius):
    x = rect.centerx
    y = rect.bottom - (slot_idx * (radius * 2)) - radius - 5
    return x, y

_THUMB_SURF = pygame.Surface((THUMB_WIDTH, THUMB_HEIGHT))
_THUMB_CONTAINER = pygame.Surface((THUMB_WIDTH, THUMB_HEIGHT), pygame.SRCALPHA)
_THUMB_BORDER = pygame.Surface((THUMB_WIDTH, THUMB_HEIGHT), pygame.SRCALPHA)

def draw_thumbnail(surf, rgb_small, landmark_pt, pos=None, cursor_near=False, hand_detected=False):
    """Draws razor-sharp camera thumbnail in top-right with live tracking indicator."""
    if pos is None:
        box_x, box_y = WIDTH - THUMB_WIDTH - 30, 18
    else:
        box_x, box_y = pos
    alpha = 95 if cursor_near else 245

    _THUMB_CONTAINER.fill((0, 0, 0, 0))
    if rgb_small is not None:
        pygame.surfarray.blit_array(_THUMB_SURF, np.transpose(rgb_small, (1, 0, 2)))
        _THUMB_CONTAINER.blit(_THUMB_SURF, (0, 0))
        if landmark_pt is not None:
            lx = landmark_pt[0] * THUMB_WIDTH
            ly = landmark_pt[1] * THUMB_HEIGHT
            pygame.draw.circle(_THUMB_CONTAINER, ACCENT, (int(lx), int(ly)), 4)
            pygame.draw.circle(_THUMB_CONTAINER, (255, 255, 255), (int(lx), int(ly)), 2)
    else:
        _THUMB_CONTAINER.fill((15, 23, 42))

    _THUMB_CONTAINER.set_alpha(alpha)
    surf.blit(_THUMB_CONTAINER, (box_x, box_y))

    # Outer border and CAM tag
    _THUMB_BORDER.fill((0, 0, 0, 0))
    frame_col = ACCENT if hand_detected else TUBE_BORDER
    pygame.draw.rect(_THUMB_BORDER, (*frame_col, alpha), _THUMB_BORDER.get_rect(), width=2, border_radius=10)
    pygame.draw.rect(_THUMB_BORDER, (11, 15, 25, min(alpha, 220)), (4, 4, 46, 18), border_radius=6)
    status_dot = (16, 185, 129) if hand_detected else (245, 158, 11)
    pygame.draw.circle(_THUMB_BORDER, status_dot, (11, 13), 3)
    badge = FONT_BADGE.render("CAM", True, TEXT_COLOR)
    _THUMB_BORDER.blit(badge, (19, 5))
    surf.blit(_THUMB_BORDER, (box_x, box_y))

# ============================================================
#  LEVEL-SELECT MENU LAYOUT
# ============================================================
def get_menu_buttons():
    buttons = []
    btn_w, btn_h = 190, 105
    gap_x, gap_y = 26, 20
    row1_w = 4 * btn_w + 3 * gap_x
    start_x_4 = (WIDTH - row1_w) // 2

    row3_w = 2 * btn_w + 1 * gap_x
    start_x_2 = (WIDTH - row3_w) // 2

    base_y = 155
    for i in range(10):
        if i < 4:  # Row 1: Lv 1, 2, 3, 4
            x = start_x_4 + i * (btn_w + gap_x)
            y = base_y
        elif i < 8:  # Row 2: Lv 5, 6, 7, 8
            x = start_x_4 + (i - 4) * (btn_w + gap_x)
            y = base_y + btn_h + gap_y
        else:  # Row 3: Lv 9, 10
            x = start_x_2 + (i - 8) * (btn_w + gap_x)
            y = base_y + 2 * (btn_h + gap_y)
        buttons.append((i, pygame.Rect(x, y, btn_w, btn_h)))
    return buttons

MENU_BUTTONS = get_menu_buttons()

BTN_LEADERBOARD = pygame.Rect(205, 560, 210, 48)
BTN_THEMES = pygame.Rect(435, 560, 190, 48)
INDICATOR_HAND = pygame.Rect(645, 560, 250, 48)

THEME_CARD_RECTS = {
    "galaxy": pygame.Rect(210, 185, 320, 130),
    "cyber":  pygame.Rect(565, 185, 320, 130),
    "nebula": pygame.Rect(210, 335, 320, 130),
    "sunset": pygame.Rect(565, 335, 320, 130),
}

def _build_menu_button_art():
    art = {}
    for idx, rect in MENU_BUTTONS:
        cfg = LEVELS[idx]
        base = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(base, (*SURFACE, 235), base.get_rect(), border_radius=14)
        pygame.draw.rect(base, SURFACE_BORDER, base.get_rect(), width=1, border_radius=14)

        # Key pill tag
        key_char = str(idx + 1 if idx < 9 else 0)
        key_pill = pygame.Surface((30, 20), pygame.SRCALPHA)
        pygame.draw.rect(key_pill, (*ACCENT, 45), key_pill.get_rect(), border_radius=6)
        pygame.draw.rect(key_pill, (*ACCENT, 180), key_pill.get_rect(), width=1, border_radius=6)
        pill_txt = FONT_BADGE.render(f"[{key_char}]", True, ACCENT)
        key_pill.blit(pill_txt, ((30 - pill_txt.get_width()) // 2, 3))
        base.blit(key_pill, (rect.width - 40, 12))

        # Title
        lvl_num = FONT_NUM.render(f"{idx + 1:02d}", True, (*ACCENT, 180))
        base.blit(lvl_num, (18, 14))
        lvl_txt = FONT_BTN.render(f"LEVEL {idx + 1}", True, TEXT_COLOR)
        base.blit(lvl_txt, (64, 20))

        # Stats
        tubes_txt = FONT_BTN_SUB.render(f"• {cfg['tubes']} tubes", True, TEXT_MUTED)
        base.blit(tubes_txt, (20, 56))
        colors_txt = FONT_BTN_SUB.render(f"• {cfg['colors']} colors", True, TEXT_MUTED)
        base.blit(colors_txt, (20, 76))
        art[idx] = base
    return art

def _build_menu_glow(w, h):
    s = pygame.Surface((w + 16, h + 16), pygame.SRCALPHA)
    pygame.draw.rect(s, (*TUBE_BORDER_ACTIVE, 45), s.get_rect(), border_radius=18)
    return s

MENU_BUTTON_ART = _build_menu_button_art()
MENU_GLOW = _build_menu_glow(190, 105)
MENU_FOOTER_SURF = FONT_SMALL.render("ESC  Quit    ·    1-9, 0  Select Level    ·    L  Leaderboard    ·    T  Themes    ·    C  Camera Toggle", True, (148, 163, 184))

def apply_theme(theme_key):
    global current_theme_key, BACKGROUND_TOP, BACKGROUND_BOTTOM, SURFACE, SURFACE_BORDER, ACCENT, ACCENT_SECONDARY
    global TUBE_GLASS, TUBE_BORDER, TUBE_BORDER_ACTIVE, SHADOW_COLOR, TEXT_COLOR, TEXT_MUTED, WIN_COLOR
    global BALL_COLORS, COLOR_LIST, BG_SURFACE, MENU_BUTTON_ART, MENU_GLOW
    if theme_key not in THEMES:
        return
    current_theme_key = theme_key
    t = THEMES[theme_key]
    BACKGROUND_TOP = t["bg_top"]
    BACKGROUND_BOTTOM = t["bg_bottom"]
    SURFACE = t["surface"]
    SURFACE_BORDER = t["surface_border"]
    ACCENT = t["accent"]
    ACCENT_SECONDARY = t["accent_secondary"]
    TUBE_GLASS = t["tube_glass"]
    TUBE_BORDER = t["tube_border"]
    TUBE_BORDER_ACTIVE = t["tube_border_active"]
    SHADOW_COLOR = t["shadow_color"]
    TEXT_COLOR = t["text_color"]
    TEXT_MUTED = t["text_muted"]
    WIN_COLOR = t["win_color"]
    BALL_COLORS = t["ball_colors"]
    COLOR_LIST = list(BALL_COLORS.values())

    BG_SURFACE = make_vertical_gradient((WIDTH, HEIGHT), BACKGROUND_TOP, BACKGROUND_BOTTOM)
    ambient_cosmic_field.retheme()

    MENU_BUTTON_ART = _build_menu_button_art()
    MENU_GLOW = _build_menu_glow(190, 105)
    _tube_glass_cache.clear()
    _tube_status_glow_cache.clear()
    _ball_surface_cache.clear()

def draw_action_button(surf, rect, label, icon_type, is_hover=False, active=False):
    """Draws tactile action button with vector icon."""
    btn_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg = (*ACCENT, 65) if active else ((45, 55, 78, 180) if is_hover else (*SURFACE, 210))
    border = ACCENT if (is_hover or active) else SURFACE_BORDER
    pygame.draw.rect(btn_surf, bg, btn_surf.get_rect(), border_radius=12)
    pygame.draw.rect(btn_surf, border, btn_surf.get_rect(), width=2 if is_hover else 1, border_radius=12)

    icon_cx = 30
    icon_cy = rect.height // 2
    if icon_type == "trophy":
        draw_vector_trophy(btn_surf, icon_cx, icon_cy, size=20, color=ACCENT if (is_hover or active) else (245, 158, 11))
    elif icon_type == "palette":
        draw_vector_palette(btn_surf, icon_cx, icon_cy, size=20, color=ACCENT if (is_hover or active) else ACCENT_SECONDARY)

    txt = FONT_BTN.render(label, True, TEXT_COLOR)
    btn_surf.blit(txt, (52, (rect.height - txt.get_height()) // 2))
    surf.blit(btn_surf, rect.topleft)

def draw_hand_indicator(surf, rect, hand_detected):
    """Draws live connection status pill with pulsing indicator."""
    pill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg_color = (16, 45, 30, 200) if hand_detected else (45, 20, 25, 200)
    border_color = (16, 185, 129) if hand_detected else (244, 63, 94)
    dot_color = (16, 185, 129) if hand_detected else (244, 63, 94)
    text_color = (209, 250, 229) if hand_detected else (254, 205, 211)
    txt_str = "Tracking: Active" if hand_detected else "Tracking: Searching..."

    pygame.draw.rect(pill, bg_color, pill.get_rect(), border_radius=12)
    pygame.draw.rect(pill, border_color, pill.get_rect(), width=1, border_radius=12)

    pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 150.0))
    radius = int(5 * pulse) if hand_detected else 4
    pygame.draw.circle(pill, dot_color, (22, rect.height // 2), radius)
    txt = FONT_SUB.render(txt_str, True, text_color)
    pill.blit(txt, (38, (rect.height - txt.get_height()) // 2))
    surf.blit(pill, rect.topleft)

def draw_leaderboard_modal(surf, close_hover=False):
    """Draws Global Leaderboard modal with vector medals and empty/offline states."""
    dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    dim.fill((6, 9, 16, 215))
    surf.blit(dim, (0, 0))

    modal_rect = pygame.Rect(WIDTH // 2 - 290, 110, 580, 490)
    card = pygame.Surface((modal_rect.width, modal_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(card, (*SURFACE, 250), card.get_rect(), border_radius=18)
    pygame.draw.rect(card, SURFACE_BORDER, card.get_rect(), width=1, border_radius=18)

    draw_vector_trophy(card, modal_rect.width // 2 - 150, 36, size=24, color=ACCENT)
    title = FONT_TITLE.render("GLOBAL LEADERBOARD", True, TEXT_COLOR)
    card.blit(title, (modal_rect.width // 2 - title.get_width() // 2 + 10, 24))
    sub = FONT_SMALL.render("Fewest moves wins · Verified local records", True, TEXT_MUTED)
    card.blit(sub, (modal_rect.width // 2 - sub.get_width() // 2, 56))

    header_y = 92
    pygame.draw.rect(card, (15, 23, 42, 200), (28, header_y, modal_rect.width - 56, 30), border_radius=6)
    lbl_rank = FONT_BADGE.render("RANK", True, TEXT_MUTED)
    lbl_name = FONT_BADGE.render("PLAYER", True, TEXT_MUTED)
    lbl_moves = FONT_BADGE.render("RECORD", True, TEXT_MUTED)
    card.blit(lbl_rank, (46, header_y + 8))
    card.blit(lbl_name, (110, header_y + 8))
    card.blit(lbl_moves, (modal_rect.width - 110, header_y + 8))

    with net_lock:
        loaded = leaderboard_data["loaded"]
        entries = list(leaderboard_data["entries"])

    if loaded and entries:
        for i, entry in enumerate(entries[:5]):
            row_y = 132 + i * 52
            row_rect = pygame.Rect(28, row_y, modal_rect.width - 56, 44)
            pygame.draw.rect(card, (30, 41, 59, 160), row_rect, border_radius=8)

            draw_vector_medal(card, 58, row_y + 22, rank=i+1, size=24)

            name_str = str(entry.get("name", "Player"))[:16]
            name_txt = FONT_BTN.render(name_str, True, TEXT_COLOR)
            card.blit(name_txt, (108, row_y + (44 - name_txt.get_height()) // 2))

            moves_str = f"{entry.get('moves', '?')} moves"
            moves_txt = FONT_BTN.render(moves_str, True, ACCENT)
            card.blit(moves_txt, (modal_rect.width - 44 - moves_txt.get_width(), row_y + (44 - moves_txt.get_height()) // 2))
    elif loaded:
        draw_vector_trophy(card, modal_rect.width // 2, 200, size=36, color=SURFACE_BORDER)
        empty_title = FONT_BTN.render("No Scores Recorded Yet", True, TEXT_COLOR)
        card.blit(empty_title, (modal_rect.width // 2 - empty_title.get_width() // 2, 240))
        empty_sub = FONT_SMALL.render("Be the first to solve a level and claim the #1 spot!", True, TEXT_MUTED)
        card.blit(empty_sub, (modal_rect.width // 2 - empty_sub.get_width() // 2, 270))
    else:
        loading_txt = FONT_SUB.render("Connecting to leaderboard server...", True, TEXT_MUTED)
        card.blit(loading_txt, (modal_rect.width // 2 - loading_txt.get_width() // 2, 220))

    surf.blit(card, modal_rect.topleft)

    close_rect = pygame.Rect(WIDTH // 2 - 70, modal_rect.bottom - 54, 140, 38)
    close_surf = pygame.Surface((close_rect.width, close_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(close_surf, (*ACCENT, 90) if close_hover else (30, 41, 59, 230), close_surf.get_rect(), border_radius=10)
    pygame.draw.rect(close_surf, ACCENT if close_hover else SURFACE_BORDER, close_surf.get_rect(), width=1, border_radius=10)
    cl_txt = FONT_BTN_SUB.render("CLOSE", True, TEXT_COLOR)
    close_surf.blit(cl_txt, (close_rect.width // 2 - cl_txt.get_width() // 2, (close_rect.height - cl_txt.get_height()) // 2))
    surf.blit(close_surf, close_rect.topleft)

def draw_themes_modal(surf, hovered_theme=None, close_hover=False):
    """Draws Themes Modal with live swatch dots."""
    dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    dim.fill((6, 9, 16, 215))
    surf.blit(dim, (0, 0))

    modal_rect = pygame.Rect(WIDTH // 2 - 360, 110, 720, 490)
    card = pygame.Surface((modal_rect.width, modal_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(card, (*SURFACE, 250), card.get_rect(), border_radius=18)
    pygame.draw.rect(card, SURFACE_BORDER, card.get_rect(), width=1, border_radius=18)

    draw_vector_palette(card, modal_rect.width // 2 - 140, 36, size=24, color=ACCENT)
    title = FONT_TITLE.render("VISUAL THEMES", True, TEXT_COLOR)
    card.blit(title, (modal_rect.width // 2 - title.get_width() // 2 + 10, 24))
    sub = FONT_SMALL.render("Pinch or click any palette to switch aesthetics immediately", True, TEXT_MUTED)
    card.blit(sub, (modal_rect.width // 2 - sub.get_width() // 2, 56))

    surf.blit(card, modal_rect.topleft)

    for t_key, t_rect in THEME_CARD_RECTS.items():
        t = THEMES[t_key]
        is_cur = (t_key == current_theme_key)
        is_hov = (t_key == hovered_theme)

        c_surf = pygame.Surface((t_rect.width, t_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(c_surf, (*t["bg_top"], 240), c_surf.get_rect(), border_radius=14)
        border_col = t["accent"] if (is_cur or is_hov) else t["tube_border"]
        border_w = 2 if (is_cur or is_hov) else 1
        pygame.draw.rect(c_surf, border_col, c_surf.get_rect(), width=border_w, border_radius=14)

        name_txt = FONT_BTN.render(t["name"], True, t["text_color"])
        c_surf.blit(name_txt, (20, 18))

        if is_cur:
            badge = FONT_BADGE.render("ACTIVE", True, t["accent"])
            c_surf.blit(badge, (t_rect.width - badge.get_width() - 18, 20))
            draw_vector_check(c_surf, t_rect.width - badge.get_width() - 30, 26, size=11, color=t["accent"])
        elif is_hov:
            badge = FONT_BADGE.render("SELECT", True, ACCENT)
            c_surf.blit(badge, (t_rect.width - badge.get_width() - 18, 20))

        sub_desc = "Deep space with celestial stars" if t.get("has_stars") else f"Vibrant {t['name']} palette"
        desc_txt = FONT_BTN_SUB.render(sub_desc, True, TEXT_MUTED)
        c_surf.blit(desc_txt, (20, 48))

        ball_colors_sample = list(t["ball_colors"].values())[:6]
        for c_idx, c_rgb in enumerate(ball_colors_sample):
            dot_x = 28 + c_idx * 32
            dot_y = 92
            pygame.draw.circle(c_surf, c_rgb, (dot_x, dot_y), 11)
            pygame.draw.circle(c_surf, (255, 255, 255, 120), (dot_x - 3, dot_y - 3), 3)

        surf.blit(c_surf, t_rect.topleft)

    close_rect = pygame.Rect(WIDTH // 2 - 70, modal_rect.bottom - 54, 140, 38)
    close_surf = pygame.Surface((close_rect.width, close_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(close_surf, (*ACCENT, 90) if close_hover else (30, 41, 59, 230), close_surf.get_rect(), border_radius=10)
    pygame.draw.rect(close_surf, ACCENT if close_hover else SURFACE_BORDER, close_surf.get_rect(), width=1, border_radius=10)
    cl_txt = FONT_BTN_SUB.render("CLOSE", True, TEXT_COLOR)
    close_surf.blit(cl_txt, (close_rect.width // 2 - cl_txt.get_width() // 2, (close_rect.height - cl_txt.get_height()) // 2))
    surf.blit(close_surf, close_rect.topleft)

def draw_menu(surf, hovered_idx, hovered_action, cursor_active, player_name, menu_overlay=None, hovered_theme_key=None, hovered_modal_close=False, cursor_pos=None, dt=0.016):
    """Draws level menu with living dynamic background, interactive pointer constellations, and elevated cards."""
    ambient_cosmic_field.update_and_draw(surf, dt, BG_SURFACE, cursor_pos=cursor_pos, interactive=True)

    # Top Left Badge Chip
    badge_surf = pygame.Surface((140, 30), pygame.SRCALPHA)
    pygame.draw.rect(badge_surf, (*ACCENT, 30), badge_surf.get_rect(), border_radius=15)
    pygame.draw.rect(badge_surf, (*ACCENT, 140), badge_surf.get_rect(), width=1, border_radius=15)
    b_txt = FONT_BADGE.render("10 PUZZLE LEVELS", True, ACCENT)
    badge_surf.blit(b_txt, ((140 - b_txt.get_width()) // 2, 7))
    surf.blit(badge_surf, (40, 32))

    # Center Title & Subtitle
    title = FONT_TITLE.render("BALL SORT PUZZLE", True, TEXT_COLOR)
    surf.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))
    sub = FONT_SUB.render("Select a level  ·  Pinch thumb + index or click", True, TEXT_MUTED)
    surf.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 68))

    # Top Right Player Card
    player_lbl = FONT_SUB.render(f"Player: {player_name}", True, TEXT_COLOR)
    pill_w = player_lbl.get_width() + 36
    pill_h = 34
    pill_x = WIDTH - pill_w - 40
    pill_y = 30
    pill_surf = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
    pygame.draw.rect(pill_surf, (*SURFACE, 230), pill_surf.get_rect(), border_radius=17)
    pygame.draw.rect(pill_surf, SURFACE_BORDER, pill_surf.get_rect(), width=1, border_radius=17)
    pygame.draw.circle(pill_surf, ACCENT, (16, pill_h // 2), 5)
    surf.blit(pill_surf, (pill_x, pill_y))
    surf.blit(player_lbl, (pill_x + 28, pill_y + (pill_h - player_lbl.get_height()) // 2))

    # Level Grid with Interactive Magnetic Proximity Aura
    for idx, rect in MENU_BUTTONS:
        is_hover = (idx == hovered_idx and menu_overlay is None)
        if is_hover:
            surf.blit(MENU_GLOW, (rect.x - 8, rect.y - 8))
        elif cursor_pos is not None and menu_overlay is None:
            cx, cy = rect.center
            dist = math.hypot(cursor_pos[0] - cx, cursor_pos[1] - cy)
            if dist < 135:
                prox = 1.0 - dist / 135.0
                prox_s = pygame.Surface((rect.width + 12, rect.height + 12), pygame.SRCALPHA)
                pygame.draw.rect(prox_s, (*ACCENT, int(prox * 50)), prox_s.get_rect(), border_radius=16)
                surf.blit(prox_s, (rect.x - 6, rect.y - 6))

        surf.blit(MENU_BUTTON_ART[idx], rect.topleft)
        border_color = TUBE_BORDER_ACTIVE if is_hover else SURFACE_BORDER
        pygame.draw.rect(surf, border_color, rect, width=2 if is_hover else 1, border_radius=14)

    # Bottom Action Bar
    draw_action_button(surf, BTN_LEADERBOARD, "Leaderboard", "trophy", is_hover=(hovered_action == "leaderboard" and menu_overlay is None), active=(menu_overlay == "leaderboard"))
    draw_action_button(surf, BTN_THEMES, "Themes", "palette", is_hover=(hovered_action == "themes" and menu_overlay is None), active=(menu_overlay == "themes"))
    draw_hand_indicator(surf, INDICATOR_HAND, cursor_active)

    surf.blit(MENU_FOOTER_SURF, (WIDTH // 2 - MENU_FOOTER_SURF.get_width() // 2, 696))

    if menu_overlay == "leaderboard":
        draw_leaderboard_modal(surf, close_hover=hovered_modal_close)
    elif menu_overlay == "themes":
        draw_themes_modal(surf, hovered_theme=hovered_theme_key, close_hover=hovered_modal_close)

# ============================================================
#  BACKEND NETWORKING
# ============================================================
net_lock = threading.Lock()
score_submit_status = {"sent": False, "ok": False}
leaderboard_data = {"loaded": False, "entries": []}

_net_session = requests.Session()
_net_session.trust_env = False

def ensure_backend_running():
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
                except Exception:
                    time.sleep(0.2)
            if ok:
                break

        with net_lock:
            score_submit_status["sent"] = True
            score_submit_status["ok"] = ok

        fetch_leaderboard_async()

    threading.Thread(target=worker, daemon=True).start()

def fetch_leaderboard_async():
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
#  NAME ENTRY SCREEN
# ============================================================
def run_name_entry():
    pygame.key.start_text_input()
    name = ""
    start_btn = pygame.Rect(WIDTH // 2 - 95, 455, 190, 46)

    while True:
        clock.tick(30)
        mouse_pos = pygame.mouse.get_pos()
        btn_hover = start_btn.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.release()
                detector.close()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.TEXTINPUT:
                if len(name) < 16:
                    name += event.text
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_hover or pygame.Rect(WIDTH // 2 - 210, 330, 420, 54).collidepoint(mouse_pos):
                    pygame.key.stop_text_input()
                    pygame.event.clear()
                    return name.strip() if name.strip() else "Player"
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

        ambient_cosmic_field.update_and_draw(screen, 0.033, BG_SURFACE, cursor_pos=mouse_pos, interactive=True)

        # Centered Onboarding Card
        card_rect = pygame.Rect(WIDTH // 2 - 280, 175, 560, 380)
        card = pygame.Surface((card_rect.width, card_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(card, (*SURFACE, 245), card.get_rect(), border_radius=20)
        pygame.draw.rect(card, SURFACE_BORDER, card.get_rect(), width=1, border_radius=20)

        # Decorative Mini Ball Trio
        pygame.draw.circle(card, (6, 182, 212), (card_rect.width // 2 - 22, 44), 11)
        pygame.draw.circle(card, (244, 63, 94), (card_rect.width // 2, 44), 11)
        pygame.draw.circle(card, (245, 158, 11), (card_rect.width // 2 + 22, 44), 11)

        title = FONT_TITLE.render("WELCOME TO BALL SORT", True, TEXT_COLOR)
        card.blit(title, (card_rect.width // 2 - title.get_width() // 2, 76))
        sub = FONT_SUB.render("Enter your player callsign to record high scores", True, TEXT_MUTED)
        card.blit(sub, (card_rect.width // 2 - sub.get_width() // 2, 112))

        # Text input box
        box_rect = pygame.Rect(card_rect.width // 2 - 190, 160, 380, 52)
        pygame.draw.rect(card, (15, 23, 42), box_rect, border_radius=12)
        focus_col = ACCENT if (time.time() % 1.0 < 0.8) else SURFACE_BORDER
        pygame.draw.rect(card, focus_col, box_rect, width=2, border_radius=12)

        cursor_str = "|" if (time.time() % 1.0 < 0.5) else ""
        display_name = (name + cursor_str) if name else ""
        if name:
            name_surf = FONT_NAME.render(display_name, True, TEXT_COLOR)
            card.blit(name_surf, (box_rect.x + 18, box_rect.y + 11))
        else:
            placeholder = FONT_NAME.render("Type your name..." + cursor_str, True, (100, 116, 139))
            card.blit(placeholder, (box_rect.x + 18, box_rect.y + 11))

        # CTA Button
        btn_local = pygame.Rect(card_rect.width // 2 - 95, 255, 190, 46)
        pygame.draw.rect(card, (*ACCENT, 245 if btn_hover else 215), btn_local, border_radius=12)
        btn_txt = FONT_BTN.render("START GAME", True, (15, 23, 42))
        card.blit(btn_txt, (btn_local.centerx - btn_txt.get_width() // 2, btn_local.centery - btn_txt.get_height() // 2))

        hint = FONT_SMALL.render("Press ENTER or click to continue", True, TEXT_MUTED)
        card.blit(hint, (card_rect.width // 2 - hint.get_width() // 2, 325))

        screen.blit(card, card_rect.topleft)
        pygame.display.flip()

# ============================================================
#  GAME STATE INITIALIZATION
# ============================================================
player_name = run_name_entry()
pygame.event.clear()

game_state = "menu"          # "menu" or "playing"
menu_overlay = None          # None, "leaderboard", or "themes"
current_level_idx = 0
tubes_data = []
tube_rects = []
cur_ball_radius = 28
cur_tube_height = 280

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
last_pinch_hx, last_pinch_hy = WIDTH // 2, HEIGHT // 2
last_pinch_tube = None

FILTER_BETA_IDLE = 0.018
FILTER_BETA_HOLD = 0.06
cursor_filter_x = OneEuroFilter(min_cutoff=1.0, beta=FILTER_BETA_IDLE)
cursor_filter_y = OneEuroFilter(min_cutoff=1.0, beta=FILTER_BETA_IDLE)

drop_animations = {}
moves = 0
game_won = False
all_levels_complete = False
show_camera_preview = True

level_intro_start = 0
level_intro_active = False
victory_seq_active = False
victory_seq_start = 0
victory_confetti_spawned = False
sparked_tubes = set()
victory_sparks = []
confetti = []

def spawn_tube_sparks(cx, cy, count=16):
    palette = [ACCENT, (250, 204, 21), (52, 211, 153), (255, 255, 255), ACCENT_SECONDARY]
    for _ in range(count):
        angle = math.radians(random.uniform(-135, -45))
        speed = random.uniform(90, 240)
        victory_sparks.append({
            "x": cx + random.uniform(-10, 10),
            "y": cy + random.uniform(-4, 4),
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "size": random.uniform(2.5, 4.5),
            "color": random.choice(palette),
            "life": 1.0,
            "decay": random.uniform(1.3, 2.2),
        })

def update_and_draw_sparks(surf, dt):
    survivors = []
    for p in victory_sparks:
        p["life"] -= p["decay"] * dt
        if p["life"] > 0:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vy"] += 280 * dt
            p["vx"] *= 0.97
            alpha = max(0, min(255, int(255 * p["life"])))
            r, g, b = p["color"][:3]
            s = p["size"] * (0.35 + 0.65 * p["life"])
            spark_surf = pygame.Surface((int(s * 2 + 2), int(s * 2 + 2)), pygame.SRCALPHA)
            pygame.draw.circle(spark_surf, (r, g, b, alpha), (int(s + 1), int(s + 1)), int(s))
            surf.blit(spark_surf, (p["x"] - s - 1, p["y"] - s - 1))
            survivors.append(p)
    victory_sparks[:] = survivors

def spawn_confetti_cannons():
    confetti.clear()
    palette = COLOR_LIST if COLOR_LIST else [ACCENT, ACCENT_SECONDARY, (250, 204, 21), (52, 211, 153), (244, 63, 94)]
    # Left cannon burst shooting up and toward center
    for _ in range(70):
        angle = math.radians(random.uniform(-78, -25))
        speed = random.uniform(420, 850)
        confetti.append({
            "x": random.uniform(10, 60),
            "y": HEIGHT - random.uniform(10, 40),
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "size": random.uniform(5, 9),
            "color": random.choice(palette),
            "spin": random.uniform(-10, 10),
            "angle": random.uniform(0, 360),
            "drag": random.uniform(0.97, 0.985),
            "is_cannon": True,
        })
    # Right cannon burst shooting up and toward center
    for _ in range(70):
        angle = math.radians(random.uniform(-155, -102))
        speed = random.uniform(420, 850)
        confetti.append({
            "x": WIDTH - random.uniform(10, 60),
            "y": HEIGHT - random.uniform(10, 40),
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "size": random.uniform(5, 9),
            "color": random.choice(palette),
            "spin": random.uniform(-10, 10),
            "angle": random.uniform(0, 360),
            "drag": random.uniform(0.97, 0.985),
            "is_cannon": True,
        })

def spawn_confetti():
    spawn_confetti_cannons()

def update_and_draw_confetti(surf, dt):
    for p in confetti:
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt
        if p.get("is_cannon", False):
            p["vx"] *= p.get("drag", 0.98)
            p["vy"] += 440 * dt
        else:
            p["vy"] += 260 * dt
        p["angle"] += p["spin"]
        if p["y"] > HEIGHT + 20:
            p["y"] = random.uniform(-40, -10)
            p["x"] = random.uniform(0, WIDTH)
            p["vy"] = random.uniform(130, 280)
            p["vx"] = random.uniform(-45, 45)
            p["is_cannon"] = False
        s = p["size"]
        rect_surf = pygame.Surface((int(s * 2), int(s)), pygame.SRCALPHA)
        rect_surf.fill(p["color"])
        rotated = pygame.transform.rotate(rect_surf, p["angle"])
        surf.blit(rotated, (p["x"], p["y"]))

def load_level(idx):
    global current_level_idx, tubes_data, tube_rects, cur_ball_radius, cur_tube_height
    global selected_ball_color, source_tube_idx, drop_animations, moves, game_won, all_levels_complete
    global level_intro_start, level_intro_active
    global victory_seq_active, victory_seq_start, victory_confetti_spawned, sparked_tubes
    current_level_idx = idx
    cfg = LEVELS[idx]
    _, _, cur_ball_radius, cur_tube_height, tube_rects = compute_layout(cfg["tubes"])
    tubes_data = generate_puzzle(cfg)
    selected_ball_color = None
    source_tube_idx = None
    drop_animations = {}
    shaking_tubes.clear()
    floating_toasts.clear()
    moves = 0
    game_won = False
    all_levels_complete = False
    victory_seq_active = False
    victory_seq_start = 0
    victory_confetti_spawned = False
    sparked_tubes.clear()
    victory_sparks.clear()
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

# ============================================================
#  MAIN RENDER & INTERACTION LOOP
# ============================================================
running = True
while running:
    clock.tick(60)
    dt = min(0.05, max(0.001, clock.get_time() / 1000.0))

    mouse_triggered_click = False
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_down = pygame.mouse.get_pressed()[0]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_triggered_click = True
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
                elif event.key == pygame.K_n and game_won:
                    if current_level_idx + 1 < len(LEVELS):
                        load_level(current_level_idx + 1)
                        game_state = "playing"
                    else:
                        all_levels_complete = True

    # ---------------- Camera & MediaPipe Landmark Detection ----------------
    frame = camera.read()
    rgb_thumb = None

    if frame is not None:
        frame = cv2.flip(frame, 1)
        # Use INTER_AREA for crystal-clear thumbnail downscaling
        thumb_bgr = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb_thumb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)

        with _result_lock:
            busy = _detection_busy
        if not busy:
            det_bgr = cv2.resize(frame, (DET_WIDTH, DET_HEIGHT), interpolation=cv2.INTER_LINEAR)
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

        ix = remap_cursor_norm(index_tip.x) * WIDTH
        iy = remap_cursor_norm(index_tip.y) * HEIGHT
        tx = remap_cursor_norm(thumb_tip.x) * WIDTH
        ty = remap_cursor_norm(thumb_tip.y) * HEIGHT

        f_dx = (index_tip.x - thumb_tip.x) * CAM_WIDTH
        f_dy = (index_tip.y - thumb_tip.y) * CAM_HEIGHT
        finger_dist = math.hypot(f_dx, f_dy)

        p_dx = (middle_mcp.x - wrist.x) * CAM_WIDTH
        p_dy = (middle_mcp.y - wrist.y) * CAM_HEIGHT
        palm_size = max(25.0, math.hypot(p_dx, p_dy))

        current_pinch_ratio = finger_dist / palm_size

        if not is_pinching:
            pinch_detected = (current_pinch_ratio < PINCH_START_RATIO)
        else:
            pinch_detected = (current_pinch_ratio < PINCH_END_RATIO)

        cursor_found = True
        if pinch_detected:
            hand_x, hand_y = (ix + tx) / 2.0, (iy + ty) / 2.0
        else:
            hand_x, hand_y = ix, iy
        thumb_landmark_norm = (index_tip.x, index_tip.y)
    else:
        if selected_ball_color is not None and lost_hand_grace < LOST_HAND_MAX_GRACE:
            lost_hand_grace += 1
            cursor_found = True
            pinch_detected = True
        else:
            cursor_found = True
            hand_x, hand_y = mouse_x, mouse_y
            pinch_detected = mouse_down

    if cursor_found:
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
    pinch_rising_edge = (cursor_found and prev_cursor_found and pinch_detected and not is_pinching) or mouse_triggered_click

    # ================================================================
    #  MENU STATE LOGIC & RENDERING
    # ================================================================
    if game_state == "menu":
        hovered_idx = None
        hovered_action = None
        hovered_theme_key = None
        hovered_modal_close = False

        if menu_overlay == "leaderboard":
            close_btn = pygame.Rect(WIDTH // 2 - 70, 546, 140, 38)
            if cursor_found and close_btn.collidepoint(hx, hy):
                hovered_modal_close = True
            if pinch_rising_edge:
                if hovered_modal_close or not pygame.Rect(WIDTH // 2 - 290, 110, 580, 490).collidepoint(hx, hy):
                    menu_overlay = None
        elif menu_overlay == "themes":
            for t_key, t_rect in THEME_CARD_RECTS.items():
                if cursor_found and t_rect.collidepoint(hx, hy):
                    hovered_theme_key = t_key
                    break
            close_btn = pygame.Rect(WIDTH // 2 - 70, 546, 140, 38)
            if cursor_found and close_btn.collidepoint(hx, hy):
                hovered_modal_close = True
            if pinch_rising_edge:
                if hovered_theme_key:
                    apply_theme(hovered_theme_key)
                elif hovered_modal_close or not pygame.Rect(WIDTH // 2 - 360, 110, 720, 490).collidepoint(hx, hy):
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

        draw_menu(screen, hovered_idx, hovered_action, (hand_landmarks_list is not None), player_name,
                  menu_overlay=menu_overlay, hovered_theme_key=hovered_theme_key,
                  hovered_modal_close=hovered_modal_close,
                  cursor_pos=(hx, hy) if cursor_found else None,
                  dt=dt)

        # Draw Camera Preview PIP in top-right
        if show_camera_preview:
            cursor_near = (hx >= WIDTH - THUMB_WIDTH - 50 and hy <= 18 + THUMB_HEIGHT + 35)
            draw_thumbnail(screen, rgb_thumb, thumb_landmark_norm, cursor_near=cursor_near, hand_detected=(hand_landmarks_list is not None))

        if cursor_found:
            ring_color = ACCENT if is_pinching else (255, 255, 255)
            pygame.draw.circle(screen, ring_color, (hx, hy), 9, width=2)
            pygame.draw.circle(screen, ring_color, (hx, hy), 3)

        pygame.display.flip()
        continue

    # ================================================================
    #  PLAYING STATE LOGIC & RENDERING
    # ================================================================
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
                        last_pinch_tube = hovered_tube
                is_pinching = True
                last_pinch_hx, last_pinch_hy = hx, hy
                if hovered_tube is not None:
                    last_pinch_tube = hovered_tube
            else:
                if is_pinching and selected_ball_color is not None:
                    drop_hx, drop_hy = last_pinch_hx, last_pinch_hy
                    drop_tube = hovered_tube if hovered_tube is not None else last_pinch_tube
                    if drop_tube is None:
                        drop_tube = find_nearest_tube(drop_hx, drop_hy, tube_rects)
                    if drop_tube is None:
                        drop_tube = find_nearest_tube(hx, hy, tube_rects)

                    val_status, val_msg = get_drop_validation(drop_tube, tubes_data, selected_ball_color, source_tube_idx)

                    if val_status == 'valid':
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
                            "duration_ms": DROP_ANIM_MS,
                            "is_return": False,
                        }
                        moves += 1
                    elif val_status == 'same_source':
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
                            "duration_ms": 220,
                            "is_return": False,
                        }
                    else:
                        if drop_tube is not None:
                            trigger_tube_shake(drop_tube)
                            if val_status == 'full':
                                add_toast("Tube Full (Max 4)", drop_hx, drop_hy - 15, color=(244, 63, 94))
                            elif val_status == 'mismatch':
                                add_toast("Color Mismatch", drop_hx, drop_hy - 15, color=(245, 158, 11))
                            else:
                                add_toast("Invalid Move", drop_hx, drop_hy - 15, color=(244, 63, 94))
                        else:
                            add_toast("Drop inside a tube", drop_hx, drop_hy - 15, color=(245, 158, 11))

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
                            "duration_ms": RETURN_ANIM_MS,
                            "is_return": True,
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
                    "duration_ms": RETURN_ANIM_MS,
                    "is_return": True,
                }
                add_toast("Hand lost - ball returned", hx, hy - 15, color=(148, 163, 184))
                selected_ball_color = None
                source_tube_idx = None
                last_pinch_tube = None
            is_pinching = False
    else:
        if selected_ball_color is not None:
            tubes_data[source_tube_idx].append(selected_ball_color)
            selected_ball_color = None
            source_tube_idx = None
        is_pinching = cursor_found and pinch_detected

    if not game_won and check_victory(tubes_data):
        game_won = True
        victory_seq_active = True
        victory_seq_start = pygame.time.get_ticks()
        victory_confetti_spawned = False
        sparked_tubes.clear()
        victory_sparks.clear()
        submit_score_async(f"{player_name} (Lv{current_level_idx + 1})", moves)

    # ---------------- HUD & Header Rendering ----------------
    ambient_cosmic_field.update_and_draw(screen, dt, BG_SURFACE, cursor_pos=None, interactive=False)

    lvl_cfg = LEVELS[current_level_idx]

    # Top Left Level Badge & Title
    badge_lvl = pygame.Surface((110, 26), pygame.SRCALPHA)
    pygame.draw.rect(badge_lvl, (*ACCENT, 35), badge_lvl.get_rect(), border_radius=13)
    pygame.draw.rect(badge_lvl, (*ACCENT, 160), badge_lvl.get_rect(), width=1, border_radius=13)
    lvl_num_txt = FONT_BADGE.render(f"LEVEL {current_level_idx + 1:02d}", True, ACCENT)
    badge_lvl.blit(lvl_num_txt, ((110 - lvl_num_txt.get_width()) // 2, 5))
    screen.blit(badge_lvl, (40, 26))

    title_txt = FONT_TITLE.render(f"Level {current_level_idx + 1}  ·  {lvl_cfg['tubes']} Tubes", True, TEXT_COLOR)
    screen.blit(title_txt, (40, 58))

    sub_hint = FONT_SUB.render("Pinch thumb + index to grab · Hover over tube · Release to drop", True, TEXT_MUTED)
    screen.blit(sub_hint, (40, 90))

    # Moves Badge
    moves_w = 125
    moves_badge = pygame.Surface((moves_w, 34), pygame.SRCALPHA)
    pygame.draw.rect(moves_badge, (*SURFACE, 230), moves_badge.get_rect(), border_radius=10)
    pygame.draw.rect(moves_badge, SURFACE_BORDER, moves_badge.get_rect(), width=1, border_radius=10)
    pygame.draw.circle(moves_badge, ACCENT, (16, 17), 4)
    m_txt = FONT_BTN_SUB.render(f"{moves} MOVES", True, TEXT_COLOR)
    moves_badge.blit(m_txt, (30, (34 - m_txt.get_height()) // 2))
    screen.blit(moves_badge, (40, 122))

    # Quick Action Buttons in Top Bar
    btn_reset = pygame.Rect(410, 26, 95, 34)
    btn_menu = pygame.Rect(515, 26, 90, 34)
    btn_cam = pygame.Rect(615, 26, 90, 34)

    for b_rect, b_lbl, b_key in [(btn_reset, "RESET", "R"), (btn_menu, "MENU", "M"), (btn_cam, "CAM", "C")]:
        is_hov = b_rect.collidepoint(hx, hy)
        b_surf = pygame.Surface((b_rect.width, b_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(b_surf, (*ACCENT, 60) if is_hov else (*SURFACE, 210), b_surf.get_rect(), border_radius=8)
        pygame.draw.rect(b_surf, ACCENT if is_hov else SURFACE_BORDER, b_surf.get_rect(), width=1, border_radius=8)
        b_t = FONT_BADGE.render(f"[{b_key}] {b_lbl}", True, ACCENT if is_hov else TEXT_MUTED)
        b_surf.blit(b_t, ((b_rect.width - b_t.get_width()) // 2, (b_rect.height - b_t.get_height()) // 2))
        screen.blit(b_surf, b_rect.topleft)

        if pinch_rising_edge and is_hov:
            if b_key == "R":
                load_level(current_level_idx)
            elif b_key == "M":
                game_state = "menu"
                is_pinching = False
                prev_cursor_found = False
            elif b_key == "C":
                show_camera_preview = not show_camera_preview

    footer_hint = FONT_SMALL.render("R  Reset Level    ·    M  Level Menu    ·    C  Toggle Camera    ·    ESC  Quit Game", True, (100, 116, 139))
    screen.blit(footer_hint, (40, HEIGHT - 32))

    now_ms = pygame.time.get_ticks()
    still_animating_intro = False

    # Draw Tubes
    for idx, base_rect in enumerate(tube_rects):
        shake_x = 0
        if idx in shaking_tubes:
            s_info = shaking_tubes[idx]
            s_elapsed = now_ms - s_info["t0"]
            if s_elapsed < s_info["duration_ms"]:
                s_prog = s_elapsed / s_info["duration_ms"]
                shake_x = int(math.sin(s_elapsed * 0.06) * 6.0 * (1.0 - s_prog))
            else:
                del shaking_tubes[idx]

        bounce_y = 0
        is_celebrating_tube = False
        if victory_seq_active:
            v_elapsed = now_ms - victory_seq_start
            tube_delay = idx * 60  # Staggered wave ripple
            t_wave = v_elapsed - tube_delay
            if 0 <= t_wave <= 480:
                p = t_wave / 480.0
                bounce_y = int(-18.0 * math.sin(p * math.pi))
                if 0.25 <= p <= 0.45 and idx not in sparked_tubes:
                    sparked_tubes.add(idx)
                    spawn_tube_sparks(base_rect.centerx, base_rect.top)
            if t_wave > 0:
                is_celebrating_tube = True

        rect = base_rect.move(shake_x, bounce_y)
        pygame.draw.rect(screen, SHADOW_COLOR, rect, border_bottom_left_radius=18, border_bottom_right_radius=18)

        tube_status = None
        badge_info = None

        if is_celebrating_tube and tubes_data[idx]:
            tube_status = 'valid'
            badge_info = ("CLEARED", (16, 185, 129), (52, 211, 153), "check")
        elif selected_ball_color is not None and hovered_tube == idx:
            val_status, val_msg = get_drop_validation(idx, tubes_data, selected_ball_color, source_tube_idx)
            if val_status == 'valid':
                tube_status = 'valid'
                badge_info = ("Drop Here", (16, 185, 129), (52, 211, 153), "check")
            elif val_status == 'same_source':
                tube_status = 'active'
                badge_info = ("Return", (30, 41, 59), (148, 163, 184), "return")
            elif val_status == 'full':
                tube_status = 'invalid'
                badge_info = ("Tube Full", (225, 29, 72), (251, 113, 133), "cross")
            elif val_status == 'mismatch':
                tube_status = 'invalid'
                badge_info = ("Mismatch", (217, 119, 6), (251, 191, 36), "cross")

        anim = drop_animations.get(idx)
        skip_slot = anim["slot"] if anim else -1

        for b_idx, b_color in enumerate(tubes_data[idx]):
            if b_idx == skip_slot:
                continue
            bx, by = slot_position(rect, b_idx, cur_ball_radius)

            if level_intro_active:
                delay = idx * 30 + b_idx * 40
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

        # Translucent Ghost Ball Drop Preview
        if tube_status == 'valid' and selected_ball_color is not None:
            ghost_slot = len(tubes_data[idx])
            if ghost_slot < TUBE_CAPACITY:
                gx, gy = slot_position(rect, ghost_slot, cur_ball_radius)
                pulse = int(120 + 60 * math.sin(now_ms / 140.0))
                draw_ball(screen, selected_ball_color, gx, gy, cur_ball_radius, alpha=pulse)
                pygame.draw.circle(screen, (16, 185, 129), (int(gx), int(gy)), cur_ball_radius + 2, width=1)

        draw_tube(screen, rect, tube_status, ball_radius=cur_ball_radius)

        if badge_info is not None:
            draw_tube_badge(screen, rect, badge_info[0], badge_info[1], badge_info[2], badge_type=badge_info[3])

    if level_intro_active and not still_animating_intro:
        level_intro_active = False

    # Drop animation physics
    finished = []
    for idx, anim in drop_animations.items():
        elapsed = now_ms - anim["t0"]
        dur = anim.get("duration_ms", DROP_ANIM_MS)
        t = min(1.0, elapsed / dur)
        e = ease_out_cubic(t)
        cx = anim["start_x"] + (anim["end_x"] - anim["start_x"]) * e
        cy = anim["start_y"] + (anim["end_y"] - anim["start_y"]) * e
        if anim.get("is_return"):
            cy -= int(40.0 * math.sin(t * math.pi))
        draw_ball(screen, anim["color"], cx, cy, cur_ball_radius)
        if elapsed >= dur:
            finished.append(idx)
    for idx in finished:
        del drop_animations[idx]

    update_and_draw_toasts(screen)

    # Top-Right PIP Webcam Frame
    if show_camera_preview:
        cursor_near = (hx >= WIDTH - THUMB_WIDTH - 50 and hy <= 18 + THUMB_HEIGHT + 35)
        draw_thumbnail(screen, rgb_thumb, thumb_landmark_norm, cursor_near=cursor_near, hand_detected=(hand_landmarks_list is not None))

    # Ball carried at cursor
    if selected_ball_color is not None:
        draw_ball(screen, selected_ball_color, hx, hy, cur_ball_radius, alpha=235)

    # Targeting Cursor
    if cursor_found:
        ring_color = ACCENT if is_pinching else (255, 255, 255)
        pygame.draw.circle(screen, ring_color, (hx, hy), 9, width=2)
        pygame.draw.circle(screen, ring_color, (hx, hy), 3)

    # Update & Render Victory Sequence Sparks & Confetti
    if victory_sparks:
        update_and_draw_sparks(screen, dt)
    if confetti:
        update_and_draw_confetti(screen, dt)

    # Celebratory Banner during Phase 1 & 2 (before modal arrives)
    if victory_seq_active:
        v_elapsed = now_ms - victory_seq_start
        # Launch dual celebratory cannons at 800ms
        if not victory_confetti_spawned and v_elapsed >= 800:
            spawn_confetti_cannons()
            victory_confetti_spawned = True

        # Celebratory header banner before modal takes over
        if v_elapsed < 1400:
            b_alpha = min(255, int(255 * math.sin(min(1.0, v_elapsed / 1200.0) * math.pi)))
            if b_alpha > 5:
                b_w, b_h = 360, 48
                banner_surf = pygame.Surface((b_w, b_h), pygame.SRCALPHA)
                pygame.draw.rect(banner_surf, (*SURFACE, int(240 * (b_alpha / 255.0))), banner_surf.get_rect(), border_radius=24)
                pygame.draw.rect(banner_surf, (*WIN_COLOR, b_alpha), banner_surf.get_rect(), width=2, border_radius=24)
                draw_vector_trophy(banner_surf, 32, b_h // 2, size=24, color=WIN_COLOR)
                v_txt = FONT_BTN.render("PUZZLE SOLVED!", True, WIN_COLOR)
                banner_surf.blit(v_txt, (58, (b_h - v_txt.get_height()) // 2))
                screen.blit(banner_surf, ((WIDTH - b_w) // 2, 70))

    # ================================================================
    #  VICTORY OVERLAY (Phase 3 of Closing Sequence: Ease-Out-Back Modal)
    # ================================================================
    if game_won:
        v_elapsed = now_ms - victory_seq_start
        MODAL_DELAY = 1050
        if v_elapsed >= MODAL_DELAY:
            modal_time = (v_elapsed - MODAL_DELAY) / 480.0
            modal_prog = min(1.0, max(0.0, modal_time))
            alpha_fac = ease_out_cubic(modal_prog)
            slide_fac = ease_out_back(modal_prog)

            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((6, 9, 16, int(215 * alpha_fac)))
            screen.blit(overlay, (0, 0))

            # Celebratory Card with smooth slide-up
            base_card_y = HEIGHT // 2 - 220
            card_slide_offset = int((1.0 - slide_fac) * 75)
            win_card_rect = pygame.Rect(WIDTH // 2 - 280, base_card_y + card_slide_offset, 560, 440)
            win_card = pygame.Surface((win_card_rect.width, win_card_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(win_card, (*SURFACE, int(250 * alpha_fac)), win_card.get_rect(), border_radius=20)
            pygame.draw.rect(win_card, (*WIN_COLOR, int(180 * alpha_fac)), win_card.get_rect(), width=2, border_radius=20)

            draw_vector_trophy(win_card, win_card_rect.width // 2, 44, size=34, color=WIN_COLOR)
            h_text = "ALL LEVELS COMPLETE!" if all_levels_complete else "PUZZLE SOLVED!"
            win_title = FONT_BIG.render(h_text, True, WIN_COLOR)
            win_card.blit(win_title, (win_card_rect.width // 2 - win_title.get_width() // 2, 82))

            stat_y = 140
            stat_w = (win_card_rect.width - 64) // 3
            for s_i, (s_lbl, s_val) in enumerate([
                ("LEVEL", f"{current_level_idx + 1}"),
                ("MOVES", f"{moves}"),
                ("STATUS", "RECORDED" if score_submit_status["ok"] else "SUBMITTED")
            ]):
                s_box = pygame.Rect(24 + s_i * (stat_w + 8), stat_y, stat_w, 58)
                pygame.draw.rect(win_card, (15, 23, 42, 180), s_box, border_radius=10)
                pygame.draw.rect(win_card, SURFACE_BORDER, s_box, width=1, border_radius=10)
                sl = FONT_BADGE.render(s_lbl, True, TEXT_MUTED)
                sv = FONT_BTN.render(s_val, True, ACCENT if s_i == 1 else TEXT_COLOR)
                win_card.blit(sl, (s_box.centerx - sl.get_width() // 2, s_box.y + 8))
                win_card.blit(sv, (s_box.centerx - sv.get_width() // 2, s_box.y + 27))

            with net_lock:
                loaded, entries = leaderboard_data["loaded"], list(leaderboard_data["entries"])

            lb_preview_y = 216
            pygame.draw.rect(win_card, (15, 23, 42, 160), (24, lb_preview_y, win_card_rect.width - 48, 110), border_radius=10)
            lbl_top = FONT_BADGE.render("CURRENT TOP SCORES", True, TEXT_MUTED)
            win_card.blit(lbl_top, (38, lb_preview_y + 8))

            if loaded and entries:
                for e_i, entry in enumerate(entries[:3]):
                    e_y = lb_preview_y + 32 + e_i * 25
                    draw_vector_medal(win_card, 48, e_y + 8, rank=e_i+1, size=18)
                    e_name = FONT_BTN_SUB.render(str(entry.get("name", "Player"))[:18], True, TEXT_COLOR)
                    e_moves = FONT_BTN_SUB.render(f"{entry.get('moves', '?')} moves", True, ACCENT)
                    win_card.blit(e_name, (68, e_y + 2))
                    win_card.blit(e_moves, (win_card_rect.width - 48 - e_moves.get_width(), e_y + 2))
            else:
                lb_wait = FONT_BTN_SUB.render("Leaderboard updated on server", True, TEXT_MUTED)
                win_card.blit(lb_wait, (win_card_rect.width // 2 - lb_wait.get_width() // 2, lb_preview_y + 48))

            # Action Buttons Row
            cta_y = 350
            btn_w_action = 155
            btn_next = pygame.Rect(WIDTH // 2 - 250, win_card_rect.y + cta_y, btn_w_action, 46)
            btn_replay = pygame.Rect(WIDTH // 2 - 80, win_card_rect.y + cta_y, btn_w_action, 46)
            btn_back_menu = pygame.Rect(WIDTH // 2 + 90, win_card_rect.y + cta_y, btn_w_action, 46)

            screen.blit(win_card, win_card_rect.topleft)

            for act_rect, act_text, act_cmd, is_primary in [
                (btn_next, "NEXT LEVEL [N]", "next", True),
                (btn_replay, "REPLAY [R]", "replay", False),
                (btn_back_menu, "MENU [M]", "menu", False)
            ]:
                act_hov = (modal_prog >= 0.75 and act_rect.collidepoint(hx, hy))
                b_s = pygame.Surface((act_rect.width, act_rect.height), pygame.SRCALPHA)
                bg_col = (*WIN_COLOR, 235 if act_hov else 195) if is_primary else ((*SURFACE, 235 if act_hov else 185))
                border_col = WIN_COLOR if is_primary else (ACCENT if act_hov else SURFACE_BORDER)
                pygame.draw.rect(b_s, bg_col, b_s.get_rect(), border_radius=10)
                pygame.draw.rect(b_s, border_col, b_s.get_rect(), width=2 if act_hov else 1, border_radius=10)
                t_col = (15, 23, 42) if is_primary else TEXT_COLOR
                t_s = FONT_BADGE.render(act_text, True, t_col)
                b_s.blit(t_s, ((act_rect.width - t_s.get_width()) // 2, (act_rect.height - t_s.get_height()) // 2))
                screen.blit(b_s, act_rect.topleft)

                if modal_prog >= 0.75 and pinch_rising_edge and act_hov:
                    if act_cmd == "next":
                        if current_level_idx + 1 < len(LEVELS):
                            load_level(current_level_idx + 1)
                            game_state = "playing"
                        else:
                            all_levels_complete = True
                    elif act_cmd == "replay":
                        load_level(current_level_idx)
                        game_state = "playing"
                    elif act_cmd == "menu":
                        game_state = "menu"
                        is_pinching = False
                        prev_cursor_found = False


    prev_cursor_found = cursor_found
    pygame.display.flip()

camera.release()
detector.close()
pygame.quit()
sys.exit()
