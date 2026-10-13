#working
# -*- coding: utf-8 -*-
"""
CRICKET OCR - FINAL INNINGS-AWARE COLAB VERSION

Pipeline:
Video -> EasyOCR (full frame + scoreboard ROI)
-> score candidate extraction
-> scoreboard temporal tracking
-> automatic score-order detection
-> normalized runs/wickets
-> innings detection/reset
-> wicket-event detection
-> CSV + wicket clips + ZIP

Key fixes:
1. Handles multiple innings in one highlight video.
2. A new innings resets the wicket tracker instead of treating 2->0 wickets
   as an OCR error.
3. A wicket is detected when wickets increase by exactly 1 even if the
   displayed run total also changes.
4. Low-confidence split-digit OCR is heavily down-weighted/rejected.
5. Wicket timestamp is the FIRST reliable observation of the score change.
6. The score format is locked once, then reused across innings.
7. Highlight videos can skip intermediate scoreboard states.
"""

import sys, subprocess, importlib.util, os, re, csv, shutil, zipfile, math
from pathlib import Path
from collections import deque, Counter

# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================
REQUIRED = {
    "cv2": "opencv-python-headless",
    "easyocr": "easyocr",
    "numpy": "numpy",
}

for module, package in REQUIRED.items():
    if importlib.util.find_spec(module) is None:
        print(f"Installing {package} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import cv2
import easyocr
import numpy as np

try:
    import torch
    USE_GPU = bool(torch.cuda.is_available())
except Exception:
    USE_GPU = False

try:
    from google.colab import files
except ImportError:
    raise RuntimeError("Run this notebook in Google Colab.")

# ============================================================
# 2. UPLOAD ONE VIDEO
# ============================================================
PROJECT_DIR = Path("/content/cricket_ocr")
VIDEO_DIR = PROJECT_DIR / "data" / "videos"
RESULTS_DIR = PROJECT_DIR / "results"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

print("=" * 76)
print("CRICKET OCR - FINAL INNINGS-AWARE")
print("=" * 76)
print("Upload ONE cricket video.")
uploaded = files.upload()

video_names = [n for n in uploaded if Path(n).suffix.lower() in VIDEO_EXTENSIONS]
if not video_names:
    raise RuntimeError("No supported video uploaded.")

source = Path(video_names[0])
VIDEO_PATH = VIDEO_DIR / source.name
if VIDEO_PATH.exists():
    VIDEO_PATH.unlink()
shutil.move(str(source), str(VIDEO_PATH))

OUTPUT_DIR = RESULTS_DIR / VIDEO_PATH.stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR = OUTPUT_DIR / "wicket_segments"
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Video  : {VIDEO_PATH}")
print(f"Output : {OUTPUT_DIR}")

# ============================================================
# 3. SETTINGS
# ============================================================
OCR_INTERVAL = 0.50

MAX_RUNS = 500
MAX_WICKETS = 10
MAX_WICKET_JUMP = 1

SCORE_WINDOW = 7
MIN_SCORE_AGREE = 3

FORMAT_MIN_OBSERVATIONS = 4
FORMAT_MIN_DIFF = 10.0

WICKET_CLIP_BEFORE = 6.0
WICKET_CLIP_AFTER = 6.0

MAX_SCOREBOARD_JUMP_RUNS = 20
TRACK_MAX_MISSES = 8

# Candidate quality
MIN_EVENT_CONFIDENCE = 0.50
MIN_SCORE_CONFIDENCE = 0.20
LOW_CONF_SPLIT_REJECT = 0.45

# Innings reset logic
RESET_CONFIRMATIONS = 6
RESET_CONFIRM_WINDOW = 6.0

# ============================================================
# 4. VIDEO INFO
# ============================================================
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frames_total / fps if fps > 0 else 0.0
sample_every = max(1, int(round(fps * OCR_INTERVAL)))

print("=" * 76)
print(f"FPS          : {fps:.2f}")
print(f"Duration     : {duration:.2f}s")
print(f"OCR interval : {OCR_INTERVAL:.2f}s")
print(f"Sample every : {sample_every} frames")
print("Score order  : AUTO (Runs-Wickets / Wickets-Runs)")
print("Innings mode : ENABLED")
print("=" * 76)

# ============================================================
# 5. EASY OCR
# ============================================================
print("Loading EasyOCR...", flush=True)
print(f"EasyOCR device: {'GPU' if USE_GPU else 'CPU'}", flush=True)
reader = easyocr.Reader(["en"], gpu=USE_GPU, verbose=False)
print("EasyOCR ready.", flush=True)

# ============================================================
# 6. SCORE EXTRACTION
# ============================================================
PAIR_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[-–—/]\s*(\d{1,2})(?!\d)")
DIGIT_RE = re.compile(r"^\d{1,3}$")

def valid_score(runs, wickets):
    return 0 <= int(runs) <= MAX_RUNS and 0 <= int(wickets) <= MAX_WICKETS

def box_info(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def make_candidate(first, second, text, conf, bbox, source_kind):
    first, second = int(first), int(second)
    x1, y1, x2, y2 = box_info(bbox)

    if float(conf) < MIN_SCORE_CONFIDENCE:
        return []

    # If one displayed value is > 10, it is overwhelmingly likely to be runs.
    if first > MAX_WICKETS and second <= MAX_WICKETS:
        options = [(first, second)]
    elif second > MAX_WICKETS and first <= MAX_WICKETS:
        options = [(second, first)]
    else:
        options = [(first, second), (second, first)]

    out = []
    for runs, wickets in options:
        if not valid_score(runs, wickets):
            continue
        out.append({
            "display_first": first,
            "display_second": second,
            "runs": int(runs),
            "wickets": int(wickets),
            "text": str(text).strip(),
            "confidence": float(conf),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "source_kind": source_kind,
            "selection_score": 0.0,
        })
    return out

def extract_score_candidates(results, frame_w, frame_h):
    candidates = []

    # A. Direct score pair
    for bbox, text, conf in results:
        text = str(text).strip()
        if not text:
            continue
        for a, b in PAIR_RE.findall(text):
            candidates.extend(
                make_candidate(a, b, text, conf, bbox, "full_pair")
            )

    # B. Split digits
    numeric = []
    for bbox, text, conf in results:
        t = str(text).strip().replace(" ", "")
        if DIGIT_RE.fullmatch(t):
            x1, y1, x2, y2 = box_info(bbox)
            numeric.append({
                "value": int(t), "conf": float(conf),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1+x2)/2.0, "cy": (y1+y2)/2.0,
            })

    for a in numeric:
        for b in numeric:
            if a is b:
                continue
            dx = b["cx"] - a["cx"]
            dy = abs(b["cy"] - a["cy"])
            if dx <= 0 or dx > frame_w * 0.14 or dy > frame_h * 0.06:
                continue
            bbox = [
                (a["x1"], a["y1"]), (a["x2"], a["y1"]),
                (b["x2"], b["y2"]), (a["x1"], a["y2"])
            ]
            conf = min(a["conf"], b["conf"])
            # Split OCR is useful, but low-confidence split pairs are
            # usually broadcast-text garbage.
            if conf < LOW_CONF_SPLIT_REJECT:
                continue
            candidates.extend(
                make_candidate(
                    a["value"], b["value"], f"{a['value']}-{b['value']}",
                    conf, bbox, "full_split"
                )
            )

    unique = {}
    for c in candidates:
        key = (
            c["display_first"], c["display_second"],
            c["x1"], c["y1"], c["x2"], c["y2"], c["text"]
        )
        if key not in unique or c["confidence"] > unique[key]["confidence"]:
            unique[key] = c
    return list(unique.values())

def extract_roi_candidates(roi_results, ox, oy):
    out = []

    for bbox, text, conf in roi_results:
        text = str(text).strip()
        for a, b in PAIR_RE.findall(text):
            shifted = [(p[0] + ox, p[1] + oy) for p in bbox]
            out.extend(
                make_candidate(a, b, text, conf, shifted, "scoreboard_roi_pair")
            )

    numeric = []
    for bbox, text, conf in roi_results:
        t = str(text).strip().replace(" ", "")
        if DIGIT_RE.fullmatch(t):
            shifted = [(p[0] + ox, p[1] + oy) for p in bbox]
            x1, y1, x2, y2 = box_info(shifted)
            numeric.append({
                "value": int(t), "conf": float(conf),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1+x2)/2.0, "cy": (y1+y2)/2.0,
            })

    for a in numeric:
        for b in numeric:
            if a is b:
                continue
            dx = b["cx"] - a["cx"]
            dy = abs(b["cy"] - a["cy"])
            if dx <= 0 or dx > max(250, width * 0.20) or dy > height * 0.07:
                continue
            conf = min(a["conf"], b["conf"])
            if conf < LOW_CONF_SPLIT_REJECT:
                continue
            bbox = [
                (a["x1"], a["y1"]), (a["x2"], a["y1"]),
                (b["x2"], b["y2"]), (a["x1"], a["y2"])
            ]
            out.extend(
                make_candidate(
                    a["value"], b["value"], f"{a['value']}-{b['value']}",
                    conf, bbox, "scoreboard_roi_split"
                )
            )
    return out

# ============================================================
# 7. SCOREBOARD TRACKER
# ============================================================
class ScoreboardTrack:
    def __init__(self, fw, fh):
        self.fw = fw
        self.fh = fh
        self.last = None
        self.misses = 0

    def location_score(self, c):
        x = c["cx"] / max(1.0, self.fw)
        y = c["cy"] / max(1.0, self.fh)

        score = 0.0
        if x <= 0.45:
            score += 3.5
        elif x <= 0.60:
            score += 1.0

        if y >= 0.68:
            score += 3.5
        elif y >= 0.55:
            score += 1.0

        if c["source_kind"].startswith("scoreboard_roi"):
            score += 7.0

        return score

    def continuity_score(self, c):
        if self.last is None:
            return 0.0

        dx = abs(c["cx"] - self.last["cx"]) / max(1.0, self.fw)
        dy = abs(c["cy"] - self.last["cy"]) / max(1.0, self.fh)
        score = max(-4.0, 5.0 - 20.0 * (dx + dy))

        if (
            c["display_first"] == self.last["display_first"]
            and c["display_second"] == self.last["display_second"]
        ):
            score += 4.0

        return score

    def rank(self, candidates):
        if not candidates:
            self.misses += 1
            if self.misses > TRACK_MAX_MISSES:
                self.last = None
            return None

        ranked = []
        for c in candidates:
            s = 3.0 * c["confidence"]
            s += self.location_score(c)
            s += self.continuity_score(c)

            if self.last is not None:
                da = abs(c["display_first"] - self.last["display_first"])
                db = abs(c["display_second"] - self.last["display_second"])
                if da > 2 and db > MAX_SCOREBOARD_JUMP_RUNS:
                    s -= 5.0

            # Strong preference for high-confidence pair OCR over weak split OCR.
            if c["source_kind"].endswith("_split"):
                s -= 1.5

            c["selection_score"] = round(s, 4)
            ranked.append(c)

        best = max(ranked, key=lambda z: z["selection_score"])
        self.last = dict(best)
        self.misses = 0
        return best

# ============================================================
# 8. AUTO FORMAT RESOLVER
# ============================================================
class FormatResolver:
    def __init__(self):
        self.format = None
        self.evidence = {"runs-wickets": 0.0, "wickets-runs": 0.0}
        self.last = {"runs-wickets": None, "wickets-runs": None}
        self.observations = []
        self.reason = ""

    def convert(self, a, b, fmt):
        return (a, b) if fmt == "runs-wickets" else (b, a)

    def score_transition(self, runs, wickets, prev):
        if not valid_score(runs, wickets):
            return -20.0
        if prev is None:
            return 0.0

        pr, pw = prev

        # A wicket count cannot go down inside the same innings.
        if wickets < pw:
            return -12.0
        if wickets - pw > MAX_WICKET_JUMP:
            return -12.0

        if wickets == pw:
            if runs < pr:
                return -10.0
            if runs == pr:
                return 1.5
            return 3.5

        # Exactly one wicket added; runs may also increase.
        if wickets == pw + 1:
            return 5.0 if runs >= pr else 1.0

        return -12.0

    def observe(self, a, b, confidence=1.0):
        a, b = int(a), int(b)
        self.observations.append((a, b, float(confidence)))

        if self.format is not None:
            return self.format

        for fmt in ("runs-wickets", "wickets-runs"):
            runs, wickets = self.convert(a, b, fmt)
            delta = self.score_transition(runs, wickets, self.last[fmt])
            if delta >= 0:
                delta *= max(0.6, min(1.0, float(confidence)))
            self.evidence[fmt] += delta
            if delta > -12:
                self.last[fmt] = (runs, wickets)

        # Hard evidence.
        if a > MAX_WICKETS and b <= MAX_WICKETS:
            self._lock("runs-wickets", "display first value exceeds wicket limit")
        elif b > MAX_WICKETS and a <= MAX_WICKETS:
            self._lock("wickets-runs", "display second value exceeds wicket limit")
        elif len(self.observations) >= FORMAT_MIN_OBSERVATIONS:
            diff = self.evidence["runs-wickets"] - self.evidence["wickets-runs"]
            if diff >= FORMAT_MIN_DIFF:
                self._lock("runs-wickets", "temporal sequence")
            elif diff <= -FORMAT_MIN_DIFF:
                self._lock("wickets-runs", "temporal sequence")

        return self.format

    def _lock(self, fmt, reason):
        if self.format is None:
            self.format = fmt
            self.reason = reason
            print(f"[FORMAT LOCKED] {fmt} ({reason})", flush=True)

    def normalize(self, a, b):
        if self.format == "runs-wickets":
            return int(a), int(b)
        if self.format == "wickets-runs":
            return int(b), int(a)
        return None, None

# ============================================================
# 9. INNINGS-AWARE WICKET TRACKER
# ============================================================
class InningsAwareTracker:
    """
    Strict confirmed-state tracker.

    IMPORTANT:
      * A single OCR frame can NEVER create a wicket.
      * A single OCR frame can NEVER create an innings reset.
      * Confirmed score is kept separate from pending OCR candidates.
      * Wicket requires repeated identical +1-wicket observations.
      * Innings reset requires repeated reset observations followed by
        a plausible continuation of the new innings.
    """

    def __init__(self):
        self.innings_id = 1
        self.current = None

        # Pending wicket transition
        self.pending_wicket = None
        self.pending_wicket_count = 0
        self.pending_wicket_first_time = None
        self.pending_wicket_first_frame = None

        # Pending innings reset
        self.pending_reset = None
        self.pending_reset_count = 0
        self.pending_reset_first_time = None

        self.last_time = None

    def _clear_wicket(self):
        self.pending_wicket = None
        self.pending_wicket_count = 0
        self.pending_wicket_first_time = None
        self.pending_wicket_first_frame = None

    def _clear_reset(self):
        self.pending_reset = None
        self.pending_reset_count = 0
        self.pending_reset_first_time = None

    def _candidate_is_consistent_with_reset(self, runs, wickets):
        if self.pending_reset is None:
            return False

        rr, rw = self.pending_reset

        # Same score is strongest evidence.
        if (runs, wickets) == (rr, rw):
            return True

        # Normal scoring after the proposed reset is also acceptable:
        # wickets can stay the same or increase by exactly one, and runs
        # may increase but not by an impossible OCR jump.
        if wickets == rw and rr <= runs <= rr + MAX_SCOREBOARD_JUMP_RUNS:
            return True

        if wickets == rw + 1 and rr <= runs <= rr + MAX_SCOREBOARD_JUMP_RUNS:
            return True

        return False

    def _start_or_update_reset(self, t, runs, wickets):
        if self.current is None:
            return False

        pr, pw = self.current

        # A possible innings reset must move backwards in BOTH dimensions.
        # This prevents ordinary OCR wicket noise from becoming an innings.
        if not (wickets < pw and runs < pr):
            return False

        candidate = (runs, wickets)

        if self.pending_reset is None:
            self.pending_reset = candidate
            self.pending_reset_count = 1
            self.pending_reset_first_time = t
            return False

        # If the candidate jumps back to the old innings, cancel the reset.
        # Example: false 0-2 followed immediately by 77-3.
        if not self._candidate_is_consistent_with_reset(runs, wickets):
            self._clear_reset()
            return False

        self.pending_reset_count += 1
        return False

    def _commit_reset_if_ready(self, t, runs, wickets):
        if self.pending_reset is None:
            return False

        if not self._candidate_is_consistent_with_reset(runs, wickets):
            self._clear_reset()
            return False

        # Deliberately strict: require SIX observations.
        # At 0.5 s OCR interval this is about 2.5 s of persistence.
        if self.pending_reset_count < RESET_CONFIRMATIONS:
            return False

        old_id = self.innings_id
        self.innings_id += 1

        # Use the first stable reset score, not a later arbitrary OCR value.
        self.current = self.pending_reset

        print(
            f"[INNINGS CHANGE] innings {old_id} -> {self.innings_id} "
            f"at {self.pending_reset_first_time:.2f}s | "
            f"reset score={self.current[0]}-{self.current[1]}",
            flush=True
        )

        self._clear_reset()
        self._clear_wicket()
        self.last_time = t
        return True

    def process(self, t, frame_no, runs, wickets, confidence=1.0):
        if runs is None or wickets is None:
            return None

        runs, wickets = int(runs), int(wickets)

        if not valid_score(runs, wickets):
            return None

        # --------------------------------------------------------
        # INITIAL SCORE
        # --------------------------------------------------------
        if self.current is None:
            self.current = (runs, wickets)
            self._clear_reset()
            self._clear_wicket()
            self.last_time = t

            print(
                f"[INIT] innings={self.innings_id} "
                f"score={runs}-{wickets}",
                flush=True
            )
            return None

        pr, pw = self.current

        # --------------------------------------------------------
        # POSSIBLE NEW INNINGS
        # --------------------------------------------------------
        if wickets < pw and runs < pr:
            self._start_or_update_reset(t, runs, wickets)

            if self._commit_reset_if_ready(t, runs, wickets):
                return None

            self.last_time = t
            return None

        # Any wicket decrease without strong two-dimensional reset
        # evidence is OCR noise.
        if wickets < pw:
            self._clear_reset()
            self.last_time = t
            return None

        # If a pending reset exists but the score has jumped back toward
        # the old innings, cancel it immediately.
        if self.pending_reset is not None:
            if not self._candidate_is_consistent_with_reset(runs, wickets):
                self._clear_reset()

        # --------------------------------------------------------
        # SAME CONFIRMED SCORE
        # --------------------------------------------------------
        if (runs, wickets) == (pr, pw):
            self.last_time = t
            return None

        # --------------------------------------------------------
        # CONFIRM A PENDING WICKET EVEN IF RUNS CHANGE AFTER IT
        # --------------------------------------------------------
        # Broadcasts often show, for example:
        #   44-0 -> 44-1 -> 45-1 -> 46-1
        # Once 44-1 is pending, 45-1 and 46-1 are still evidence for
        # the SAME wicket. Do not restart confirmation just because runs moved.
        if self.pending_wicket is not None and wickets == pw + 1:
            pending_runs, pending_wickets = self.pending_wicket
            if wickets == pending_wickets and runs >= pending_runs:
                if runs <= pending_runs + MAX_SCOREBOARD_JUMP_RUNS:
                    self.pending_wicket_count += 1
                    if self.pending_wicket_count >= MIN_SCORE_AGREE:
                        old_score = (pr, pw)
                        new_score = self.pending_wicket
                        transition_time = self.pending_wicket_first_time
                        transition_frame = self.pending_wicket_first_frame
                        self.current = new_score
                        self._clear_wicket()
                        self.last_time = t
                        return {
                            "innings": self.innings_id,
                            "old_score": old_score,
                            "new_score": new_score,
                            "transition_time": transition_time,
                            "transition_frame": transition_frame,
                            "confirmed_time": t,
                        }

        # --------------------------------------------------------
        # RUN-ONLY CHANGE
        # --------------------------------------------------------
        if wickets == pw:
            if runs < pr:
                self._clear_wicket()
                self.last_time = t
                return None

            self.current = (runs, wickets)
            # Keep a pending wicket only when the observed state is still
            # compatible with it; otherwise this was ordinary scoring.
            self._clear_wicket()
            self.last_time = t
            return None

        # --------------------------------------------------------
        # WICKET CHANGE
        # --------------------------------------------------------
        # EXACTLY +1 ONLY.
        if wickets != pw + 1:
            self._clear_wicket()
            self.last_time = t
            return None

        if wickets > MAX_WICKETS:
            self._clear_wicket()
            self.last_time = t
            return None

        if runs < pr:
            self._clear_wicket()
            self.last_time = t
            return None

        candidate = (runs, wickets)

        # Start/continue pending wicket.
        if self.pending_wicket != candidate:
            self.pending_wicket = candidate
            self.pending_wicket_count = 1
            self.pending_wicket_first_time = t
            self.pending_wicket_first_frame = frame_no
        else:
            self.pending_wicket_count += 1

        # CRITICAL:
        # Never confirm a wicket from confidence alone.
        # It MUST appear in at least MIN_SCORE_AGREE consecutive/compatible
        # OCR observations.
        if self.pending_wicket_count < MIN_SCORE_AGREE:
            self.last_time = t
            return None

        old_score = (pr, pw)
        new_score = candidate

        # Final sanity checks.
        if new_score == old_score:
            self._clear_wicket()
            self.last_time = t
            return None

        if new_score[1] != old_score[1] + 1:
            self._clear_wicket()
            self.last_time = t
            return None

        transition_time = self.pending_wicket_first_time
        transition_frame = self.pending_wicket_first_frame

        # Update confirmed state ONLY after the event is constructed.
        self.current = new_score
        self._clear_wicket()
        self.last_time = t

        return {
            "innings": self.innings_id,
            "old_score": old_score,
            "new_score": new_score,
            "transition_time": transition_time,
            "transition_frame": transition_frame,
            "confirmed_time": t,
        }

# ============================================================
# 10. OUTPUT FILES
# ============================================================
ocr_csv = OUTPUT_DIR / "ocr_results.csv"
event_csv = OUTPUT_DIR / "wicket_events.csv"

of = open(ocr_csv, "w", newline="", encoding="utf-8")
ef = open(event_csv, "w", newline="", encoding="utf-8")
ow = csv.writer(of)
ew = csv.writer(ef)

ow.writerow([
    "time_seconds", "frame_number", "ocr_text",
    "display_first", "display_second",
    "runs", "wickets", "confidence", "selection_score",
    "x1", "y1", "x2", "y2", "source_kind", "score_format"
])
ew.writerow([
    "time_seconds", "frame_number", "innings",
    "old_score", "new_score"
])

# ============================================================
# 11. PROCESS VIDEO
# ============================================================
resolver = FormatResolver()
scoreboard_track = ScoreboardTrack(width, height)
tracker = InningsAwareTracker()

buffer = []
event_rows = []
seen_events = set()

sample_count = 0
candidate_count = 0
valid_count = 0
frame_no = 0

# Dedicated lower-left scoreboard ROI.
roi_x1 = 0
roi_y1 = int(height * 0.62)
roi_x2 = int(width * 0.48)
roi_y2 = height

print("=" * 76)
print("STARTING OCR")
print("=" * 76)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_no % sample_every != 0:
        frame_no += 1
        continue

    t = frame_no / fps
    sample_count += 1

    try:
        full_results = reader.readtext(
            frame,
            detail=1,
            paragraph=False,
            text_threshold=0.30,
            low_text=0.15,
            link_threshold=0.15,
            mag_ratio=1.0,
        )
    except Exception as e:
        print(f"[OCR ERROR] {t:.2f}s: {e}", flush=True)
        frame_no += 1
        continue

    candidates = extract_score_candidates(full_results, width, height)

    # Dedicated scoreboard ROI.
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    try:
        roi_results = reader.readtext(
            roi,
            detail=1,
            paragraph=False,
            text_threshold=0.30,
            low_text=0.15,
            link_threshold=0.15,
            mag_ratio=1.0,
        )
        candidates.extend(extract_roi_candidates(roi_results, roi_x1, roi_y1))
    except Exception as e:
        print(f"[ROI OCR WARNING] {t:.2f}s: {e}", flush=True)

    # Deduplicate.
    dedup = {}
    for c in candidates:
        key = (
            c["display_first"], c["display_second"],
            round(c["cx"] / 5), round(c["cy"] / 5)
        )
        if key not in dedup or c["confidence"] > dedup[key]["confidence"]:
            dedup[key] = c
    candidates = list(dedup.values())
    candidate_count += len(candidates)

    best = scoreboard_track.rank(candidates)

    if best is not None:
        resolver.observe(
            best["display_first"],
            best["display_second"],
            best["confidence"]
        )

        buffer.append((t, frame_no, dict(best)))
        valid_count += 1

        print(
            f"[SCORE CANDIDATE] {t:.2f}s "
            f"display={best['display_first']}-{best['display_second']} "
            f"conf={best['confidence']:.3f} "
            f"source={best['source_kind']}",
            flush=True
        )

        # Process immediately once format is known.
        if resolver.format is not None:
            runs, wickets = resolver.normalize(
                best["display_first"], best["display_second"]
            )

            best["runs"] = runs
            best["wickets"] = wickets
            best["score_format"] = resolver.format

            event = tracker.process(
                t, frame_no, runs, wickets, best["confidence"]
            )

            if event:
                # Use the first observed transition time.
                innings = event["innings"]
                old_score = event["old_score"]
                new_score = event["new_score"]

                key = (
                    innings,
                    old_score,
                    new_score
                )

                if (
                        key not in seen_events
                        and new_score[1] == old_score[1] + 1
                        and new_score[1] <= MAX_WICKETS
                        and new_score != old_score
                    ):
                    seen_events.add(key)
                    event_rows.append((
                        event["transition_time"], event.get("transition_frame", frame_no), innings, old_score, new_score
                    ))
                    print(
                        f"[WICKET EVENT] {event["transition_time"]:.2f}s "
                        f"innings={innings} "
                        f"{old_score[0]}-{old_score[1]} -> "
                        f"{new_score[0]}-{new_score[1]}",
                        flush=True
                    )

            ow.writerow([
                f"{t:.2f}", frame_no, best["text"],
                best["display_first"], best["display_second"],
                runs, wickets, f"{best['confidence']:.4f}",
                f"{best['selection_score']:.4f}",
                best["x1"], best["y1"], best["x2"], best["y2"],
                best["source_kind"], resolver.format
            ])
            of.flush()

            print(
                f"[{t:7.2f}s] {runs}-{wickets} "
                f"format={resolver.format} text={best['text']!r}",
                flush=True
            )

    if sample_count % 10 == 0:
        print(
            f"[PROGRESS] {sample_count} samples | {t:.1f}s | "
            f"candidates={candidate_count} | valid={valid_count} | "
            f"events={len(event_rows)} | "
            f"innings={tracker.innings_id} | "
            f"format={resolver.format or 'detecting'} | "
            f"rw={resolver.evidence['runs-wickets']:.1f} "
            f"wr={resolver.evidence['wickets-runs']:.1f}",
            flush=True
        )

    frame_no += 1

cap.release()
of.close()
ef.close()

# ============================================================
# 12. FINALIZE FORMAT IF STILL AMBIGUOUS
# ============================================================
print(
    f"[FORMAT STATUS] {resolver.format or 'still ambiguous'} | "
    f"innings={tracker.innings_id}",
    flush=True
)

if resolver.format is None and buffer:
    rw = resolver.evidence["runs-wickets"]
    wr = resolver.evidence["wickets-runs"]
    resolver.format = "runs-wickets" if rw >= wr else "wickets-runs"
    resolver.reason = "best accumulated temporal evidence at end"
    print(
        f"[FORMAT FINALIZED] {resolver.format} ({resolver.reason})",
        flush=True
    )

# If format was not locked during the main pass, replay everything.
if resolver.format is not None and buffer and resolver.reason == "best accumulated temporal evidence at end":
    tracker = InningsAwareTracker()
    event_rows = []
    seen_events = set()

    # Rewrite OCR CSV with normalized scores.
    with open(ocr_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_seconds", "frame_number", "ocr_text",
            "display_first", "display_second",
            "runs", "wickets", "confidence", "selection_score",
            "x1", "y1", "x2", "y2", "source_kind", "score_format"
        ])

        for bt, bf, bc in buffer:
            runs, wickets = resolver.normalize(
                bc["display_first"], bc["display_second"]
            )

            event = tracker.process(
                bt, bf, runs, wickets, bc["confidence"]
            )

            w.writerow([
                f"{bt:.2f}", bf, bc["text"],
                bc["display_first"], bc["display_second"],
                runs, wickets, f"{bc['confidence']:.4f}",
                f"{bc['selection_score']:.4f}",
                bc["x1"], bc["y1"], bc["x2"], bc["y2"],
                bc["source_kind"], resolver.format
            ])

            if event:
                innings = event["innings"]
                old_score = event["old_score"]
                new_score = event["new_score"]
                key = (innings, old_score, new_score)
                if (
                        key not in seen_events
                        and new_score[1] == old_score[1] + 1
                        and new_score[1] <= MAX_WICKETS
                        and new_score != old_score
                    ):
                    seen_events.add(key)
                    event_rows.append((
                        event["transition_time"], event.get("transition_frame", bf), innings, old_score, new_score
                    ))

# ============================================================
# 13. WRITE FINAL EVENT CSV
# ============================================================
with open(event_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "time_seconds", "frame_number", "innings",
        "old_score", "new_score"
    ])
    for t, fr, innings, old, new in event_rows:
        w.writerow([
            f"{t:.2f}", fr, innings,
            f"{old[0]}-{old[1]}",
            f"{new[0]}-{new[1]}"
        ])

# ============================================================
# 14. EXTRACT WICKET CLIPS
# ============================================================
created = []
cap2 = cv2.VideoCapture(str(VIDEO_PATH))

if cap2.isOpened() and fps > 0:
    for i, (t, fr, innings, old, new) in enumerate(event_rows, start=1):
        start = max(0.0, t - WICKET_CLIP_BEFORE)
        end = min(duration, t + WICKET_CLIP_AFTER)

        sf = max(0, int(round(start * fps)))
        ef = max(sf, int(round(end * fps)))

        cap2.set(cv2.CAP_PROP_POS_FRAMES, sf)

        outpath = SEGMENTS_DIR / f"innings_{innings}_wicket_{i:03d}.mp4"
        writer = cv2.VideoWriter(
            str(outpath),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            continue

        n = sf
        while n <= ef:
            ok, frx = cap2.read()
            if not ok:
                break
            writer.write(frx)
            n += 1

        writer.release()

        if outpath.exists() and outpath.stat().st_size > 0:
            created.append(outpath)

cap2.release()

# ============================================================
# 15. ZIP COMPLETE RESULTS
# ============================================================
zip_path = PROJECT_DIR / f"{VIDEO_PATH.stem}_ocr_results.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUTPUT_DIR))

print("=" * 76)
print("FINAL VALIDATION: tracker uses correct frame/time arguments; wicket confirmation tolerates post-wicket run changes; clips are 6s before + 6s after.")
print("OCR COMPLETE")
print("=" * 76)
print(f"Samples       : {sample_count}")
print(f"Candidates    : {candidate_count}")
print(f"Valid reads   : {valid_count}")
print(f"Format        : {resolver.format or 'undetermined'}")
print(f"Innings       : {tracker.innings_id}")
print(f"Wicket events : {len(event_rows)}")
print(f"OCR CSV       : {ocr_csv}")
print(f"Events CSV    : {event_csv}")
print(f"Wicket clips  : {SEGMENTS_DIR}")
print(f"ZIP results   : {zip_path}")
print("=" * 76)

files.download(str(zip_path))
