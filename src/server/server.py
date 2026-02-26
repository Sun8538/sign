import os
import json
import time
import logging
import sqlite3
import torch
from dotenv import load_dotenv
from flask import Flask
from flask_socketio import SocketIO, emit

# Fix for PyTorch 2.6+ weights_only default change
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils.store import Store

# Configuration - load .env from project root or server directory
_server_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_server_dir, "..", ".."))
load_dotenv(os.path.join(_server_dir, ".env"))
load_dotenv(os.path.join(_project_root, ".env"))
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

print("[BOOT] Starting server...")
_t0 = time.time()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Lazy-loaded heavy modules (loaded in background after server binds) ──
recognition = None
embedding_model = None
llm = None
_boot_ready = [False]


def _load_heavy_modules():
    """Load ML models in a background thread so the server socket binds immediately."""
    global recognition, embedding_model, llm
    import threading

    def _inner():
        global recognition, embedding_model, llm
        try:
            from utils.recognition import Recognition
            recognition = Recognition(min_confidence=0.65)  # lowered from 0.80 (safe with EMA)
            print(f"[BOOT] Recognition ready  ({time.time()-_t0:.1f}s)")

            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"[BOOT] Embedding model ready ({time.time()-_t0:.1f}s)")

            from utils.llm import LLM
            llm = LLM()
            print(f"[BOOT] LLM ready ({time.time()-_t0:.1f}s)")

            from utils.bert import Bert
            Bert.set_socketio(socketio)

            _boot_ready[0] = True
            print(f"[BOOT] All modules loaded in {time.time()-_t0:.1f}s")
        except Exception as e:
            print(f"[ERROR] Background loading failed: {e}")
            import traceback; traceback.print_exc()

    t = threading.Thread(target=_inner, daemon=True)
    t.start()


_load_heavy_modules()

# Database — use SQLite (PostgreSQL is optional and rarely available)
db_type = "sqlite"
try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(
        database="signs",
        host="localhost",
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
        port=5432,
    )
    register_vector(conn)
    db_type = "postgres"
    print("Connected to PostgreSQL")
except Exception:
    conn = sqlite3.connect("signs.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS signs (word TEXT, points TEXT, embedding BLOB)")
    conn.commit()
    cursor.close()
    print("Using SQLite")

# Store Fingerspelling Animations
alphabet_frames = {}
_alphabets_dir = os.path.join(_server_dir, "alphabets")
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    file_path = os.path.join(_alphabets_dir, f"{letter}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            alphabet_frames[letter] = json.load(file)
    else:
        print(f"[WARN] Missing alphabet file: {letter}.json")
        alphabet_frames[letter] = []


@app.route("/")
def index():
    return "ASL Translation Server running. Browser-based detection active.", 200


@socketio.on("connect")
def on_connect():
    """Triggered when client-server SocketIO connection is established"""
    print("Connected to client")
    emit("R-TRANSCRIPTION", Store.parsed)

    # Send hello sign (only if embedding model is loaded)
    if embedding_model is not None:
        try:
            cursor = conn.cursor()
            animations = []
            embedding = embedding_model.encode("hello")

            if db_type == "postgres":
                cursor.execute(
                    "SELECT word, points, (embedding <=> %s) AS cosine_similarity FROM signs ORDER BY cosine_similarity ASC LIMIT 1",
                    (embedding,),
                )
                result = cursor.fetchone()
                if result and 1 - result[2] > 0.70:
                    animations.append(("hello", result[1]))
            else:
                cursor.execute("SELECT word, points FROM signs WHERE word = ?", ("hello",))
                result = cursor.fetchone()
                if result:
                    animations.append(("hello", result[1]))

            emit("E-ANIMATION", animations)
            cursor.close()
        except Exception as e:
            print(f"[WARN] Hello animation failed: {e}")


@socketio.on("R-CLEAR-TRANSCRIPTION")
def on_clear_transcription():
    """Triggered when client requests to clear the receptive transcription"""

    Store.reset()
    emit("R-TRANSCRIPTION", Store.parsed)
    log.log(logging.INFO, "STORE RESET")


@socketio.on("E-REQUEST-ANIMATION")
def on_request_animation(words: str):
    """Triggered when client requests an expressive animation for a word or sentence"""

    animations = []
    words = words.strip()

    if not words:
        return

    # Gloss the words (LLM may still be loading)
    if llm is not None:
        words = llm.gloss(words)
    else:
        words = words.upper().split()
    # words = words.split()

    cursor = conn.cursor()
    for word in words:
        word = word.strip()
        if not word:
            continue

        if embedding_model is None:
            # Model still loading — fingerspell everything
            animation = []
            for ch in word:
                animation.extend(alphabet_frames.get(ch.upper(), []))
            for i in range(len(animation)):
                animation[i][0] = i
            animations.append((f"fs-{word.upper()}", animation))
            continue

        embedding = embedding_model.encode(word)
        result = None
        
        if db_type == "postgres":
            cursor.execute(
                "SELECT word, points, (embedding <=> %s) AS cosine_similarity FROM signs ORDER BY cosine_similarity ASC LIMIT 1",
                (embedding,),
            )
            result = cursor.fetchone()
            # Add sign to animation
            if result and 1 - result[2] > 0.70:
                animations.append((word, result[1]))
            else:
                result = None  # Force fingerspelling
        else:
            # SQLite fallback: Exact match for simplicity
            cursor.execute("SELECT word, points FROM signs WHERE word = ?", (word,))
            result = cursor.fetchone()
            if result:
                animations.append((word, result[1]))

        # Add fingerspell to animation if no sign found
        if result is None or (db_type == "sqlite" and not result):
            animation = []
            for letter in word:
                animation.extend(alphabet_frames.get(letter.upper(), []))

            for i in range(len(animation)):
                animation[i][0] = i
            animations.append((f"fs-{word.upper()}", animation))

        if "." in word:
            space = []
            if animations and animations[-1][1]:
                last_frame = animations[-1][1][-1]
                for i in range(50):
                    space.append(last_frame)
                    space[-1][0] = i
                animations.append(("", space))

    print(f"Emitting animations for {len(animations)} words: {[a[0] for a in animations]}")
    emit("E-ANIMATION", animations)
    cursor.close()


_last_hand_time = [0.0]  # mutable container for last hand detection time
_no_hand_frames = [0]  # count of consecutive no-hand frames
_last_classify_time = [0.0]  # throttle classification calls
_CLASSIFY_INTERVAL = 0.06  # ~16 fps max classification rate (seconds)
_autocorrect_enabled = [True]  # per-session autocorrect toggle


@socketio.on("set-autocorrect")
def on_set_autocorrect(data):
    """Toggle autocorrect on/off from the client UI checkbox."""
    _autocorrect_enabled[0] = bool(data.get("enabled", True))
    print(f"[CONFIG] Autocorrect {'ON' if _autocorrect_enabled[0] else 'OFF'}")

@socketio.on("hand-landmarks")
def on_hand_landmarks(data):
    """Handle hand landmarks from browser-based detection.

    Uses EMA-smoothed softmax + majority-vote for stable letter acceptance.
    """
    import numpy as np
    try:
        if recognition is None:
            return  # ML models still loading

        now = time.time()
        if now - _last_classify_time[0] < _CLASSIFY_INTERVAL:
            return
        _last_classify_time[0] = now

        landmarks = np.array(data["landmarks"], dtype=np.float32)
        if landmarks.shape != (21, 3):
            return

        _last_hand_time[0] = now
        _no_hand_frames[0] = 0

        # Normalize landmarks
        min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
        range_x = max_x - min_x if (max_x - min_x) > 1e-6 else 1.0
        range_y = max_y - min_y if (max_y - min_y) > 1e-6 else 1.0

        normalized = landmarks.copy()
        normalized[:, 0] = (normalized[:, 0] - min_x) / range_x
        normalized[:, 1] = (normalized[:, 1] - min_y) / range_y
        normalized = np.expand_dims(normalized, axis=0)

        # Classify — returns EMA-smoothed letter + confidence
        letter, confidence = recognition.classifier.classify(normalized)

        # Post-process with handedness-aware geometric rules
        handedness = data.get("handedness", "right")
        letter = recognition.fix_misrecognition(letter, normalized, handedness)

        # ── Majority-vote letter acceptance ──
        # The classifier internally maintains a sliding window of recent letters.
        # We accept a letter only when ≥60% of the window agrees AND confidence
        # exceeds the (now lower, EMA-smoothed) threshold.
        if confidence > recognition.min_confidence:
            majority_letter, ratio = recognition.classifier.majority_letter()
            if majority_letter is not None:
                # Accept the letter — but prevent double-runs (e.g. "AA" → only if intentional)
                if not Store.raw_word or Store.raw_word[-1] != majority_letter:
                    Store.raw_word += majority_letter
                    print(f"[LOG] Letter accepted: {majority_letter}  (conf={confidence:.0%}, vote={ratio:.0%})")
                elif len(Store.raw_word) >= 2 and Store.raw_word[-2:] != majority_letter * 2:
                    # Allow a repeated letter only once (e.g. "LL" ok, "LLL" blocked)
                    Store.raw_word += majority_letter
                    print(f"[LOG] Letter repeated: {majority_letter}  (conf={confidence:.0%}, vote={ratio:.0%})")

        # Build output and emit
        output = (
            " ".join(Store.corrected_transcription) + " " + Store.raw_word
        ).strip()
        if output != Store.parsed:
            Store.parse(output)

        socketio.emit("R-TRANSCRIPTION", Store.parsed)
        socketio.emit("R-LIVE-LETTER", {
            "letter": letter,
            "confidence": confidence,
        })

    except Exception as e:
        print(f"[ERROR] Hand landmarks classification error: {e}")
        import traceback
        traceback.print_exc()


@socketio.on("no-hand-detected")
def on_no_hand():
    """Handle when browser detects no hand — triggers word completion."""
    import threading

    _no_hand_frames[0] += 1

    # Reset EMA state so the next hand appearance starts fresh
    if recognition is not None and _no_hand_frames[0] == 1:
        recognition.classifier.reset_ema()

    # After ~15 consecutive no-hand signals (~1 sec at 16fps), finalize word
    if _no_hand_frames[0] >= 15 and Store.raw_word:
        Store.raw_transcription.append(Store.raw_word)
        print(f"[LOG] Word completed: {Store.raw_word}")

        if _autocorrect_enabled[0]:
            from utils.bert import Bert
            thread = threading.Thread(target=Bert.fix)
            thread.daemon = True
            thread.start()
        else:
            Store.raw_word = ""
            raw_transcription = " ".join(Store.raw_transcription).upper()
            Store.corrected_transcription = raw_transcription.split()
            output = raw_transcription.strip()
            Store.parse(output)
            socketio.emit("R-TRANSCRIPTION", Store.parsed)

        _no_hand_frames[0] = 0
        socketio.emit("R-TRANSCRIPTION", Store.parsed)


@socketio.on("disconnect")
def on_disconnect():
    log.log(logging.INFO, "Disconnected from client")


if __name__ == "__main__":
    socketio.run(app, debug=False, port=1234)
