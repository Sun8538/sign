import os
import keras
import numpy as np
from collections import deque

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
NUM_CLASSES = len(LETTERS)  # 24

# Try multiple model files in order of preference
MODEL_CANDIDATES = [
    "model5.keras",
    "model6.keras",
    "model4.keras",
    "model3.keras",
    "model2.keras",
    "model.keras",
    "model.h5",
]

# ── EMA (Exponential Moving Average) smoothing config ──
_EMA_ALPHA = 0.4          # weight of newest frame (higher = more responsive, lower = smoother)
_MAJORITY_WINDOW = 8      # sliding window size for majority-vote acceptance
_MAJORITY_RATIO = 0.60    # letter must appear in ≥60% of window to be accepted


class Classifier:
    def __init__(self):
        self.model = None
        server_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(server_dir)  # src/server/

        for model_name in MODEL_CANDIDATES:
            model_path = os.path.join(parent_dir, model_name)
            if os.path.exists(model_path):
                try:
                    self.model = keras.models.load_model(model_path)
                    print(f"[OK] Classifier loaded model: {model_name}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load {model_name}: {e}")
                    continue

        if self.model is None:
            print("[ERROR] No classifier model could be loaded!")
        else:
            # Warm up the model with a dummy prediction so the first real
            # call doesn't incur the tf.function tracing overhead.
            try:
                dummy = np.zeros((1, 21, 2), dtype=np.float32)
                self.model(dummy, training=False)
                print("[OK] Classifier warmed up (no more retracing)")
            except Exception as e:
                print(f"[WARN] Warm-up failed: {e}")

        # ── EMA state ──
        self._ema_probs = np.zeros(NUM_CLASSES, dtype=np.float64)  # running smoothed softmax
        self._ema_initialized = False
        # Sliding window of recent smoothed-letter predictions (for majority vote)
        self._recent_letters: deque = deque(maxlen=_MAJORITY_WINDOW)

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────
    def classify(self, points):
        """Return (letter, smoothed_confidence) using EMA-smoothed softmax."""
        if self.model is None:
            return "A", 0.0

        try:
            # Ensure correct input shape: (batch, 21, 2)
            input_data = points[:, :, :2]
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
            else:
                input_data = input_data.astype(np.float32)

            # Use model.__call__ directly (avoids tf.function retracing)
            raw_probs = np.array(self.model(input_data, training=False)[0], dtype=np.float64)

            # ── EMA smoothing ──
            if not self._ema_initialized:
                self._ema_probs = raw_probs.copy()
                self._ema_initialized = True
            else:
                self._ema_probs = _EMA_ALPHA * raw_probs + (1 - _EMA_ALPHA) * self._ema_probs

            idx = int(np.argmax(self._ema_probs))
            if idx < 0 or idx >= NUM_CLASSES:
                return "A", 0.0

            letter = LETTERS[idx]
            confidence = float(self._ema_probs[idx])

            # Record letter in sliding window for majority vote
            self._recent_letters.append(letter)

            return letter, confidence
        except Exception as e:
            print(f"[ERROR] Classification error: {e}")
            return "A", 0.0

    def majority_letter(self) -> tuple[str | None, float]:
        """Return the majority letter if it exceeds the ratio threshold, else None.

        Returns:
            (letter, ratio)  – letter is None when no consensus.
        """
        if len(self._recent_letters) < 3:
            return None, 0.0
        from collections import Counter
        counts = Counter(self._recent_letters)
        top_letter, top_count = counts.most_common(1)[0]
        ratio = top_count / len(self._recent_letters)
        if ratio >= _MAJORITY_RATIO:
            return top_letter, ratio
        return None, ratio

    def reset_ema(self):
        """Reset temporal state (call when hand disappears)."""
        self._ema_probs = np.zeros(NUM_CLASSES, dtype=np.float64)
        self._ema_initialized = False
        self._recent_letters.clear()
