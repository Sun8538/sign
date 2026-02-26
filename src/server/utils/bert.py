import threading
from spellchecker import SpellChecker
from utils.store import Store


class Bert:
    """Spell-correction utility using pyspellchecker (fast, dictionary-based).

    Replaces the previous NeuSpell-based checker which had corrupted model
    downloads and concurrent-initialization race conditions.
    """

    _checker: SpellChecker | None = None
    _initialized = False
    _lock = threading.Lock()
    _socketio = None  # set by server.py so fix() can emit updates

    @classmethod
    def set_socketio(cls, sio):
        """Give Bert a reference to the SocketIO instance for real-time emit."""
        cls._socketio = sio

    @classmethod
    def _ensure_initialized(cls):
        if cls._initialized:
            return
        with cls._lock:
            if cls._initialized:
                return
            try:
                cls._checker = SpellChecker()
                cls._initialized = True
                print("[OK] SpellChecker (pyspellchecker) loaded.")
            except Exception as e:
                print(f"[WARN] Could not initialize SpellChecker: {e}")
                print("Spell checking will be disabled.")
                cls._initialized = True  # don't retry

    @classmethod
    def _correct_text(cls, text: str) -> str:
        """Correct a multi-word string using pyspellchecker."""
        if not cls._checker:
            return text

        words = text.split()
        corrected_words = []
        for w in words:
            if not w:
                continue
            # If word is known, keep it; otherwise try correction
            if cls._checker.unknown([w]):
                candidate = cls._checker.correction(w)
                corrected_words.append(candidate if candidate else w)
            else:
                corrected_words.append(w)
        return " ".join(corrected_words)

    @classmethod
    def fix(cls):
        """Run spell correction and broadcast the result via SocketIO."""
        Store.raw_word = ""

        raw_transcription = " ".join(Store.raw_transcription).lower()
        if not raw_transcription.strip():
            return

        cls._ensure_initialized()

        try:
            corrected = cls._correct_text(raw_transcription).strip()
        except Exception as e:
            print(f"[WARN] Spell correction error: {e}")
            corrected = raw_transcription.strip()

        Store.corrected_transcription = corrected.upper().split()

        # Build the new display string and emit it immediately
        output = (
            " ".join(Store.corrected_transcription) + " " + Store.raw_word
        ).strip()
        Store.parse(output)

        if cls._socketio is not None:
            cls._socketio.emit("R-TRANSCRIPTION", Store.parsed)
        print(f"[SPELL] '{raw_transcription}' → '{corrected}'")
