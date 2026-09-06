# note_engine.py
# Chunked recording with adaptive silence detection for Note mode
import threading
import time
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, List
from pathlib import Path


class NoteRecorder:
    """Records audio and cuts at natural pauses between 40-80 seconds."""

    MIN_CHUNK_SEC = 40.0
    MAX_CHUNK_SEC = 80.0
    SILENCE_AT_MIN = 1.8   # silence needed at 40s mark
    SILENCE_AT_MAX = 0.4   # silence needed at 80s mark
    FRAME_MS = 30
    RMS_THRESHOLD = 0.008  # below this = silence
    # Mic-health watchdog thresholds (read by the GUI timer tick).
    HEALTH_RMS_FLOOR = 0.001  # at/above this counts as "real input"
    DEAD_AFTER_SEC = 5.0  # no callback frames at all -> stream dead
    SILENT_AFTER_SEC = 10.0  # callbacks flow but digital-zero -> suspicious

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._recording = False
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._chunk_lock = threading.Lock()

        # Audio buffer for current chunk
        self._frames: List[np.ndarray] = []
        self._chunk_start_time: float = 0.0

        # Silence detection
        self._silence_start: Optional[float] = None
        self._current_elapsed: float = 0.0

        # Mic-health timestamps (plain attributes, no locks: single-float
        # stores are atomic enough, and the GUI only reads them ~1/sec).
        self._last_cb_time: float = 0.0
        self._last_sound_time: float = 0.0

        # Callbacks
        self._on_chunk_ready: Optional[Callable[[np.ndarray, int], None]] = None
        self._on_level: Optional[Callable[[float], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._chunk_index: int = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def chunk_count(self) -> int:
        return self._chunk_index

    def set_callbacks(self, on_chunk_ready=None, on_level=None, on_status=None):
        self._on_chunk_ready = on_chunk_ready
        self._on_level = on_level
        self._on_status = on_status

    def _required_silence(self, elapsed_sec: float) -> float:
        """Adaptive silence threshold: longer silence needed early, shorter later."""
        if elapsed_sec <= self.MIN_CHUNK_SEC:
            return self.SILENCE_AT_MIN
        if elapsed_sec >= self.MAX_CHUNK_SEC:
            return self.SILENCE_AT_MAX
        t = (elapsed_sec - self.MIN_CHUNK_SEC) / (self.MAX_CHUNK_SEC - self.MIN_CHUNK_SEC)
        return self.SILENCE_AT_MIN - t * (self.SILENCE_AT_MIN - self.SILENCE_AT_MAX)

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            should_cut = False
            with self._lock:
                if not self._recording:
                    return
                self._frames.append(indata.copy())

                f = indata.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(f ** 2)))

                if self._on_level:
                    self._on_level(min(1.0, rms * 5.0))

                now = time.monotonic()
                elapsed = now - self._chunk_start_time
                self._last_cb_time = now
                if rms > self.HEALTH_RMS_FLOOR:
                    self._last_sound_time = now

                if rms < self.RMS_THRESHOLD:
                    if self._silence_start is None:
                        self._silence_start = now
                    else:
                        silence_dur = now - self._silence_start
                        required = self._required_silence(elapsed)
                        if elapsed >= self.MIN_CHUNK_SEC and silence_dur >= required:
                            should_cut = True
                        elif elapsed >= self.MAX_CHUNK_SEC:
                            should_cut = True
                else:
                    self._silence_start = None

                if elapsed >= self.MAX_CHUNK_SEC + 2.0:
                    should_cut = True

            if should_cut:
                self._cut_chunk()
        except Exception:
            pass

    def _cut_chunk(self):
        """Cut current buffer into a chunk and emit it."""
        with self._chunk_lock:
            with self._lock:
                if not self._frames:
                    return
                frames = self._frames[:]
                self._frames = []
                self._chunk_start_time = time.monotonic()
                self._silence_start = None

            audio = np.concatenate(frames, axis=0)
            # Trim trailing silence (last 0.5s max)
            audio = self._trim_trailing_silence(audio, self.sample_rate)

            if len(audio) < self.sample_rate * 2:  # skip < 2s
                return

            self._chunk_index += 1
            idx = self._chunk_index

            if self._on_chunk_ready:
                self._on_chunk_ready(audio, idx)

            if self._on_status:
                dur = len(audio) / self.sample_rate
                self._on_status(f"Chunk {idx} ({dur:.0f}s) sent for processing")

    @staticmethod
    def _trim_trailing_silence(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Remove up to 1.5s of trailing silence."""
        if len(audio) == 0:
            return audio
        f = audio.astype(np.float32) / 32768.0
        frame_len = int(sr * 0.03)
        n_frames = len(f) // frame_len
        if n_frames == 0:
            return audio
        # Check from the end
        trim_frames = 0
        max_trim = int(1.5 / 0.03)  # 1.5s worth of frames
        for i in range(n_frames - 1, max(0, n_frames - max_trim - 1), -1):
            chunk = f[i * frame_len:(i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < 0.006:
                trim_frames += 1
            else:
                break
        if trim_frames > 0:
            cut_samples = trim_frames * frame_len
            audio = audio[:-cut_samples] if cut_samples < len(audio) else audio
        return audio

    def start(self):
        if self._recording:
            return
        self._frames = []
        self._chunk_index = 0
        self._silence_start = None
        self._chunk_start_time = time.monotonic()
        self._last_cb_time = time.monotonic()
        # Grace from record-press: a not-yet-speaking user is not a dead mic.
        self._last_sound_time = self._last_cb_time
        self._recording = True

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
        except Exception as e:
            self._recording = False
            self._stream = None
            raise RuntimeError(f"Microphone error: {e}")

    def stop(self) -> Optional[np.ndarray]:
        """Stop recording and return any remaining audio."""
        if not self._recording:
            return None
        self._recording = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._lock:
            frames = self._frames[:]
            self._frames = []

        if frames:
            audio = np.concatenate(frames, axis=0)
            audio = self._trim_trailing_silence(audio, self.sample_rate)
            if len(audio) > self.sample_rate * 1:  # > 1s
                return audio
        return None

    def get_elapsed(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._chunk_start_time

    def mic_health(self):
        """(state, detail) for the GUI watchdog tick: "ok" | "dead" (no
        callback frames or stream down) | "silent" (frames flow but only
        digital-zero for a while) | "idle" (not recording). Lock-free
        read; the tick runs ~1/sec so cost is nil."""
        try:
            if not self._recording:
                return "idle", ""
            now = time.monotonic()
            try:
                alive = self._stream is not None
                if alive:
                    try:
                        alive = bool(self._stream.active)
                    except Exception:
                        pass
            except Exception:
                alive = False
            if not alive:
                return "dead", "stream down"
            if now - (self._last_cb_time or now) > self.DEAD_AFTER_SEC:
                return "dead", "no audio frames"
            if now - (self._last_sound_time or now) > self.SILENT_AFTER_SEC:
                return "silent", "digital silence"
            return "ok", ""
        except Exception:
            return "ok", ""


class NoteTranscriber:
    """Processes chunks asynchronously and accumulates text."""

    # Engine failure strings (NOT transcriptions): surface them via status
    # instead of appending - or worse, silently dropping as "no speech".
    # Genuine text - even "[Music]"-led - always passes through to append.
    _ERROR_PREFIXES = ("[Error:", "[Whisper Error:", "[Canary Error:")

    def __init__(self):
        self._queue: List[tuple] = []
        self._queue_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._gen = 0  # session generation: stale workers drop callbacks
        self._on_text: Optional[Callable[[str, int], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_done: Optional[Callable[[int, bool], None]] = None
        self._transcribe_fn: Optional[Callable] = None

    def set_callbacks(self, on_text=None, on_status=None, on_done=None):
        self._on_text = on_text
        self._on_status = on_status
        self._on_done = on_done

    def set_transcribe_fn(self, fn):
        """Set the transcription function: fn(audio, sr) -> str"""
        self._transcribe_fn = fn

    def start(self):
        with self._queue_lock:
            # New session: old worker (if still finishing a tail chunk)
            # carries a stale generation - its late callbacks are dropped
            # and it never touches the new queue.
            self._gen += 1
            self._queue = []
        self._stop_event.clear()
        # Never join a previous worker here: start() runs on the GUI
        # thread and the old one may be mid-transcription.
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def submit_chunk(self, audio: np.ndarray, index: int):
        with self._queue_lock:
            self._queue.append((audio, index))

    def stop(self):
        # Event-only: the worker keeps draining the queue WITH callbacks
        # (so the tail chunk still appears), then exits itself. Returns at
        # once - the old join(timeout=30) froze the UI mid-transcription.
        self._stop_event.set()

    def _current_gen(self, my_gen: int) -> bool:
        try:
            with self._queue_lock:
                return self._gen == my_gen
        except Exception:
            return False

    def _worker(self):
        with self._queue_lock:
            my_gen = self._gen
        while True:
            with self._queue_lock:
                if self._gen != my_gen:
                    break  # superseded by a newer session
                if self._stop_event.is_set() and not self._queue:
                    break  # drained after stop
                item = self._queue.pop(0) if self._queue else None
            if item is None:
                time.sleep(0.1)
                continue

            audio, index = item
            try:
                if self._on_status and self._current_gen(my_gen):
                    self._on_status(f"Transcribing chunk {index}...")

                text = ""
                if self._transcribe_fn:
                    text = self._transcribe_fn(audio, 16000)

                txt = (text or "").strip()
                if txt and not txt.startswith(self._ERROR_PREFIXES):
                    if self._on_text and self._current_gen(my_gen):
                        self._on_text(txt, index)
                    if self._on_status and self._current_gen(my_gen):
                        self._on_status(f"Chunk {index} done ({len(txt)} chars)")
                    if self._on_done and self._current_gen(my_gen):
                        self._on_done(index, True)
                elif txt:
                    # Engine failure string, NOT a transcription: report it
                    # as a failure instead of appending - or misreporting
                    # "no speech detected" as the old code did.
                    if self._on_status and self._current_gen(my_gen):
                        self._on_status(f"Chunk {index} failed: {txt}")
                    if self._on_done and self._current_gen(my_gen):
                        self._on_done(index, False)
                else:
                    if self._on_status and self._current_gen(my_gen):
                        self._on_status(f"Chunk {index}: no speech detected")
                    if self._on_done and self._current_gen(my_gen):
                        self._on_done(index, False)

            except Exception as e:
                if self._on_status and self._current_gen(my_gen):
                    self._on_status(f"Chunk {index} error: {e}")
                if self._on_done and self._current_gen(my_gen):
                    self._on_done(index, False)
