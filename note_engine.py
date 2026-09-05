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
            with self._lock:
                if not self._recording:
                    return
                self._frames.append(indata.copy())

            # Level meter
            if self._on_level:
                f = indata.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(f ** 2)))
                level = min(1.0, rms * 5.0)
                self._on_level(level)

            # Silence detection
            f = indata.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(f ** 2)))
            now = time.monotonic()
            elapsed = now - self._chunk_start_time

            if rms < self.RMS_THRESHOLD:
                if self._silence_start is None:
                    self._silence_start = now
                else:
                    silence_dur = now - self._silence_start
                    required = self._required_silence(elapsed)
                    if elapsed >= self.MIN_CHUNK_SEC and silence_dur >= required:
                        self._cut_chunk()
                    elif elapsed >= self.MAX_CHUNK_SEC:
                        self._cut_chunk()
            else:
                self._silence_start = None

            # Hard cut at MAX
            if elapsed >= self.MAX_CHUNK_SEC + 2.0:
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
            audio = self._trim_trailing_silence(audio)

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


class NoteTranscriber:
    """Processes chunks asynchronously and accumulates text."""

    def __init__(self):
        self._results: List[str] = []
        self._lock = threading.Lock()
        self._processing = False
        self._queue: List[tuple] = []
        self._queue_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_text: Optional[Callable[[str, int], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._transcribe_fn: Optional[Callable] = None

    def set_callbacks(self, on_text=None, on_status=None):
        self._on_text = on_text
        self._on_status = on_status

    def set_transcribe_fn(self, fn):
        """Set the transcription function: fn(audio, sr) -> str"""
        self._transcribe_fn = fn

    @property
    def is_processing(self) -> bool:
        return self._processing

    @property
    def full_text(self) -> str:
        with self._lock:
            return "\n".join(self._results)

    @property
    def result_count(self) -> int:
        with self._lock:
            return len(self._results)

    def start(self):
        self._results = []
        self._queue = []
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def submit_chunk(self, audio: np.ndarray, index: int):
        with self._queue_lock:
            self._queue.append((audio, index))

    def stop(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=30)

    def _worker(self):
        while not self._stop_event.is_set():
            item = None
            with self._queue_lock:
                if self._queue:
                    item = self._queue.pop(0)

            if item is None:
                time.sleep(0.1)
                continue

            audio, index = item
            self._processing = True
            try:
                if self._on_status:
                    self._on_status(f"Transcribing chunk {index}...")

                text = ""
                if self._transcribe_fn:
                    text = self._transcribe_fn(audio, 16000)

                if text and text.strip() and not text.startswith("["):
                    with self._lock:
                        self._results.append(text.strip())
                    if self._on_text:
                        self._on_text(text.strip(), index)
                    if self._on_status:
                        self._on_status(f"Chunk {index} done ({len(text)} chars)")
                else:
                    if self._on_status:
                        self._on_status(f"Chunk {index}: no speech detected")

            except Exception as e:
                if self._on_status:
                    self._on_status(f"Chunk {index} error: {e}")
            finally:
                self._processing = False

        # Process remaining queue items
        with self._queue_lock:
            remaining = self._queue[:]
            self._queue = []
        for audio, index in remaining:
            try:
                if self._transcribe_fn:
                    text = self._transcribe_fn(audio, 16000)
                    if text and text.strip() and not text.startswith("["):
                        with self._lock:
                            self._results.append(text.strip())
            except Exception:
                pass
