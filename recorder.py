# recorder.py
import numpy as np
import sounddevice as sd
import threading
from typing import Optional, Callable
class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._on_level: Optional[Callable[[float], None]] = None
    @property
    def is_recording(self) -> bool:
        return self._recording
    def set_level_callback(self, callback: Optional[Callable[[float], None]]):
        self._on_level = callback
    def start(self):
        if self._recording:
            return
        self._frames = []
        self._recording = True
        def audio_callback(indata, frames, time_info, status):
            try:
                with self._lock:
                    if not self._recording:
                        return
                    self._frames.append(indata.copy())
                if self._on_level:
                    f = indata.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(f ** 2)))
                    level = min(1.0, rms * 5.0)
                    self._on_level(level)
            except Exception:
                pass
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=audio_callback,
                blocksize=1024,
            )
            self._stream.start()
        except Exception:
            self._recording = False
            self._stream = None
            raise
    def stop(self) -> Optional[np.ndarray]:
        if not self._recording:
            return None
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return None
        try:
            return np.concatenate(frames, axis=0)
        except Exception:
            return None
    def get_level(self) -> float:
        if not self._recording or not self._frames:
            return 0.0
        with self._lock:
            if not self._frames:
                return 0.0
            f = self._frames[-1].astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(f ** 2)))
            return min(1.0, rms * 10.0)
    @staticmethod
    def list_devices():
        return sd.query_devices()
    @staticmethod
    def get_default_input():
        try:
            dev = sd.query_devices(kind="input")
            return dev["name"]
        except Exception:
            return None