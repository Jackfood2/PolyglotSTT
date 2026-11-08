import pathlib
# recorder.py
import numpy as np
import sounddevice as sd
import threading
from typing import Optional, Callable
class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
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
    def set_device(self, device):
        """Pick an input device index (None, -1 or "" = system default)."""
        try:
            if device in (None, "", "-1", -1):
                self.device = None
            else:
                self.device = int(device)
        except Exception:
            self.device = None
    def _resolved_device(self):
        """The chosen index, or None (system default) when the saved device
        is gone, disabled, or not a capture endpoint at all."""
        if self.device is None:
            return None
        try:
            info = sd.query_devices(int(self.device))
            if int(info.get("max_input_channels") or 0) > 0:
                return int(self.device)
        except Exception:
            pass
        return None
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
                device=self._resolved_device(),
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
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
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
    def list_input_devices():
        """Capture endpoints usable at 16 kHz (see module-level helper)."""
        return list_input_devices()
    @staticmethod
    def get_default_input():
        try:
            dev = sd.query_devices(kind="input")
            return dev["name"]
        except Exception:
            return None
def list_input_devices():
    """Capture endpoints the app can actually open at 16 kHz, best first.

    Only DirectSound and MME are listed: both resample inside Windows, so
    they accept the 16 kHz the engines expect. WASAPI refuses anything but
    the endpoint's native rate and WDM-KS fails on most drivers. Entries
    that just alias "whatever Windows defaults to" are skipped, and the
    same endpoint exposed by both host APIs is listed once (DirectSound
    first, because its names are not truncated).
    """
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return []
    order = ("Windows DirectSound", "MME")
    out = []
    seen = set()
    for wanted in order:
        for ha in hostapis:
            try:
                if str(ha.get("name") or "") != wanted:
                    continue
                idxs = list(ha.get("devices") or [])
            except Exception:
                continue
            for idx in idxs:
                try:
                    d = devices[idx]
                    if int(d.get("max_input_channels") or 0) <= 0:
                        continue
                    name = str(d.get("name") or "").strip()
                except Exception:
                    continue
                low = name.lower()
                if (not name or low.startswith("microsoft sound mapper")
                        or low.startswith("primary sound")):
                    continue
                if low in seen:
                    continue
                # MME truncates long names; skip the clipped twin.
                if wanted == "MME" and any(
                        s.startswith(low) for s in seen):
                    continue
                seen.add(low)
                out.append({"index": int(idx), "name": name})
    return out
def default_input_index():
    """PortAudio index of the Windows default capture device, or -1."""
    try:
        return int(sd.default.device[0])
    except Exception:
        return -1