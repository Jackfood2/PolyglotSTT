# recorder.py
import re
import threading
import time
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

# ── PortAudio lifecycle ────────────────────────────────────────────────
# MME and DirectSound enumerate their endpoints once, at Pa_Initialize(),
# so a mic unplugged and plugged back in while the app runs keeps its
# stale index (and its dead endpoint handle) until PortAudio is restarted.
# Restarting PortAudio is only safe while no stream is open: Pa_Terminate
# with a live stream wedges the host API, and from then on every open
# fails with -9999 / DirectSound -2005401480 until the process exits.
# One lock plus one registry of live streams keeps the two apart, and a
# stream that only half-opened is closed immediately instead of lingering.
_DEV_LOCK = threading.RLock()
_LIVE_STREAMS: set = set()


def open_input_stream(**kwargs):
    """Open and start a PortAudio input stream, leaking nothing on failure.

    Every capture stream in the app goes through here (Live and Note), so a
    re-enumeration can tell whether PortAudio is in use. A stream whose
    ``start()`` failed is closed at once: a half-open stream left behind
    makes the next ``Pa_Terminate()`` fail, which keeps the endpoint list
    stale and leaves the mic unusable until the app is restarted.
    """
    with _DEV_LOCK:
        stream = sd.InputStream(**kwargs)
        try:
            stream.start()
        except Exception:
            close_input_stream(stream)
            raise
        _LIVE_STREAMS.add(stream)
        return stream


def close_input_stream(stream) -> None:
    """Close a stream completely. Never raises, never leaves it open."""
    if stream is None:
        return
    with _DEV_LOCK:
        _LIVE_STREAMS.discard(stream)
        for name in ("stop", "abort", "close"):
            try:
                getattr(stream, name)()
            except Exception:
                pass


def live_stream_count() -> int:
    """How many capture streams this app currently has open."""
    with _DEV_LOCK:
        return len(_LIVE_STREAMS)


def refresh_devices(force: bool = False) -> bool:
    """Force PortAudio to re-scan the Windows endpoint list.

    Returns True when PortAudio really was re-enumerated. A refresh is
    declined (returns False) while any capture stream is open, because
    terminating under a live stream wedges the host API. Pass
    ``force=True`` only when the caller has just closed its own stream.
    """
    with _DEV_LOCK:
        if _LIVE_STREAMS and not force:
            return False
        for stream in list(_LIVE_STREAMS):
            close_input_stream(stream)
        try:
            sd._terminate()
        except Exception:
            # Not initialised (or a stream still held by PortAudio): bring
            # it back up so opens keep working, but report no re-scan.
            try:
                sd._initialize()
            except Exception:
                pass
            return False
        try:
            sd._initialize()
        except Exception:
            pass
        return True


def _name_key(name) -> str:
    """Normalise an endpoint name for matching across an unplug/replug.

    Windows exposes the same USB mic as "USB Audio Device", "2- USB Audio
    Device" or "Microphone (USB Audio Devi" (MME truncates long names), and
    a replugged headset often gains a duplicate-instance number.
    """
    text = str(name or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"^\d+\s*-\s*", "", text)       # "2- USB Mic"
    text = re.sub(r"\s*\(\d+\)\s*$", "", text)    # "USB Mic (2)"
    text = re.sub(r"[^0-9a-z]+", " ", text).strip()
    return text


def _match_device_name(name, devices=None):
    """Input index whose name matches ``name``, or None.

    Prefers an exact normalised match, then a truncated/aliased twin. A
    replugged endpoint keeps its name but gets a new index, so the name is
    the only stable handle PortAudio gives us.
    """
    key = _name_key(name)
    if not key:
        return None
    try:
        devs = sd.query_devices() if devices is None else devices
    except Exception:
        return None
    loose = None
    try:
        entries = list(enumerate(devs))
    except Exception:
        return None
    for idx, dev in entries:
        try:
            if int(dev.get("max_input_channels") or 0) <= 0:
                continue
            cand = _name_key(dev.get("name"))
        except Exception:
            continue
        if not cand:
            continue
        if cand == key:
            return int(idx)
        if loose is None and len(key) >= 12 and (
                cand.startswith(key) or key.startswith(cand)
                or key in cand or cand in key):
            loose = int(idx)
    return loose


def match_listed_device(name, devices):
    """Input index of a listed endpoint matching ``name``, or None.

    Same matching rules as :func:`_match_device_name`, applied to the
    filtered list (``list_input_devices`` output) the UI shows. The UI has
    no raw PortAudio table to search, and matching every picker rebuild by
    hand is how a replugged mic ends up bound to the wrong endpoint.
    """
    key = _name_key(name)
    if not key:
        return None
    loose = None
    for dev in devices or []:
        try:
            cand = _name_key(dev.get("name"))
            idx = int(dev.get("index", -1))
        except Exception:
            continue
        if not cand or idx < 0:
            continue
        if cand == key:
            return idx
        if loose is None and len(key) >= 12 and (
                cand.startswith(key) or key.startswith(cand)
                or key in cand or cand in key):
            loose = idx
    return loose


def _input_indexes():
    """Every known capture index, best (DirectSound/MME) first."""
    out = []
    try:
        for d in list_input_devices():
            try:
                idx = int(d.get("index", -1))
            except Exception:
                continue
            if idx >= 0 and idx not in out:
                out.append(idx)
    except Exception:
        pass
    if not out:
        try:
            for idx, dev in enumerate(sd.query_devices()):
                try:
                    if int(dev.get("max_input_channels") or 0) > 0:
                        out.append(int(idx))
                except Exception:
                    continue
        except Exception:
            pass
    return out


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.device_name = None
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        # Serialises open/close against stop()/reopen(): a reopen finishing
        # after stop() must not publish a stream nobody will ever close.
        self._open_lock = threading.RLock()
        self._on_level: Optional[Callable[[float], None]] = None
        self._last_block_t = time.monotonic()
        self._zero_since: Optional[float] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def has_stream(self) -> bool:
        """True while a capture stream is open (even if it is dead silent)."""
        try:
            return self._stream is not None
        except Exception:
            return False

    def set_level_callback(self, callback: Optional[Callable[[float], None]]):
        self._on_level = callback

    def set_device(self, device, name=None):
        """Pick an input device index (None, -1 or "" = system default).

        ``name`` is remembered so the endpoint can be found again by name
        after a replug, when Windows hands out a different index.
        """
        try:
            if device in (None, "", "-1", -1):
                self.device = None
            else:
                self.device = int(device)
        except Exception:
            self.device = None
        try:
            self.device_name = str(name).strip() if name else None
        except Exception:
            self.device_name = None

    def _resolved_device(self):
        """The chosen index, or None (system default) when the saved device
        is gone, disabled, or not a capture endpoint at all. Prefers a name
        match, because a replugged USB mic comes back on a new index."""
        name = (self.device_name or "").strip()
        if name:
            idx = _match_device_name(name)
            if idx is not None:
                self.device = int(idx)
                return int(idx)
        if self.device is None:
            return None
        try:
            info = sd.query_devices(int(self.device))
            if int(info.get("max_input_channels") or 0) > 0:
                return int(self.device)
        except Exception:
            pass
        return None

    def _candidates(self):
        """Indexes to try, best first: the saved endpoint resolved by name,
        the saved index, then the system default."""
        out = []
        idx = _match_device_name(self.device_name)
        for cand in (idx, self.device, None):
            if cand not in out:
                out.append(cand)
        return out

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            now = time.monotonic()
            self._last_block_t = now
            try:
                if indata.size and not indata.any():
                    if self._zero_since is None:
                        self._zero_since = now
                else:
                    self._zero_since = None
            except Exception:
                pass
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

    def _open_stream(self, device):
        return open_input_stream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._audio_callback,
            blocksize=1024,
            device=device,
        )

    def _open_with_recovery(self):
        """Open the stream, surviving a mic that was unplugged or replugged
        while the app was running.

        A removed endpoint leaves a dead index behind, so the first open
        fails with -9999. PortAudio's cached list does not contain the new
        index either, and re-enumerating is only legal with no stream open,
        so: try the saved endpoint, re-scan once, resolve by name again,
        then fall back to the system default and finally to any capture
        endpoint -- holding F2 always records something rather than
        reporting "Mic error" until the app is restarted.
        """
        with _DEV_LOCK:
            first = None
            for cand in self._candidates():
                try:
                    return self._open_stream(cand)
                except Exception as e:
                    first = first or e
            reenumerated = False
            try:
                reenumerated = bool(refresh_devices())
            except Exception:
                reenumerated = False
            if reenumerated:
                for cand in self._candidates():
                    try:
                        return self._open_stream(cand)
                    except Exception as e:
                        first = first or e
            tried = self._candidates()
            for cand in _input_indexes():
                if cand in tried:
                    continue
                try:
                    return self._open_stream(cand)
                except Exception as e:
                    first = first or e
            raise first or RuntimeError("no microphone available")

    def start(self):
        with self._open_lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True
            self._last_block_t = time.monotonic()
            self._zero_since = None
            try:
                self._stream = self._open_with_recovery()
            except Exception:
                self._recording = False
                self._stream = None
                raise

    def stalled_for(self) -> float:
        """Seconds since the last block arrived (0 on a fresh stream)."""
        try:
            return max(0.0, time.monotonic() - float(self._last_block_t))
        except Exception:
            return 0.0

    def silent_for(self) -> float:
        """Seconds of continuous digital silence. A live mic always carries
        some noise floor, so exact zeros mean the endpoint was suspended."""
        try:
            if self._zero_since is None:
                return 0.0
            return max(0.0, time.monotonic() - float(self._zero_since))
        except Exception:
            return 0.0

    def reopen(self) -> bool:
        """Drop and re-open the capture stream without losing audio already
        collected. Used when the endpoint disappears, sleeps, or stops
        delivering. Safe to call while stop() is running: the two are
        serialised, and a stream opened after stop() is closed again."""
        with self._open_lock:
            if not self._recording:
                return False
            old, self._stream = self._stream, None
            close_input_stream(old)
            try:
                self._last_block_t = time.monotonic()
                self._zero_since = None
                stream = self._open_with_recovery()
            except Exception:
                self._stream = None
                return False
            if not self._recording:
                close_input_stream(stream)
                return False
            self._stream = stream
            return True

    def stop(self) -> Optional[np.ndarray]:
        with self._open_lock:
            if not self._recording:
                return None
            self._recording = False
            self._zero_since = None
            stream, self._stream = self._stream, None
            close_input_stream(stream)
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
    def list_input_devices(refresh: bool = False):
        """Capture endpoints usable at 16 kHz (see module-level helper)."""
        return list_input_devices(refresh=refresh)

    @staticmethod
    def get_default_input():
        try:
            dev = sd.query_devices(kind="input")
            return dev["name"]
        except Exception:
            return None


def list_input_devices(refresh: bool = False):
    """Capture endpoints the app can actually open at 16 kHz, best first.

    Only DirectSound and MME are listed: both resample inside Windows, so
    they accept the 16 kHz the engines expect. WASAPI refuses anything but
    the endpoint's native rate and WDM-KS fails on most drivers. Entries
    that just alias "whatever Windows defaults to" are skipped, and the
    same endpoint exposed by both host APIs is listed once (DirectSound
    first, because its names are not truncated).

    Pass ``refresh=True`` to ask for a re-enumeration first, which is what
    makes a replugged USB mic visible without an app restart. The re-scan
    is skipped while a capture stream is open (PortAudio cannot be
    restarted then); the cached list is returned instead.
    """
    if refresh:
        try:
            refresh_devices()
        except Exception:
            pass
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
