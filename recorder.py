# recorder.py
import math
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


def _hostapi_rank(index) -> int:
    """Order endpoints the way the picker lists them (see
    _HOSTAPI_PREFERENCE): MME first, then WASAPI, then DirectSound.

    One physical microphone is exposed once per host API under the same
    name, so a lookup by name has to choose one twin. MME is the twin that
    reliably carries 16 kHz audio; DirectSound is last because on some
    machines it opens the endpoint and then delivers near-silence, which
    is what makes a working USB mic look "too quiet to catch the sound".
    """
    try:
        dev = sd.query_devices(int(index))
        api = sd.query_hostapis(int(dev.get("hostapi")))
        name = str(api.get("name") or "")
    except Exception:
        return 9
    return _HOSTAPI_PREFERENCE.get(name, 9)


#: Host APIs that can actually carry 16 kHz capture, best first. MME
#: resamples inside Windows so it accepts the rate the engines want;
#: WASAPI refuses anything but the endpoint's native rate (-9997) and is
#: only useful as a second try; DirectSound is last because on some
#: machines it opens the endpoint and then hands back digital zeros.
_HOSTAPI_PREFERENCE = {"MME": 0, "Windows WASAPI": 1,
                       "Windows DirectSound": 2}


def _hostapi_name(index) -> str:
    """Host API name behind a device index, or "" when it cannot be read."""
    try:
        dev = sd.query_devices(int(index))
        return str(sd.query_hostapis(int(dev.get("hostapi"))).get("name") or "")
    except Exception:
        return ""


def _indexes_for_name(name):
    """Every capture index exposing one endpoint name, best host API first.

    Windows lists one physical mic once per host API (DirectSound, MME,
    WASAPI) and on a given machine only some of those twins actually carry
    audio: DirectSound here hands back digital zeros while MME and WASAPI
    stream the same microphone correctly. Sorting twins by host API lets a
    silent twin be abandoned for a working one.
    """
    key = _name_key(name)
    if not key:
        return []
    try:
        devs = sd.query_devices()
    except Exception:
        return []
    out = []
    for idx, dev in enumerate(devs):
        try:
            if int(dev.get("max_input_channels") or 0) <= 0:
                continue
        except Exception:
            continue
        cand = _name_key(dev.get("name"))
        if not cand:
            continue
        # MME truncates long names, so the twin of "Microphone Array (3-
        # Intel(R) Smart Sound Technology ...)" reads "Microphone Array (3-
        # Intel(R) Sma". A truncated twin is still the same microphone.
        if cand == key or (
                len(key) >= 12 and len(cand) >= 12 and
                (cand.startswith(key) or key.startswith(cand))):
            out.append(int(idx))
    out.sort(key=lambda i: _HOSTAPI_PREFERENCE.get(_hostapi_name(i), 9))
    return out


def _match_device_name(name, devices=None):
    """Input index whose name matches ``name``, or None.

    Prefers an exact normalised match, then a truncated/aliased twin. A
    replugged endpoint keeps its name but gets a new index, so the name is
    the only stable handle PortAudio gives us. When one name exists on
    several host APIs, the picker's preference decides (see _hostapi_rank).
    """
    key = _name_key(name)
    if not key:
        return None
    try:
        devs = sd.query_devices() if devices is None else devices
    except Exception:
        return None
    best = None
    best_rank = None
    loose = None
    try:
        entries = list(enumerate(devs))
    except Exception:
        return None
    for idx, dev in entries:
        try:
            if int(dev.get("max_input_channels", 1) or 0) <= 0:
                continue
            cand = _name_key(dev.get("name"))
        except Exception:
            continue
        if not cand:
            continue
        if cand == key:
            rank = _hostapi_rank(idx)
            if best is None or best_rank is None or rank < best_rank:
                best, best_rank = int(idx), rank
            continue
        if loose is None and len(key) >= 12 and (
                cand.startswith(key) or key.startswith(cand)
                or key in cand or cand in key):
            loose = int(idx)
    if best is not None:
        return best
    return loose


def level_from_rms(rms) -> float:
    """RMS of 0..1 full-scale samples -> the 0..1 the meters draw.

    A linear scale hides quiet endpoints: a USB mic idling at -55 dBFS
    reports ~0.002 RMS, i.e. a flat line, so "working but quiet" looks
    exactly like "dead microphone". Mapping -60 dBFS..0 dBFS onto 0..1
    keeps the top of the scale for loud speech while still moving the
    meter for a whisper.
    """
    try:
        value = float(rms)
    except Exception:
        return 0.0
    if not value or value <= 0.0:
        return 0.0
    try:
        db = 20.0 * math.log10(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, (db + 60.0) / 60.0))


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
        # Feed a block of live audio into a candidate probe. A dead host API
        # path (DirectSound on some machines) opens cleanly and then hands
        # back only exact zeros, so arrival alone does not prove it works.
        self._block_count = 0
        self._nonzero_seen = False

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
        is gone, disabled, or not a capture endpoint at all.

        The stored endpoint wins while it still names the stored microphone,
        so a replug that shuffles Windows' index does not silently move the
        recording to a different mic. Which *twin* of that microphone is
        opened is decided by host API, not by the stale saved index: a
        config saved before the picker started preferring MME can point at
        the DirectSound twin, which opens and then delivers near-silence.
        Only when the endpoint is gone -- or its index now names a different
        device -- is it followed by name, which is how a replugged USB mic
        is found again on its new index.
        """
        name = (self.device_name or "").strip()
        if self.device is not None:
            try:
                info = sd.query_devices(int(self.device))
                if int(info.get("max_input_channels") or 0) > 0:
                    if (not name
                            or _name_key(info.get("name")) == _name_key(name)):
                        twins = _indexes_for_name(name) if name else []
                        if twins:
                            self.device = int(twins[0])
                        return int(self.device)
            except Exception:
                pass
        if name:
            idx = _match_device_name(name)
            if idx is not None:
                self.device = int(idx)
                return int(idx)
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
            self._block_count += 1
            try:
                if indata.size and indata.any():
                    self._nonzero_seen = True
            except Exception:
                pass
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
                self._on_level(level_from_rms(rms))
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

    #: How long a freshly opened stream may take to hand over its first
    #: block before the endpoint is treated as dead. A live mic at 16 kHz
    #: delivers its first 64 ms block within a few hundred milliseconds.
    PROBE_SECONDS = 2.0

    def _first_block_within(self, stream, seconds=None):
        """True once the stream has really delivered audio. Opening a dead
        endpoint (disabled in Windows, unplugged, or a host API that cannot
        pull from this driver) succeeds and then yields nothing -- no blocks
        at all, or blocks of exact zeros -- so the first non-silent block is
        the only cheap proof that a candidate works."""
        wait = self.PROBE_SECONDS if seconds is None else float(seconds)
        mark = time.monotonic()
        self._last_block_t = mark
        self._nonzero_seen = False
        deadline = mark + max(0.05, wait)
        while time.monotonic() < deadline:
            try:
                if self._nonzero_seen and float(self._last_block_t) >= mark:
                    return True
            except Exception:
                return True
            time.sleep(0.02)
        return False

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

        A stream that opens but never delivers is not success: DirectSound
        returns digital zeros on some machines for a microphone that MME
        streams perfectly. Such a candidate is closed at once and its twin
        on another host API is tried before any other microphone.
        """
        with _DEV_LOCK:
            tried = set()
            silent_names = []
            errors = []

            def attempt(cand):
                if cand in tried:
                    return None
                tried.add(cand)
                try:
                    stream = self._open_stream(cand)
                except Exception as e:
                    errors.append(e)
                    return None
                if self._first_block_within(stream):
                    if cand is not None:
                        # Remember the host API that really works, so the
                        # next F2 press does not pay the probe again.
                        self.device = int(cand)
                    return stream
                try:
                    name = str(sd.query_devices(int(cand)).get("name") or "")
                except Exception:
                    name = ""
                if name:
                    silent_names.append(name)
                errors.append(RuntimeError(
                    "microphone opened but delivered no audio"))
                close_input_stream(stream)
                return None

            def attempt_all(cands):
                for cand in cands:
                    stream = attempt(cand)
                    if stream is not None:
                        return stream
                return None

            def twins():
                """The chosen mic and any mic that opened but stayed silent,
                on every host API Windows lists them under. A twin of the
                chosen endpoint beats the system default: the user asked
                for that microphone, not for whatever Windows likes best."""
                names = [n for n in silent_names]
                if self.device_name:
                    names.append(self.device_name)
                for name in names:
                    for cand in _indexes_for_name(name):
                        yield cand

            saved = [c for c in self._candidates() if c is not None]
            stream = attempt_all(saved)
            if stream is not None:
                return stream
            stream = attempt_all(twins())
            if stream is not None:
                return stream
            stream = attempt_all([None])
            if stream is not None:
                return stream
            reenumerated = False
            try:
                reenumerated = bool(refresh_devices())
            except Exception:
                reenumerated = False
            if reenumerated:
                stream = attempt_all(self._candidates())
                if stream is not None:
                    return stream
                stream = attempt_all(twins())
                if stream is not None:
                    return stream
            stream = attempt_all(_input_indexes())
            if stream is not None:
                return stream
            raise (errors[0] if errors
                   else RuntimeError("no microphone available"))

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
            return level_from_rms(rms)

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

    Only MME, WASAPI and DirectSound are listed. MME resamples inside
    Windows, so it accepts the 16 kHz the engines expect, and it is listed
    first: on machines where DirectSound opens an endpoint and then hands
    back near-silence, MME is the only twin that carries real audio. WASAPI
    is next (it only accepts the endpoint's native rate, so the recorder's
    own fallback usually has to skip it) and DirectSound last. WDM-KS fails
    on most drivers and is never listed. Entries that just alias "whatever
    Windows defaults to" are skipped, and one physical microphone shown by
    several host APIs is listed once, under the longest (untruncated) name
    while pointing at the best-ordered twin.

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
    order = ("MME", "Windows WASAPI", "Windows DirectSound")
    out = []
    by_key = {}
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
                key = _name_key(name)
                if not key:
                    continue
                entry = by_key.get(key)
                if entry is None:
                    # MME truncates long names, so its twin of "Microphone
                    # Array (3- Intel(R) Smart Sound ...)" reads "Microphone
                    # Array (3- Intel(R) Sma". Match a clipped twin to the
                    # entry it belongs to instead of listing the mic twice.
                    for other_key, other in by_key.items():
                        if len(other_key) >= 12 and len(key) >= 12 and (
                                other_key.startswith(key)
                                or key.startswith(other_key)):
                            entry = other
                            break
                if entry is None:
                    entry = {"index": int(idx), "name": name}
                    by_key[key] = entry
                    out.append(entry)
                elif len(name) > len(entry["name"]):
                    # Keep the readable full name; the index stays with the
                    # best host API so the stream opens on the twin that
                    # actually carries audio.
                    entry["name"] = name
    return out


def default_input_index():
    """PortAudio index of the Windows default capture device, or -1."""
    try:
        return int(sd.default.device[0])
    except Exception:
        return -1


#: Words Windows puts in the name of an external USB microphone
#: ("Microphone (USB2.0 Device)", "Microphone (USB PnP Sound Device)",
#: "Microphone (2- USB Audio Device)") but never in the laptop's own array.
_USB_HINTS = ("usb",)
#: Words that mark a built-in microphone, so a laptop whose internal array
#: is exposed through a USB-attached codec is not mistaken for a USB mic.
_BUILTIN_HINTS = ("array", "realtek", "smart sound", "digital microphones")


def is_usb_mic(name) -> bool:
    """True when an endpoint name looks like an external USB microphone."""
    low = str(name or "").lower()
    if not low or not any(h in low for h in _USB_HINTS):
        return False
    return not any(h in low for h in _BUILTIN_HINTS)


def auto_input_choice(devices, prefer_usb: bool = True):
    """Which listed endpoint should be used without being asked.

    A plugged-in and enabled USB microphone wins over the laptop's own
    microphone; with no USB endpoint listed the answer is ``None``, which
    means "system default" - the built-in mic Windows already uses. A USB
    mic that is unplugged or disabled in Windows is simply absent from the
    list, so it cannot win, which is exactly the behaviour wanted.

    Returns ``(index, name)`` or ``None``.
    """
    if not prefer_usb:
        return None
    best = None
    for d in devices or []:
        try:
            index = int(d.get("index", -1))
            name = str(d.get("name") or "").strip()
        except Exception:
            continue
        if index < 0 or not is_usb_mic(name):
            continue
        rank = _hostapi_rank(index)
        if best is None or rank < best[0]:
            best = (rank, index, name)
    if best is None:
        return None
    return best[1], best[2]


def choose_input(devices, current=-1, current_name="", auto=True):
    """Which listed endpoint the app should use right now -> (index, name).

    ``current`` and ``current_name`` are the saved choice; ``index`` -1
    means "system default" (the laptop's own microphone).

    Order of precedence:

    1. the saved microphone, followed by name -- a replug hands it a new
       index, and its old index may now belong to a different device;
    2. an external USB microphone that is plugged in and enabled (only
       when ``auto`` is on);
    3. the system default, so capture never stops while the saved mic is
       away. The saved name is kept so the same mic is chosen again the
       moment it returns.
    """
    listed = set()
    for d in devices or []:
        try:
            listed.add(int(d.get("index", -1)))
        except Exception:
            continue
    name = str(current_name or "").strip()
    try:
        want = int(current)
    except Exception:
        want = -1
    if name:
        hit = match_listed_device(name, devices)
        if hit is not None:
            want = int(hit)
        elif want >= 0 and want not in listed:
            want = -1
    elif want >= 0 and want not in listed:
        want = -1
    if auto:
        usb = auto_input_choice(devices)
        if usb is not None:
            return int(usb[0]), str(usb[1])
        if name and is_usb_mic(name):
            # The USB mic that was in use is gone: record from the built-in
            # microphone now, keeping the name so it returns on replug.
            return -1, name
    if want >= 0:
        return want, name or str(
            next((d.get("name") for d in devices
                  if int(d.get("index", -1)) == want), "") or "")
    return -1, name
