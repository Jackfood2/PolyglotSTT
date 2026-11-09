"""mic_check.py - find out whether a microphone actually delivers audio.

PortAudio will happily open a dead endpoint: the stream opens, costs no
error, and then delivers zero blocks or 1.2 seconds of digital zeros. The
app then shows a flat meter and "(no audio captured)". This script opens
every capture endpoint listed under every host API, reads a second of
audio with a hard timeout, and says which ones really carry sound.

Run it from the project folder:

    venv\\Scripts\\python.exe mic_check.py            (all endpoints)
    venv\\Scripts\\python.exe mic_check.py 7          (one index)
    venv\\Scripts\\python.exe mic_check.py --secs 3   (longer sample)

Sound input is measured, never saved.
"""
import json
import os
import sys
import threading

import numpy as np
import sounddevice as sd

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "moonshine_config.json")


def configured_mic():
    try:
        with open(CFG, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return int(cfg.get("input_device", -1)), str(
            cfg.get("input_device_name", "") or "")
    except Exception:
        return -1, ""


def capture(index, samplerate, channels, secs, timeout):
    """Read ``secs`` of audio in a worker thread; guillemets never hang."""
    out = {}

    def work():
        try:
            stream = sd.InputStream(device=index, samplerate=samplerate,
                                    channels=channels, dtype="int16",
                                    blocksize=0)
            try:
                stream.start()
                data, overflowed = stream.read(int(samplerate * secs))
            finally:
                for name in ("stop", "close"):
                    try:
                        getattr(stream, name)()
                    except Exception:
                        pass
            out["data"] = data
            out["overflowed"] = bool(overflowed)
        except Exception as exc:            # noqa: BLE001 - report, never crash
            out["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        out["timeout"] = True
    return out


def verdict(result, samplerate, secs):
    if result.get("timeout"):
        return "NO DATA", "stream opened, read blocked for 8 s - endpoint dead"
    if "error" in result:
        return "FAILED", str(result["error"])[:70]
    data = result.get("data")
    if data is None or not len(data):
        return "NO DATA", "stream opened, delivered zero blocks"
    x = np.asarray(data, dtype=np.float32) / 32768.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    peak_db = 20 * np.log10(peak) if peak > 0 else -999.0
    rms_db = 20 * np.log10(rms) if rms > 0 else -999.0
    detail = (f"peak {peak_db:6.1f} dBFS  rms {rms_db:6.1f} dBFS  "
              f"{x.size} frames in {secs:.1f} s")
    if peak <= 0.0:
        return "SILENT", detail + "  (all samples exactly zero)"
    if peak_db < -70.0:
        return "TOO QUIET", detail + "  (barely above the noise floor)"
    return "OK", detail


def main(argv):
    secs = 1.2
    timeout = 8.0
    wanted = []
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--secs" and args:
            secs = float(args.pop(0))
        elif arg.startswith("--secs="):
            secs = float(arg.split("=", 1)[1])
        else:
            wanted.append(int(arg))

    cfg_index, cfg_name = configured_mic()
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:            # noqa: BLE001
        print(f"PortAudio unavailable: {exc}")
        return 1

    rows = []
    for index, dev in enumerate(devices):
        try:
            if int(dev.get("max_input_channels") or 0) <= 0:
                continue
        except Exception:
            continue
        if wanted and index not in wanted:
            continue
        api_name = str(hostapis[int(dev["hostapi"])]["name"])
        name = str(dev.get("name") or "")
        native = int(round(float(dev.get("default_samplerate") or 44100)))
        channels = 1
        result = capture(index, native, channels, secs, timeout)
        state, detail = verdict(result, native, secs)
        rows.append((index, api_name, name, native, state, detail))

    print(f"Sampling {secs:.1f} s per endpoint "
          f"(app is configured for index {cfg_index}: {cfg_name!r})\n")
    width = max([len(r[2]) for r in rows] + [10])
    for index, api_name, name, native, state, detail in rows:
        mark = "*" if index == cfg_index else " "
        print(f"{mark}[{index:>2}] {state:<9} {api_name:<20} "
              f"{name[:width]:<{width}} {native:>6} Hz")
        print(f"        {detail}")

    print("\nLegend: OK = real audio arrived, TOO QUIET = gain/level problem,")
    print("        SILENT = digital zeros, NO DATA = endpoint never streams.")
    print("Only OK / TOO QUIET rows are usable. SILENT and NO DATA are a")
    print("Windows/driver/USB problem, not an app problem: try another port,")
    print("replug, set the device as the Windows default, or another mic.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
