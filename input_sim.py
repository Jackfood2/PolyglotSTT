import ctypes
import ctypes.wintypes as wintypes
import logging
import math
import os
import threading
import time
from contextlib import contextmanager


if os.name != "nt":
    raise OSError("input_sim is supported only on Windows")


CF_UNICODETEXT = 13

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

INPUT_KEYBOARD = 1

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_SPACE = 0x20
VK_V = 0x56
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5

SW_RESTORE = 9
GA_ROOT = 2
GMEM_MOVEABLE = 0x0002
GMEM_ZEROINIT = 0x0040
MAPVK_VK_TO_VSC = 0

ULONG_PTR = ctypes.c_size_t

HOTKEY_VK_MAP = {
    f"F{number}": 0x6F + number
    for number in range(1, 25)
}

_MODIFIER_KEYS = (
    VK_LCONTROL,
    VK_RCONTROL,
    VK_LSHIFT,
    VK_RSHIFT,
    VK_LMENU,
    VK_RMENU,
    VK_LWIN,
    VK_RWIN,
)

_EXTENDED_KEYS = {
    VK_RCONTROL,
    VK_RMENU,
    VK_LWIN,
    VK_RWIN,
}

_LOG = logging.getLogger(__name__)
_INPUT_LOCK = threading.RLock()
_CLIPBOARD_LOCK = threading.RLock()


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _bind(library, name, argtypes, restype):
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = restype
    return function


_bind(
    user32,
    "SendInput",
    [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int],
    wintypes.UINT,
)
_bind(user32, "GetForegroundWindow", [], wintypes.HWND)
_bind(
    user32,
    "SetForegroundWindow",
    [wintypes.HWND],
    wintypes.BOOL,
)
_bind(
    user32,
    "GetWindowThreadProcessId",
    [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)],
    wintypes.DWORD,
)
_bind(
    user32,
    "AttachThreadInput",
    [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL],
    wintypes.BOOL,
)
_bind(
    user32,
    "GetGUIThreadInfo",
    [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)],
    wintypes.BOOL,
)
_bind(user32, "GetFocus", [], wintypes.HWND)
_bind(user32, "SetFocus", [wintypes.HWND], wintypes.HWND)
_bind(user32, "IsWindow", [wintypes.HWND], wintypes.BOOL)
_bind(user32, "IsIconic", [wintypes.HWND], wintypes.BOOL)
_bind(
    user32,
    "IsChild",
    [wintypes.HWND, wintypes.HWND],
    wintypes.BOOL,
)
_bind(
    user32,
    "ShowWindow",
    [wintypes.HWND, ctypes.c_int],
    wintypes.BOOL,
)
_bind(
    user32,
    "GetAsyncKeyState",
    [ctypes.c_int],
    ctypes.c_short,
)
_bind(
    user32,
    "GetAncestor",
    [wintypes.HWND, wintypes.UINT],
    wintypes.HWND,
)
_bind(
    user32,
    "MapVirtualKeyW",
    [wintypes.UINT, wintypes.UINT],
    wintypes.UINT,
)
_bind(
    user32,
    "GetWindowTextLengthW",
    [wintypes.HWND],
    ctypes.c_int,
)
_bind(
    user32,
    "GetWindowTextW",
    [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int],
    ctypes.c_int,
)
_bind(user32, "OpenClipboard", [wintypes.HWND], wintypes.BOOL)
_bind(user32, "CloseClipboard", [], wintypes.BOOL)
_bind(user32, "EmptyClipboard", [], wintypes.BOOL)
_bind(
    user32,
    "IsClipboardFormatAvailable",
    [wintypes.UINT],
    wintypes.BOOL,
)
_bind(
    user32,
    "GetClipboardData",
    [wintypes.UINT],
    wintypes.HANDLE,
)
_bind(
    user32,
    "SetClipboardData",
    [wintypes.UINT, wintypes.HANDLE],
    wintypes.HANDLE,
)
_bind(
    user32,
    "CreateWindowExW",
    [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ],
    wintypes.HWND,
)
_bind(user32, "DestroyWindow", [wintypes.HWND], wintypes.BOOL)
_bind(kernel32, "GetCurrentThreadId", [], wintypes.DWORD)
_bind(
    kernel32,
    "GetModuleHandleW",
    [wintypes.LPCWSTR],
    wintypes.HMODULE,
)
_bind(
    kernel32,
    "GlobalAlloc",
    [wintypes.UINT, ctypes.c_size_t],
    wintypes.HGLOBAL,
)
_bind(
    kernel32,
    "GlobalFree",
    [wintypes.HGLOBAL],
    wintypes.HGLOBAL,
)
_bind(
    kernel32,
    "GlobalLock",
    [wintypes.HGLOBAL],
    ctypes.c_void_p,
)
_bind(
    kernel32,
    "GlobalUnlock",
    [wintypes.HGLOBAL],
    wintypes.BOOL,
)
_bind(
    kernel32,
    "GlobalSize",
    [wintypes.HGLOBAL],
    ctypes.c_size_t,
)


SIZEOF_INPUT = ctypes.sizeof(INPUT)
_EXPECTED_INPUT_SIZE = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28

if SIZEOF_INPUT != _EXPECTED_INPUT_SIZE:
    raise RuntimeError(f"Unexpected Windows INPUT size: {SIZEOF_INPUT}")


def _debug(message):
    _LOG.debug("%s", message)


def _log(message, log_func=None):
    _LOG.debug("%s", message)

    if log_func is not None:
        try:
            log_func(message)
        except Exception:
            _LOG.exception("Input logging callback failed")


def _nonnegative_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    return max(0.0, result) if math.isfinite(result) else default


def _valid_child(hwnd):
    return bool(hwnd and user32.IsWindow(hwnd))


def _root_window(hwnd):
    if not _valid_child(hwnd):
        return None

    return user32.GetAncestor(hwnd, GA_ROOT) or hwnd


def _belongs_to_window(top_hwnd, child_hwnd):
    root = _root_window(top_hwnd)

    return bool(
        root
        and _valid_child(child_hwnd)
        and (
            child_hwnd == root
            or user32.IsChild(root, child_hwnd)
        )
    )


def _foreground_matches(hwnd):
    target_root = _root_window(hwnd)
    actual_root = _root_window(get_foreground_window())
    return bool(target_root and actual_root == target_root)


@contextmanager
def _attached_threads(*thread_ids):
    current_thread = kernel32.GetCurrentThreadId()
    attached = []

    try:
        for thread_id in dict.fromkeys(thread_ids):
            if not thread_id or thread_id == current_thread:
                continue

            if user32.AttachThreadInput(
                current_thread,
                thread_id,
                True,
            ):
                attached.append(thread_id)

        yield
    finally:
        for thread_id in reversed(attached):
            user32.AttachThreadInput(
                current_thread,
                thread_id,
                False,
            )


def _keyboard_input(vk=0, scan=0, flags=0):
    event = INPUT()
    event.type = INPUT_KEYBOARD
    event.union.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
    return event


def _virtual_key_event(vk, key_up=False):
    scan = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
    flags = KEYEVENTF_KEYUP if key_up else 0

    if vk in _EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY

    return _keyboard_input(vk, scan, flags)


def _send_input(arr, count):
    ctypes.set_last_error(0)
    return int(user32.SendInput(count, arr, SIZEOF_INPUT))


def _send_events(events, log_func=None):
    if not events:
        return True

    array = (INPUT * len(events))(*events)
    sent = _send_input(array, len(events))

    if sent == len(events):
        return True

    error = ctypes.get_last_error()
    _log(
        f"SendInput incomplete: {sent}/{len(events)}; "
        f"Windows error: {error}",
        log_func,
    )

    pending = {}

    for event in events[:sent]:
        keyboard = event.union.ki
        identity = (
            keyboard.wVk,
            keyboard.wScan,
            keyboard.dwFlags & ~KEYEVENTF_KEYUP,
        )

        if keyboard.dwFlags & KEYEVENTF_KEYUP:
            pending.pop(identity, None)
        else:
            pending[identity] = keyboard

    releases = [
        _keyboard_input(
            keyboard.wVk,
            keyboard.wScan,
            keyboard.dwFlags | KEYEVENTF_KEYUP,
        )
        for keyboard in reversed(list(pending.values()))
    ]

    if releases:
        release_array = (INPUT * len(releases))(*releases)
        released = _send_input(release_array, len(releases))

        if released != len(releases):
            _log("Could not release all partially injected keys", log_func)

    return False


def get_foreground_window():
    return user32.GetForegroundWindow()


def get_window_title(hwnd):
    if not _valid_child(hwnd):
        return "<invalid>"

    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(max(length + 1, 256))
    copied = user32.GetWindowTextW(hwnd, buffer, len(buffer))

    return buffer.value if copied else "<no title>"


def force_release_modifier_keys():
    with _INPUT_LOCK:
        events = [
            _virtual_key_event(vk, key_up=True)
            for vk in _MODIFIER_KEYS
            if user32.GetAsyncKeyState(vk) & 0x8000
        ]

        if not _send_events(events):
            return False

        if events:
            time.sleep(0.02)

        return True


def wait_for_hotkey_release(hotkey_name, timeout=3.0):
    name = str(hotkey_name or "").strip().upper()
    vk = HOTKEY_VK_MAP.get(name)

    if vk is None:
        return True

    deadline = time.monotonic() + _nonnegative_float(timeout, 3.0)

    while user32.GetAsyncKeyState(vk) & 0x8000:
        remaining = deadline - time.monotonic()

        if remaining <= 0:
            return False

        time.sleep(min(0.02, remaining))

    return True


def restore_and_focus_window(hwnd):
    root = _root_window(hwnd)

    if not root:
        return False

    if user32.IsIconic(root):
        user32.ShowWindow(root, SW_RESTORE)

    if _foreground_matches(root):
        return True

    foreground = get_foreground_window()
    target_thread = user32.GetWindowThreadProcessId(root, None)
    foreground_thread = (
        user32.GetWindowThreadProcessId(foreground, None)
        if foreground
        else 0
    )

    with _attached_threads(foreground_thread, target_thread):
        user32.SetForegroundWindow(root)

    deadline = time.monotonic() + 0.25

    while True:
        if _foreground_matches(root):
            return True

        if time.monotonic() >= deadline:
            return False

        time.sleep(0.01)


def get_focused_child(top_hwnd):
    if not _valid_child(top_hwnd):
        return None

    thread_id = user32.GetWindowThreadProcessId(top_hwnd, None)

    if not thread_id:
        return None

    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(info)

    if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
        return None

    if _belongs_to_window(top_hwnd, info.hwndFocus):
        return info.hwndFocus

    return None


def restore_focus_to_child(top_hwnd, child_hwnd):
    if not _belongs_to_window(top_hwnd, child_hwnd):
        return False

    if not _foreground_matches(top_hwnd):
        return False

    thread_id = user32.GetWindowThreadProcessId(child_hwnd, None)

    with _attached_threads(thread_id):
        if not _foreground_matches(top_hwnd):
            return False

        user32.SetFocus(child_hwnd)
        focused = user32.GetFocus() == child_hwnd

    return focused and _foreground_matches(top_hwnd)


def _focus_matches(top_hwnd, child_hwnd=None):
    if not _foreground_matches(top_hwnd):
        return False

    if child_hwnd is None:
        return True

    return (
        _belongs_to_window(top_hwnd, child_hwnd)
        and get_focused_child(top_hwnd) == child_hwnd
    )


def verify_focus_ready(
    target_top_hwnd,
    target_child_hwnd,
    max_attempts=4,
):
    if not _valid_child(target_top_hwnd):
        return False

    if target_child_hwnd and not _belongs_to_window(
        target_top_hwnd,
        target_child_hwnd,
    ):
        return False

    try:
        attempts = max(1, int(max_attempts))
    except (TypeError, ValueError, OverflowError):
        attempts = 4

    for _ in range(attempts):
        if _focus_matches(target_top_hwnd, target_child_hwnd):
            return True

        if restore_and_focus_window(target_top_hwnd):
            if target_child_hwnd:
                restore_focus_to_child(
                    target_top_hwnd,
                    target_child_hwnd,
                )

        if _focus_matches(target_top_hwnd, target_child_hwnd):
            return True

        time.sleep(0.03)

    return False


def _open_clipboard(owner=None, timeout=0.5):
    deadline = time.monotonic() + timeout

    while True:
        if user32.OpenClipboard(owner):
            return True

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            return False

        time.sleep(min(0.025, remaining))


def _create_clipboard_owner():
    instance = kernel32.GetModuleHandleW(None)

    return user32.CreateWindowExW(
        0,
        "STATIC",
        "input_sim clipboard",
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        instance,
        None,
    )


def get_clipboard_text():
    with _CLIPBOARD_LOCK:
        if not _open_clipboard():
            return None

        try:
            if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
                return None

            handle = user32.GetClipboardData(CF_UNICODETEXT)

            if not handle:
                return None

            size = kernel32.GlobalSize(handle)

            if size < 2:
                return None

            pointer = kernel32.GlobalLock(handle)

            if not pointer:
                return None

            try:
                raw = ctypes.string_at(pointer, size - size % 2)
            finally:
                kernel32.GlobalUnlock(handle)

            for index in range(0, len(raw), 2):
                if raw[index:index + 2] == b"\x00\x00":
                    raw = raw[:index]
                    break

            return raw.decode("utf-16-le", errors="replace")

        finally:
            user32.CloseClipboard()


def set_clipboard_text(text):
    text = "" if text is None else str(text)

    if "\x00" in text:
        _LOG.error("Clipboard text cannot contain embedded null characters")
        return False

    try:
        encoded = text.encode("utf-16-le") + b"\x00\x00"
    except UnicodeEncodeError:
        _LOG.exception("Clipboard text contains invalid Unicode")
        return False

    with _CLIPBOARD_LOCK:
        handle = kernel32.GlobalAlloc(
            GMEM_MOVEABLE | GMEM_ZEROINIT,
            len(encoded),
        )

        if not handle:
            return False

        owner = None
        opened = False

        try:
            pointer = kernel32.GlobalLock(handle)

            if not pointer:
                return False

            try:
                ctypes.memmove(pointer, encoded, len(encoded))
            finally:
                kernel32.GlobalUnlock(handle)

            owner = _create_clipboard_owner()

            if not owner:
                return False

            opened = _open_clipboard(owner)

            if not opened:
                return False

            if not user32.EmptyClipboard():
                return False

            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                return False

            handle = None
            return True

        finally:
            if opened:
                user32.CloseClipboard()

            if owner:
                user32.DestroyWindow(owner)

            if handle:
                kernel32.GlobalFree(handle)


def send_ctrl_v(log_func=None):
    with _INPUT_LOCK:
        if not force_release_modifier_keys():
            _log("Could not release modifier keys before pasting", log_func)
            return False

        events = [
            _virtual_key_event(VK_CONTROL),
            _virtual_key_event(VK_V),
            _virtual_key_event(VK_V, key_up=True),
            _virtual_key_event(VK_CONTROL, key_up=True),
        ]

        return _send_events(events, log_func)


def _character_events(character):
    if character in {"\n", "\t"}:
        vk = VK_RETURN if character == "\n" else VK_TAB
        return [
            _virtual_key_event(vk),
            _virtual_key_event(vk, key_up=True),
        ]

    encoded = character.encode("utf-16-le")
    events = []

    for offset in range(0, len(encoded), 2):
        unit = int.from_bytes(encoded[offset:offset + 2], "little")
        events.extend(
            [
                _keyboard_input(0, unit, KEYEVENTF_UNICODE),
                _keyboard_input(
                    0,
                    unit,
                    KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                ),
            ]
        )

    return events


def _send_unicode(
    text,
    wake=False,
    delay_ms=0,
    batch_size=64,
    log_func=None,
    focus_check=None,
):
    if not text:
        return True

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    try:
        text.encode("utf-16-le")
    except UnicodeEncodeError:
        _log("Text contains invalid Unicode", log_func)
        return False

    if "\x00" in text:
        _log("Text contains an embedded null character", log_func)
        return False

    delay = _nonnegative_float(delay_ms) / 1000.0

    try:
        batch_size = max(4, min(int(batch_size), 128))
    except (TypeError, ValueError, OverflowError):
        batch_size = 64

    def flush(events):
        if focus_check is not None and not focus_check():
            _log("Target focus changed; insertion stopped", log_func)
            return False

        return _send_events(events, log_func)

    if focus_check is not None and not focus_check():
        return False

    if not force_release_modifier_keys():
        return False

    if wake:
        events = [
            _virtual_key_event(VK_SPACE),
            _virtual_key_event(VK_SPACE, key_up=True),
            _virtual_key_event(VK_BACK),
            _virtual_key_event(VK_BACK, key_up=True),
        ]

        if not flush(events):
            return False

        time.sleep(0.05)

    events = []

    for character in text:
        character_events = _character_events(character)

        if events and len(events) + len(character_events) > batch_size:
            if not flush(events):
                return False

            events = []
            time.sleep(delay if delay > 0 else 0.005)

        events.extend(character_events)

    return flush(events) if events else True


def send_unicode_string(
    text,
    wake=False,
    delay_ms=0,
    batch_size=64,
    log_func=None,
):
    with _INPUT_LOCK:
        return _send_unicode(
            text,
            wake=wake,
            delay_ms=delay_ms,
            batch_size=batch_size,
            log_func=log_func,
        )


def prepare_insertion_focus(
    target_top_hwnd,
    target_child_hwnd,
    hotkey_name,
    our_root_hwnd=0,
    allow_self=False,
    log_func=None,
):
    with _INPUT_LOCK:
        if not wait_for_hotkey_release(hotkey_name, timeout=3.0):
            _log("Hotkey release timed out", log_func)
            return None, None, False

        own_root = _root_window(our_root_hwnd) or our_root_hwnd
        target_root = _root_window(target_top_hwnd)

        if target_root and not allow_self and own_root == target_root:
            _log("Insertion into this application's window is disabled", log_func)
            return None, None, False

        if not target_root:
            target_root = _root_window(get_foreground_window())

            if not target_root:
                _log("No valid foreground window", log_func)
                return None, None, False

            if not allow_self and own_root == target_root:
                _log("No external target window", log_func)
                return None, None, False

            target_child_hwnd = get_focused_child(target_root)

        if target_child_hwnd and not _belongs_to_window(
            target_root,
            target_child_hwnd,
        ):
            _log("The requested child window is no longer valid", log_func)
            return None, None, False

        if not force_release_modifier_keys():
            _log("Could not release modifier keys", log_func)
            return None, None, False

        if not verify_focus_ready(target_root, target_child_hwnd):
            _log("Could not verify the target window's focus", log_func)
            return None, None, False

        if not target_child_hwnd:
            target_child_hwnd = get_focused_child(target_root)

        if not _focus_matches(target_root, target_child_hwnd):
            _log("Target focus changed during preparation", log_func)
            return None, None, False

        return target_root, target_child_hwnd, True


def insert_text(text, target_top_hwnd, target_child_hwnd, hotkey_name,
                our_root_hwnd=0, method="clipboard", allow_self=False,
                wake=False, delay_ms=0, log_func=None):
    if not text:
        return True

    method = str(method).lower()

    if method not in ("unicode", "clipboard"):
        method = "clipboard"

    top, child, ok = prepare_insertion_focus(
        target_top_hwnd,
        target_child_hwnd,
        hotkey_name,
        our_root_hwnd=our_root_hwnd,
        allow_self=allow_self,
        log_func=log_func,
    )

    if not ok:
        set_clipboard_text(text)

        if log_func:
            log_func("No external target - copied to clipboard")

        return False

    if method == "clipboard":
        if set_clipboard_text(text):
            force_release_modifier_keys()
            paste_ok = send_ctrl_v(log_func=log_func)
            time.sleep(1.0)
            return paste_ok

    return send_unicode_string(
        text,
        wake=wake,
        delay_ms=delay_ms,
        log_func=log_func,
    )


def copy_to_clipboard(text: str) -> bool:
    return set_clipboard_text(text)


def paste_text(text: str) -> bool:
    if not text or not text.strip():
        return False

    fg = get_foreground_window()
    child = get_focused_child(fg)
    return insert_text(text, fg, child, "F2", our_root_hwnd=0,
                       method="clipboard")


def get_clipboard_text_simple() -> str:
    t = get_clipboard_text()
    return t or ""