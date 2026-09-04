"""Input simulation - ported from SST Whisper test.py focus logic."""

import ctypes
import ctypes.wintypes as wintypes
import time

# --- Win32 constants (ported from test.py) ---
CF_UNICODETEXT = 13
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

VK_CONTROL = 0x11
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_V = 0x56
VK_SHIFT = 0x10
VK_MENU = 0x12
VK_RETURN = 0x0D
VK_TAB = 0x09

SW_RESTORE = 9
GA_ROOT = 2
GMEM_MOVEABLE = 0x0002
GMEM_ZEROINIT = 0x0040

# --- ctypes structures (ported) ---
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
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

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

user32.SendInput.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
user32.SendInput.restype = ctypes.c_uint
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.GetFocus.restype = wintypes.HWND
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.POINTER(ctypes.c_ulong)]
user32.keybd_event.restype = None
user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
user32.GetAncestor.restype = wintypes.HWND
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.OpenClipboard.restype = wintypes.BOOL
user32.CloseClipboard.restype = wintypes.BOOL
user32.EmptyClipboard.restype = wintypes.BOOL
user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.GetClipboardData.restype = wintypes.HANDLE
user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
user32.SetClipboardData.restype = wintypes.HANDLE

kernel32.GetCurrentThreadId.restype = wintypes.DWORD
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HANDLE
kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
kernel32.GlobalUnlock.restype = wintypes.BOOL

SIZEOF_INPUT = ctypes.sizeof(INPUT)

# Hotkey VK map (subset, ported)
HOTKEY_VK_MAP = {
    "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74,
    "F6": 0x75, "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79,
}

def _keyboard_input(vk=0, scan=0, flags=0):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.wScan = scan
    inp.union.ki.dwFlags = flags
    inp.union.ki.time = 0
    inp.union.ki.dwExtraInfo = None
    return inp

def _send_input(arr, count):
    ptr = ctypes.cast(arr, ctypes.c_void_p)
    return user32.SendInput(count, ptr, SIZEOF_INPUT)

def _debug(msg):
    print(f"[input_sim] {msg}")

# --- Focus helpers (direct port from test.py) ---
def get_foreground_window():
    return user32.GetForegroundWindow()

def get_window_title(hwnd):
    if not hwnd or not user32.IsWindow(hwnd):
        return "<invalid>"
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return "<no title>"
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value

def force_release_modifier_keys():
    for vk in [VK_CONTROL, VK_LCONTROL, VK_RCONTROL, VK_SHIFT, 0xA0, 0xA1, VK_MENU, 0xA4, 0xA5, 0x5B, 0x5C]:
        if user32.GetAsyncKeyState(vk) & 0x8000:
            user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, None)
    time.sleep(0.05)

def wait_for_hotkey_release(hotkey_name, timeout=3.0):
    vk = HOTKEY_VK_MAP.get(hotkey_name)
    if not vk:
        return True
    start = time.time()
    while time.time() - start < timeout:
        if not (user32.GetAsyncKeyState(vk) & 0x8000):
            return True
        time.sleep(0.02)
    return False

def restore_and_focus_window(hwnd):
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.15)
    # If already foreground (same root), do nothing - avoids focus flicker
    # in browsers where SetForegroundWindow+BringWindowToTop dismisses the URL bar.
    try:
        actual = get_foreground_window()
        if actual:
            root_target = user32.GetAncestor(hwnd, GA_ROOT)
            root_actual = user32.GetAncestor(actual, GA_ROOT)
            if actual == hwnd or (root_target and root_actual == root_target):
                return True
    except Exception:
        pass
    fg = get_foreground_window()
    cur_thread = kernel32.GetCurrentThreadId()
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    attached = []
    if fg_thread and fg_thread != cur_thread:
        if user32.AttachThreadInput(cur_thread, fg_thread, True):
            attached.append(fg_thread)
    if target_thread and target_thread != cur_thread and target_thread not in attached:
        if user32.AttachThreadInput(cur_thread, target_thread, True):
            attached.append(target_thread)
    try:
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.03)
        # NOTE: no BringWindowToTop + no Ctrl tap here on purpose.
        # BringWindowToTop reorders Z-order and the Ctrl tap dismisses
        # browser omnibox dropdowns, both observed as "focus disappears".
    finally:
        for tid in attached:
            user32.AttachThreadInput(cur_thread, tid, False)
    actual_fg = get_foreground_window()
    root_target = user32.GetAncestor(hwnd, GA_ROOT)
    root_actual = user32.GetAncestor(actual_fg, GA_ROOT) if actual_fg else 0
    return actual_fg == hwnd or root_actual == root_target

def get_focused_child(top_hwnd):
    if not top_hwnd:
        return None
    fg_thread = user32.GetWindowThreadProcessId(top_hwnd, None)
    cur_thread = kernel32.GetCurrentThreadId()
    attached = False
    focus_hwnd = None
    try:
        if fg_thread and fg_thread != cur_thread:
            attached = bool(user32.AttachThreadInput(cur_thread, fg_thread, True))
        focus_hwnd = user32.GetFocus()
    finally:
        if attached:
            user32.AttachThreadInput(cur_thread, fg_thread, False)
    return focus_hwnd

def restore_focus_to_child(top_hwnd, child_hwnd):
    if not child_hwnd or not user32.IsWindow(child_hwnd):
        return False
    fg_thread = user32.GetWindowThreadProcessId(top_hwnd, None)
    cur_thread = kernel32.GetCurrentThreadId()
    attached = False
    try:
        if fg_thread and fg_thread != cur_thread:
            attached = bool(user32.AttachThreadInput(cur_thread, fg_thread, True))
        result = user32.SetFocus(child_hwnd)
        time.sleep(0.03)
    finally:
        if attached:
            user32.AttachThreadInput(cur_thread, fg_thread, False)
    return result != 0

def _valid_child(child_hwnd):
    try:
        return bool(child_hwnd) and bool(user32.IsWindow(child_hwnd))
    except Exception:
        return False

def verify_focus_ready(target_top_hwnd, target_child_hwnd, max_attempts=4):
    # Stale child hwnds (common in Chrome/Edge omnibox - Aura has no Win32 focus)
    # must be treated as "no child" instead of retried - retrying flashes focus.
    if not _valid_child(target_child_hwnd):
        target_child_hwnd = None
    for _ in range(max_attempts):
        actual_fg = get_foreground_window()
        if actual_fg:
            root_target = user32.GetAncestor(target_top_hwnd, GA_ROOT) if target_top_hwnd else 0
            root_actual = user32.GetAncestor(actual_fg, GA_ROOT)
            if root_actual == root_target or actual_fg == target_top_hwnd:
                if target_child_hwnd:
                    current_child = get_focused_child(actual_fg)
                    if current_child:
                        return True
                    restore_focus_to_child(target_top_hwnd, target_child_hwnd)
                    time.sleep(0.05)
                else:
                    return True
            else:
                if target_top_hwnd and user32.IsWindow(target_top_hwnd):
                    restore_and_focus_window(target_top_hwnd)
                    if target_child_hwnd:
                        restore_focus_to_child(target_top_hwnd, target_child_hwnd)
        time.sleep(0.05)
    # Final check: top window correct but child gone = still usable (browser case)
    try:
        actual_fg = get_foreground_window()
        if actual_fg and target_top_hwnd:
            root_target = user32.GetAncestor(target_top_hwnd, GA_ROOT)
            root_actual = user32.GetAncestor(actual_fg, GA_ROOT)
            if root_actual == root_target or actual_fg == target_top_hwnd:
                return True
    except Exception:
        pass
    return False

# --- Clipboard (robust, ported) ---
def get_clipboard_text():
    opened = False
    for _ in range(8):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.05)
    if not opened:
        return None
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return None
        p = kernel32.GlobalLock(h)
        if not p:
            return None
        try:
            return ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()

def set_clipboard_text(text):
    if text is None:
        text = ""
    text = str(text)
    # Try pyperclip first for simplicity
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    opened = False
    for _ in range(8):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.05)
    if not opened:
        return False
    try:
        user32.EmptyClipboard()
        buf = ctypes.create_unicode_buffer(text)
        bytes_ = (len(text) + 1) * 2
        h = kernel32.GlobalAlloc(GMEM_MOVEABLE, bytes_)
        if not h:
            return False
        p = kernel32.GlobalLock(h)
        if not p:
            kernel32.GlobalFree(h)
            return False
        ctypes.memmove(p, buf, bytes_)
        kernel32.GlobalUnlock(h)
        if not user32.SetClipboardData(CF_UNICODETEXT, h):
            kernel32.GlobalFree(h)
            return False
        return True
    finally:
        user32.CloseClipboard()

# --- Insertion (ported send_ctrl_v / send_unicode_string / insert_text) ---
def send_ctrl_v(log_func=None):
    force_release_modifier_keys()
    scan_v = user32.MapVirtualKeyW(VK_V, 0)
    arr = (INPUT * 4)()
    vals = [(VK_CONTROL, 0, 0), (VK_V, scan_v, 0), (VK_V, scan_v, KEYEVENTF_KEYUP), (VK_CONTROL, 0, KEYEVENTF_KEYUP)]
    for i, (vk, scan, flags) in enumerate(vals):
        arr[i].type = INPUT_KEYBOARD
        arr[i].union.ki.wVk = vk
        arr[i].union.ki.wScan = scan
        arr[i].union.ki.dwFlags = flags
        arr[i].union.ki.time = 0
        arr[i].union.ki.dwExtraInfo = None
    sent = _send_input(arr, 4)
    time.sleep(0.02)
    if sent != 4 and log_func:
        log_func(f"Ctrl+V send incomplete: {sent}/4")
    return sent == 4

def send_unicode_string(text, wake=False, delay_ms=0, batch_size=64, log_func=None):
    if not text:
        return True
    try:
        delay_ms = max(0, int(delay_ms))
    except Exception:
        delay_ms = 0
    try:
        batch_size = max(1, min(int(batch_size), 128))
    except Exception:
        batch_size = 64
    if wake:
        wake_arr = (INPUT * 4)()
        wake_vals = [(0x20, 0), (0x20, KEYEVENTF_KEYUP), (0x08, 0), (0x08, KEYEVENTF_KEYUP)]
        for i, (vk, flags) in enumerate(wake_vals):
            wake_arr[i].type = INPUT_KEYBOARD
            wake_arr[i].union.ki.wVk = vk
            wake_arr[i].union.ki.wScan = 0
            wake_arr[i].union.ki.dwFlags = flags
            wake_arr[i].union.ki.time = 0
            wake_arr[i].union.ki.dwExtraInfo = None
        _send_input(wake_arr, 4)
        time.sleep(0.05)

    def flush(events):
        if not events:
            return True
        arr = (INPUT * len(events))()
        for i, (vk, scan, flags) in enumerate(events):
            arr[i].type = INPUT_KEYBOARD
            arr[i].union.ki.wVk = vk
            arr[i].union.ki.wScan = scan
            arr[i].union.ki.dwFlags = flags
            arr[i].union.ki.time = 0
            arr[i].union.ki.dwExtraInfo = None
        sent = _send_input(arr, len(events))
        if sent != len(events):
            if log_func:
                log_func(f"Unicode send incomplete: {sent}/{len(events)} - retrying remainder")
            # Retry the unsent tail once (common when target message pump is busy,
            # observed as "break" mid-string in browsers).
            try:
                rest = events[sent:] if sent > 0 else events
                if rest:
                    time.sleep(0.02)
                    arr2 = (INPUT * len(rest))()
                    for i, (vk, scan, flags) in enumerate(rest):
                        arr2[i].type = INPUT_KEYBOARD
                        arr2[i].union.ki.wVk = vk
                        arr2[i].union.ki.wScan = scan
                        arr2[i].union.ki.dwFlags = flags
                        arr2[i].union.ki.time = 0
                        arr2[i].union.ki.dwExtraInfo = None
                    sent2 = _send_input(arr2, len(rest))
                    if sent2 != len(rest) and log_func:
                        log_func(f"Unicode retry incomplete: {sent2}/{len(rest)}")
                    else:
                        return True
            except Exception as e:
                if log_func:
                    log_func(f"Unicode retry error: {e}")
            return False
        return True

    ok = True
    events = []
    for char in text:
        if char == "\r":
            continue
        if char == "\n":
            scan = user32.MapVirtualKeyW(VK_RETURN, 0)
            events.append((VK_RETURN, scan, 0))
            events.append((VK_RETURN, scan, KEYEVENTF_KEYUP))
        elif char == "\t":
            scan = user32.MapVirtualKeyW(VK_TAB, 0)
            events.append((VK_TAB, scan, 0))
            events.append((VK_TAB, scan, KEYEVENTF_KEYUP))
        else:
            code = ord(char)
            if code <= 0xFFFF:
                events.append((0, code, KEYEVENTF_UNICODE))
                events.append((0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
            else:
                # Supplementary plane: send as two complete surrogate
                # keystrokes (down+up each), NOT nested down/down/up/up.
                code -= 0x10000
                high = 0xD800 + (code >> 10)
                low = 0xDC00 + (code & 0x3FF)
                events.append((0, high, KEYEVENTF_UNICODE))
                events.append((0, high, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
                events.append((0, low, KEYEVENTF_UNICODE))
                events.append((0, low, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        if len(events) >= batch_size:
            if not flush(events):
                ok = False
            events.clear()
            # Always yield to the target message pump between batches,
            # even when delay_ms=0 - prevents dropped chars ("breaks")
            # in Chrome/Edge omnibox on fast machines.
            time.sleep((delay_ms / 1000.0) if delay_ms > 0 else 0.005)
    if events:
        if not flush(events):
            ok = False
    return ok

def prepare_insertion_focus(target_top_hwnd, target_child_hwnd, hotkey_name, our_root_hwnd=0, allow_self=False, log_func=None):
    wait_for_hotkey_release(hotkey_name, timeout=3.0)
    force_release_modifier_keys()
    time.sleep(0.05)
    if target_top_hwnd and user32.IsWindow(target_top_hwnd):
        target_root = user32.GetAncestor(target_top_hwnd, GA_ROOT)
        if our_root_hwnd and target_root == our_root_hwnd and not allow_self:
            target_top_hwnd = None
            target_child_hwnd = None
        else:
            restore_and_focus_window(target_top_hwnd)
            if _valid_child(target_child_hwnd):
                time.sleep(0.05)
                restore_focus_to_child(target_top_hwnd, target_child_hwnd)
            verify_focus_ready(target_top_hwnd, target_child_hwnd, max_attempts=4)
    if not (target_top_hwnd and user32.IsWindow(target_top_hwnd)):
        fg = get_foreground_window()
        fg_root = user32.GetAncestor(fg, GA_ROOT) if fg else 0
        if fg and (allow_self or (our_root_hwnd and fg_root != our_root_hwnd) or our_root_hwnd == 0):
            target_top_hwnd = fg
            target_child_hwnd = get_focused_child(fg)
            if not _valid_child(target_child_hwnd):
                target_child_hwnd = None
            restore_and_focus_window(target_top_hwnd)
            if target_child_hwnd:
                time.sleep(0.05)
                restore_focus_to_child(target_top_hwnd, target_child_hwnd)
            verify_focus_ready(target_top_hwnd, target_child_hwnd, max_attempts=4)
        else:
            return None, None, False
    force_release_modifier_keys()
    time.sleep(0.12)
    return target_top_hwnd, target_child_hwnd, True

def insert_text(text, target_top_hwnd, target_child_hwnd, hotkey_name, our_root_hwnd=0, method="clipboard", allow_self=False, wake=False, delay_ms=0, log_func=None):
    """Ported insert_text - tries focus restore, then clipboard or unicode."""
    if not text:
        return True
    method = str(method).lower()
    if method not in ("unicode", "clipboard"):
        method = "clipboard"

    top, child, ok = prepare_insertion_focus(target_top_hwnd, target_child_hwnd, hotkey_name, our_root_hwnd=our_root_hwnd, allow_self=allow_self, log_func=log_func)
    if not ok:
        set_clipboard_text(text)
        if log_func:
            log_func("No external target - copied to clipboard")
        return False

    # Small suffix handling if caller wants (we default to no suffix)
    if method == "clipboard":
        old_text = get_clipboard_text()
        if set_clipboard_text(text):
            force_release_modifier_keys()
            paste_ok = send_ctrl_v(log_func=log_func)
            # 0.35s was too short for browsers to consume the paste on
            # slower machines - the follow-up clipboard rewrite then won
            # the race and the URL bar ended up with a partial string.
            time.sleep(0.5)
            if old_text is not None:
                # Restore old clipboard as plain text (matches test.py behavior)
                set_clipboard_text(old_text)
                # Re-copy our text so user still has it (improvement)
                set_clipboard_text(text)
            # NOTE: old_text None means the clipboard held non-text data
            # (e.g. an image) or was locked - in that case leave our text
            # in place instead of wiping the user's data with "".
            return paste_ok
        else:
            # Fallback to unicode if clipboard set failed
            pass

    return send_unicode_string(text, wake=wake, delay_ms=delay_ms, log_func=log_func)

# --- Simple wrappers for backwards compat ---
def copy_to_clipboard(text: str) -> bool:
    return set_clipboard_text(text)

def paste_text(text: str) -> bool:
    """Legacy: copy + ctrl+v without focus logic. Now calls insert_text with current fg."""
    if not text or not text.strip():
        return False
    fg = get_foreground_window()
    child = get_focused_child(fg)
    # Use 0 for our_root_hwnd so it won't reject fg if we have no GUI hwnd yet
    return insert_text(text, fg, child, "F2", our_root_hwnd=0, method="clipboard")

def get_clipboard_text_simple() -> str:
    t = get_clipboard_text()
    return t or ""
