"""Portable trivial native functions used as a cheap-FFI reference.

We need a real native call that does ~nanoseconds of work, so the
measured time is dominated by the call/marshalling path rather than the
callee. ``vkCmdDispatch`` lives in this regime (a dispatch-table call
that records a few ints); ``vkEnumerateInstanceVersion`` does not (the
loader rescans ICDs). We pick per-OS equivalents:

- 0-arg, returns an int:    GetTickCount (win) / getpid (posix)
- 1 pointer arg, returns int: QueryPerformanceCounter (win) / time (posix)
"""
from __future__ import annotations

import ctypes
import sys


def zero_arg():
    """Return (dll, symbol_name, restype, raw_callable) for a 0-arg native fn."""
    if sys.platform == "win32":
        dll = ctypes.windll.kernel32
        fn = dll.GetTickCount
        fn.restype = ctypes.c_uint32
        fn.argtypes = []
        return dll, "GetTickCount", ctypes.c_uint32, fn
    dll = ctypes.CDLL(None)
    fn = dll.getpid
    fn.restype = ctypes.c_int
    fn.argtypes = []
    return dll, "getpid", ctypes.c_int, fn


def ptr_arg():
    """Return (symbol_name, raw_callable, arg) for a native fn taking one pointer.

    *arg* is a ready-to-pass ``byref``-style pointer object so the
    benchmark loop does no per-iteration allocation.
    """
    if sys.platform == "win32":
        fn = ctypes.windll.kernel32.QueryPerformanceCounter
        fn.restype = ctypes.c_int32
        fn.argtypes = [ctypes.c_void_p]
        buf = ctypes.c_int64(0)
        return "QueryPerformanceCounter", fn, ctypes.byref(buf)
    fn = ctypes.CDLL(None).time
    fn.restype = ctypes.c_long
    fn.argtypes = [ctypes.c_void_p]
    buf = ctypes.c_long(0)
    return "time", fn, ctypes.byref(buf)
