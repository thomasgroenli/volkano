"""volkano host-side microbenchmarks.

Run with ``python -m bench``.

These measure the *host* (CPU) cost of driving Vulkan through volkano's
pure-Python ctypes layer — struct construction, the command-call wrapper
tax, and the composite operations a higher-level dispatch (see
``../kaldera``) performs per kernel launch. No GPU work is involved; the
GPU executes identical SPIR-V regardless of the host language, so this
host overhead is the entire cost difference versus a compiled C driver.

Key findings the numbers support:

- The volkano ``CommandSignature`` wrapper adds only ~0.1-0.2 us over a
  raw ctypes call; the FFI is *not* the bottleneck.
- Per-dispatch cost is dominated by Python *object construction* — most
  of it the descriptor writes — at ~1.5 us per struct. Caching the
  descriptor set and reusing recorded command buffers is what closes the
  gap to C, not a faster FFI.
- A few Vulkan loader calls (instance/version enumeration) cost hundreds
  of microseconds in *any* language; they happen once at device build,
  never per dispatch. ``host_dispatch`` records one as a labelled
  reference so the per-dispatch numbers aren't mistaken for them.
"""
