"""Minimal lazy-registry kernel — three sentinels and a memoising mapping.

This is the simplified parallel to :mod:`lazyregistry.dsl`. The DSL kernel
exposes a 5-form template-evaluation language with parent walks, lexical
scoping, multi-step ``EVAL-N``, and sub-registries. The Vulkan binding
uses only four of those patterns (force-lookup, factory-call,
quote-passthrough, literal) — everything else is unused machinery. This
module collapses the kernel to just those four patterns expressed as
plain Python objects.

Three sentinels suffice:

- :class:`_Thunk` ``(fn, *args)`` — when forced, resolves ``fn`` and each
  ``arg`` then calls ``fn(*resolved_args)``. Captures a deferred function
  call; nested thunks evaluate inside-out.
- :class:`_Raw` ``(value)`` — quote escape: ``value`` is returned without
  descending into it. The way recursively-typed structs (``VkBaseIn-
  Structure.pNext``) avoid the cycle-detection trap that eager arg
  descent would create.
- :data:`SELF` — singleton sentinel that resolves to the registry itself.
  Used as ``_Thunk(SELF, "name")`` (= cached lookup of a registry key)
  and as a placeholder arg to factories that need a registry reference.

The :class:`Lazy` mapping memoises each key's resolved value on first
force, raising :class:`ValueError` on cyclic resolution. Reading a raw
form back without forcing is :meth:`Lazy.__getitem__`; forcing (with
cache) is :meth:`Lazy.__call__`.

There is deliberately no expression form and no :func:`eval` anywhere
in the resolution path. The kernel once carried one, for arithmetic a
source language might want deferred, but the Vulkan parser never
emitted a single node of it: forcing every key across registries from
2016 to today produced zero. Arithmetic that vk.xml *does* contain —
``(~0U-1)`` sentinels, ``VK_MAKE_VERSION`` macros — is folded by hand
in :mod:`volkano.xml_parser`, over a grammar small enough to read.
Third-party XML should not reach an interpreter.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping


# Module-level logger. Stays quiet at WARNING by default; the kernel
# emits a single DEBUG record per cache-miss force (and nothing on cache
# hits), so flipping the ``volkano`` logger to DEBUG gives a clean
# audit trail of every name the program actually resolves. INFO and
# above are reserved for lifecycle events emitted from
# :mod:`vulkan_stdlib` (library attach, command bind, etc.) so callers
# can subscribe to a quiet overview without drowning in every force.
logger = logging.getLogger('volkano.lazy')


# Cache sentinels — module-private. ``_MISSING`` distinguishes "never
# forced" from "force returned None"; ``_RESOLVING`` is parked in the
# cache while a key is being resolved so re-entry raises instead of
# recursing forever.
_MISSING = object()
_RESOLVING = object()


class _SelfMarker:
    """Type of the :data:`SELF` singleton — see module docstring."""
    __slots__ = ()

    def __repr__(self):
        return 'SELF'


#: Resolves to the registry itself. The thunk-graph's way of saying "look
#: this up in / pass this registry as an arg" without holding a runtime
#: reference at parse time (which would create a chicken-and-egg between
#: data construction and registry construction).
SELF = _SelfMarker()


class _Thunk:
    """Deferred function call. Resolve fn and args, then invoke.

    The fn slot is normally either a plain Python callable (a stdlib
    factory, an already-resolved Python value), the :data:`SELF`
    sentinel (to invoke the registry's cached lookup), or another
    :class:`_Thunk` whose resolution yields a callable. Args are
    descended element-wise during resolution; pass :class:`_Raw` to
    bypass descent for a single arg.
    """

    __slots__ = ('fn', 'args')

    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args

    def __repr__(self):
        return f'_Thunk({self.fn!r}, *{self.args!r})'


class _Raw:
    """Quote escape: the contained value is returned as-is, no descent.

    Factories that receive raw data are responsible for descending it
    themselves (typically by calling :meth:`Lazy._resolve` after planting
    a placeholder in the cache, the standard cycle-safe pattern).
    """

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'_Raw({self.value!r})'


class Lazy(Mapping):
    """Memoising registry. Force a key via :meth:`__call__`; raw via ``[]``.

    Implements the :class:`collections.abc.Mapping` protocol over the
    underlying ``data`` dict so it composes with ordinary Python idioms
    (``len(vk)``, ``"VkInstance" in vk``, ``for key in vk:``). Forcing a
    key is :meth:`__call__`; reading the raw stored thunk back without
    forcing is :meth:`__getitem__`.

    Construction is cheap — just stashes the data dict and an empty
    cache. Unlike the DSL kernel, no ``freeze()`` pass walks the data
    structure at init time, so cold start is dominated by ``len(data)``
    not by tree depth.

    Forcing is thread-safe: :meth:`__call__` takes ``_lock`` on the
    cache-miss path, so two threads racing the same key can't each
    build a value and hand out two distinct ctypes classes for one
    Vulkan type. Cache *hits* stay lock-free.
    """

    __slots__ = ('_data', '_cache', '_lock')

    def __init__(self, data):
        self._data = data
        self._cache = {}
        # Re-entrant because resolution nests: forcing a struct resolves
        # its field types through the same __call__, and make_struct
        # calls registry._resolve on the quoted field list while its own
        # force frame is still open. A plain Lock would self-deadlock on
        # the first cross-reference.
        self._lock = threading.RLock()

    # --- Mapping protocol over the raw data ---------------------------------

    def __getitem__(self, key):
        """Raw stored value at ``key`` — no force, no cache touch."""
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    # --- Force / resolve ----------------------------------------------------

    def __call__(self, key):
        """Force ``key``: descend the stored thunk graph, cache, return value.

        The cache is keyed on ``key``; the first force evaluates and
        memoises, every subsequent force is constant-time. Cyclic
        resolution (``A`` references ``B`` which references ``A``)
        raises :class:`ValueError` instead of recursing — see the
        ``_RESOLVING`` sentinel below.

        Double-checked: the fast path reads the cache without the lock
        (a dict get is atomic under the GIL), and only a miss pays for
        acquisition. Note the fast path deliberately treats
        ``_RESOLVING`` as a *miss* — that sentinel means "some frame is
        building this", which is a cycle only when it's our own frame.
        Another thread's in-flight build must block on the lock and
        re-read, not raise.
        """
        cached = self._cache.get(key, _MISSING)
        if cached is not _MISSING and cached is not _RESOLVING:
            return cached
        with self._lock:
            # Re-read: a thread that was building this key may have
            # finished while we waited. Inside the lock _RESOLVING can
            # only be our own frame, so it does mean a cycle.
            cached = self._cache.get(key, _MISSING)
            if cached is _RESOLVING:
                raise ValueError(f"cyclic key reference: {key!r}")
            if cached is not _MISSING:
                return cached
            if key not in self._data:
                raise KeyError(key)
            # One DEBUG record per cache-miss force. Cache hits stay
            # silent so a "show me everything that was resolved" trace
            # doesn't get diluted with re-access noise — every line in
            # the log is one additional ctypes type materialised.
            logger.debug('force %s', key)
            self._cache[key] = _RESOLVING
            try:
                value = self._resolve(self._data[key])
            except BaseException:
                del self._cache[key]
                raise
            self._cache[key] = value
            return value

    def _resolve(self, v):
        """Recursively force a thunk graph fragment.

        - :data:`SELF` → the registry itself.
        - :class:`_Raw` → its wrapped value (quote: don't descend).
        - :class:`_Thunk` → resolve fn and each arg, then ``fn(*args)``.
        - ``list`` / ``dict`` → element-wise descent (so the JSON-shaped
          tree of struct member entries resolves the per-field type
          thunks while keeping the outer ``[name, type]`` shape).
        - Anything else → pass through as a literal atom.

        Factory functions in the stdlib treat ``_resolve`` as part of
        the public API: :func:`make_struct` calls ``registry._resolve``
        on the quoted fields list to perform descent after the
        in-progress class has been planted in the cache.
        """
        if v is SELF:
            return self
        if isinstance(v, _Raw):
            return v.value
        if isinstance(v, _Thunk):
            fn = self._resolve(v.fn)
            args = [self._resolve(a) for a in v.args]
            return fn(*args)
        if isinstance(v, list):
            return [self._resolve(x) for x in v]
        if isinstance(v, dict):
            return {k: self._resolve(x) for k, x in v.items()}
        return v

    def __repr__(self):
        return f'Lazy(keys={len(self._data)}, cached={len(self._cache)})'


# ---------------------------------------------------------------------------
# DSL-compatible builders — the same names xml_parser.Force / Call / Quote
# use, so generators are textually almost identical to the dsl.py version.
# ---------------------------------------------------------------------------

def Force(name):
    """Cached registry lookup of ``name``.

    Equivalent to the DSL's ``["@", None, ["@", None], [name], {}]``
    force-lookup, but expressed as a one-thunk graph: ``_Thunk(SELF,
    name)`` resolves to ``self(name)``, which is the cached
    ``__call__`` path. Two callers naming the same type share the same
    cached value — the property that lets ctypes class identity stay
    consistent across struct fields and command argtypes.
    """
    return _Thunk(SELF, name)


def Call(fn_name, args, with_registry=False):
    """Invoke a stdlib factory by name with the given args.

    ``with_registry=True`` prepends :data:`SELF` to ``args`` — the
    convention for factories that need to chase type cross-references
    through ``registry._resolve`` and plant entries in
    ``registry._cache``. The fn slot is itself a :func:`Force` lookup,
    so the factory name resolves through the cache just like everything
    else.
    """
    if with_registry:
        args = [SELF, *args]
    return _Thunk(Force(fn_name), *args)


def Quote(value):
    """Pass ``value`` through resolve without descending into it.

    The quote escape that lets factories receive raw, un-evaluated data
    (used for struct field lists so :func:`make_struct` can plant the
    in-progress class in the cache before descending the field types —
    breaking self-reference cycles like ``VkBaseInStructure.pNext``).
    """
    return _Raw(value)
