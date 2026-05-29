"""Stdlib factories the XML-derived thunk graph references at force time.

The simplified parallel to :mod:`lazyregistry.vulkan_stdlib`. Factories
are byte-for-byte identical between the two packages — they accept the
same arguments and produce the same ctypes types. The differences are
all in the host:

- :class:`VkRegistry` subclasses :class:`volkano.lazy.Lazy`
  (~50-line kernel) rather than :class:`lazyregistry.dsl.Registry`
  (~500-line template-evaluation language). Storage is mutable, no
  ``freeze()`` pass at construction.
- :func:`build_registry` plumbs through the same call sites; the only
  visible change is the parser module it imports.
"""

from __future__ import annotations

import ctypes
import logging
import sys

from .cbase import (
    Handle, Enum, FunctionPointer, Pointer, Array, Struct, Union, friendly,
    char,
    int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    float32, float64, size,
)
from .lazy import Lazy


# Module logger. Lifecycle events (registry built, library attached,
# rebind summaries) emit at INFO so a single ``logging.basicConfig(
# level=logging.INFO)`` is enough to see what's happening without
# drowning in every per-key force trace; per-command bindings and
# missing exports drop to DEBUG so the user opts into that detail
# explicitly. Loggers are nested under the package name so flipping
# ``logging.getLogger('volkano')`` to DEBUG enables everything
# at once, parser + kernel + stdlib.
logger = logging.getLogger('volkano.stdlib')


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

#: Primitive type table — these become top-level registry entries with
#: literal type values, so a Force-lookup of ``"uint32_t"`` resolves
#: straight to :class:`cbase.uint32` without going through any factory.
#:
#: Every slot maps to a cbase type so that whatever flows into a struct
#: field or command signature carries the cbase surface (arithmetic,
#: comparison, ``.ref``, ``.ptr``, …) and decays correctly into the
#: cbase Pointer slots ``ref()`` produces. The C primitives without a
#: fixed-width cbase scalar use the closest one (``int`` → 32-bit
#: :class:`cbase.int32`, ``bool`` → 1-byte :class:`cbase.uint8`),
#: ``void *`` is ``Pointer[None]`` and ``char *`` is ``Pointer[char]``
#: (its data supplied as ``bytes``). No raw ctypes type reaches the
#: registry.
PRIMITIVES: dict = {
    'void':     None,
    'char':     char,
    'int':      int32,
    'float':    float32,
    'double':   float64,
    'size_t':   size,
    'int8_t':   int8,
    'int16_t':  int16,
    'int32_t':  int32,
    'int64_t':  int64,
    'uint8_t':  uint8,
    'uint16_t': uint16,
    'uint32_t': uint32,
    'uint64_t': uint64,
    'bool':     uint8,
    '_Bool':    uint8,
    'intptr_t':  int64,      # 64-bit on the x86_64 targets we support
    'uintptr_t': uint64,
    'ptrdiff_t': int64,
    'void_p':   Pointer[None],
}


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def ref(target):
    """Pointer-to-``target`` for use in a Vulkan struct's ``_fields_``.

    Short-circuits ``void *`` → ``Pointer[None]`` (the type-erased
    pointer slot). IntEnum / IntFlag element types pass through
    :func:`cbase.friendly` since ctypes can't form a pointer to an enum
    class directly. Every other target — ``char`` included, so
    ``char *`` → ``Pointer[char]`` (its data supplied as ``bytes``) —
    yields a cbase :class:`cbase.Pointer` subclass.
    """
    if target is None:
        return Pointer[None]
    return Pointer[target]


def sizeof(target):
    """``ctypes.sizeof`` with enum-element normalisation.

    IntEnum / IntFlag classes don't carry storage info ctypes can size,
    so :func:`cbase.friendly` substitutes the registered backing scalar
    before measuring. ``None`` (used to represent ``void``) raises,
    matching C's ``sizeof(void)`` being ill-formed.
    """
    if target is None:
        raise TypeError("can't take sizeof void")
    return ctypes.sizeof(friendly(target))


def array(target, length):
    """Fixed-size array — cbase ``Array[target, length]``.

    Keeps arrays on the cbase substrate (consistent with :func:`ref`
    returning a cbase :class:`~cbase.Pointer`), so a struct's array
    field carries the :class:`~cbase.Mixin` surface and names itself
    ``Array[elem, N]``. ``None`` collapses to a void-pointer element
    and IntEnum / IntFlag elements are normalised — both handled inside
    :meth:`cbase.Array.__class_getitem__`.
    """
    return Array[target, length]


def make_handle(registry, name, dispatchable=True, parent=None,
                objtypeenum=None):
    """``<type category="handle">`` — opaque ABI-sized object handle.

    Returns a :class:`cbase.Handle` subclass (which is itself a
    ``c_void_p`` subclass with the :class:`cbase.Mixin` surface).
    Vulkan-specific metadata (``_vk_kind`` / ``_vk_dispatchable`` /
    ``_vk_parent`` / ``_vk_objtypeenum``) is attached as class
    attributes after construction so introspection tools and the raii
    layer can read the dispatch/parent topology back off any handle
    class.
    """
    cls = Handle.create(name)
    cls._vk_kind = 'handle'
    cls._vk_dispatchable = dispatchable
    cls._vk_parent = parent
    cls._vk_objtypeenum = objtypeenum
    return cls


def make_basetype(registry, name, underlying=None):
    """``<type category="basetype">`` — typedef for a primitive or opaque struct."""
    if underlying and underlying in PRIMITIVES:
        return PRIMITIVES[underlying]
    return Pointer[None]


def make_bitmask(registry, name, bitwidth=32, requires=None):
    """``<type category="bitmask">`` — ``typedef VkFlags X`` flag typedef."""
    return uint64 if bitwidth == 64 else uint32


def make_enum(registry, name, kind='enum', bitwidth=32, values=(),
              aliases=()):
    """``<enums>`` group — IntEnum / IntFlag class plus a backing scalar.

    Thin adapter over :meth:`cbase.Enum.create`: the XML's ``kind``
    attribute (``'enum'`` / ``'bitmask'``) maps to cbase's ``flag``
    kwarg. Backing-scalar registration (used by struct/funcpointer
    field substitution) happens inside ``Enum.create``.
    """
    return Enum.create(name, values=values, aliases=aliases,
                       bitwidth=bitwidth, flag=(kind == 'bitmask'))


def make_funcpointer(registry, name, rettype, params=()):
    """``<type category="funcpointer">`` — typedef'd C function pointer.

    Delegates to :meth:`cbase.FunctionPointer.create` with
    ``stdcall=True`` on win32 to honour Vulkan's ``VKAPI_PTR``
    (``__stdcall`` on 32-bit Windows, no-op on x86_64). Each PFN
    typedef gets its own :class:`cbase.Mixin`-bearing subclass over
    the cached ctypes ``_CFuncPtr`` — so two PFNs with identical
    signatures (e.g. ``PFN_vkInternalAllocationNotification`` and
    ``PFN_vkInternalFreeNotification``, both ``void(void*, size_t,
    VkInternalAllocationType, VkSystemAllocationScope)``) keep their
    distinct Python class names instead of collapsing into one
    ``__name__`` (the old behaviour required hand-waving in the stub
    generator).
    """
    return FunctionPointer.create(name, rettype, params or (),
                                  stdcall=(sys.platform == 'win32'))


def make_struct(registry, name, fields_raw):
    """``<type category="struct">`` with self-cycle-safe field resolution.

    ``fields_raw`` is the unevaluated fields list — wrapped in a
    :class:`_Raw` by the parser so the kernel didn't descend into it.
    The factory plants the empty class in ``registry._cache[name]``
    *before* descending the fields, so any thunk that re-asks the
    registry for ``name`` (the recursive-pNext pattern) finds the
    in-progress class instead of re-entering this factory.
    """
    return _build_struct_like(registry, name, fields_raw, Struct)


def make_union(registry, name, fields_raw):
    """``<type category="union">`` — same self-cycle handling as :func:`make_struct`."""
    return _build_struct_like(registry, name, fields_raw, Union)


def _build_struct_like(registry, name, fields_raw, base):
    """Two-step build over :class:`cbase.Struct` / :class:`cbase.Union`.

    The empty class is created first and planted in
    ``registry._cache[name]`` so self-referential field types
    (``VkBaseInStructure.pNext``, intrusive ``next`` pointers) find
    the in-progress class on lookup instead of re-entering this
    factory. After ``_resolve`` descends the field-type graph,
    :meth:`cbase.StructureMixin.set_fields` finalises the layout
    (applying :func:`cbase.friendly` per-field, handling bitfields
    and anonymous members).
    """
    cls = base.create_empty(name)
    cls._vk_name = name
    registry._cache[name] = cls
    resolved = registry._resolve(fields_raw)
    base.set_fields(cls, resolved)
    return cls


# ---------------------------------------------------------------------------
# Command signatures
# ---------------------------------------------------------------------------

class CommandSignature:
    """Lazy ctypes binding for one Vulkan command.

    Cheap to construct; :meth:`bind` resolves the symbol on an open DLL
    and configures ctypes argtypes/restype. After binding, callable
    directly via ``__call__``. Calls before binding raise a descriptive
    :class:`RuntimeError`.
    """

    __slots__ = ('name', 'rettype', 'params', 'attrs', '_fn')

    def __init__(self, name, rettype, params, attrs):
        self.name = name
        self.rettype = rettype
        self.params = params
        self.attrs = attrs
        self._fn = None

    def bind(self, dll):
        try:
            fn = getattr(dll, self.name)
        except AttributeError:
            # Most "missing" commands are extension entry points the
            # loader doesn't surface — they need vkGetInstanceProcAddr /
            # vkGetDeviceProcAddr. Log at DEBUG so the noise is opt-in.
            logger.debug('command %s not exported by loader', self.name)
            return None
        fn.argtypes = tuple(friendly(p[1]) for p in self.params)
        fn.restype = friendly(self.rettype) if self.rettype is not None else None
        self._fn = fn
        logger.debug('bound %s', self.name)
        return fn

    def __call__(self, *args):
        if self._fn is None:
            raise RuntimeError(
                f"Command {self.name!r} is not bound. Either the Vulkan "
                f"loader doesn't export this entry point (typical for "
                f"extension commands — fetch via vkGetInstanceProcAddr / "
                f"vkGetDeviceProcAddr instead), or the registry was built "
                f"without a library (pass library=True to build_registry, "
                f"or call vk.attach_library(...) followed by "
                f"vk.rebind_commands()).")
        return self._fn(*args)

    def __repr__(self):
        state = 'bound' if self._fn is not None else 'unbound'
        return f'<CommandSignature {self.name} ({state})>'


def make_command(registry, name, rettype, params=(), attrs=None):
    """``<command>`` — produces a :class:`CommandSignature`.

    Eagerly binds against ``registry._library`` if one is attached, so
    every force of a command name returns a directly-callable bound
    function. With Registry caching, this means *forcing a command =
    binding it once*.
    """
    sig = CommandSignature(name, rettype, list(params), dict(attrs or {}))
    lib = getattr(registry, '_library', None)
    if lib is not None:
        sig.bind(lib)
    return sig


# ---------------------------------------------------------------------------
# VkRegistry — attribute-style access + DSO handle
# ---------------------------------------------------------------------------

class VkRegistry(Lazy):
    """Vulkan-flavoured :class:`Lazy` with attribute access and lazy binding.

    Mirrors :class:`lazyregistry.vulkan_stdlib.VkRegistry` exactly —
    same ``__getattr__`` semantics, same ``attach_library`` /
    ``rebind_commands`` surface, same ``.ptr`` / ``.ref`` instance
    shortcuts wired into every struct/union/handle class produced by
    the factories. The only thing different is the parent kernel.
    """

    __slots__ = ('_library',)

    def __init__(self, data, *, library=None):
        super().__init__(data)
        self._library = None
        if library is not None:
            self.attach_library(library)

    def attach_library(self, library):
        """Attach a Vulkan loader DSO so subsequent command forces bind.

        ``library`` may be ``True`` (autodetect), a filename / path
        string (passed to :class:`ctypes.CDLL`), or an already-open
        :class:`ctypes.CDLL` instance. Attaching *after* commands have
        already been forced won't retroactively bind them — call
        :meth:`rebind_commands` in that case.
        """
        if library is True:
            library = _default_library_path()
        if isinstance(library, str):
            library = ctypes.CDLL(library)
        self._library = library
        # ``_name`` is set by ctypes.CDLL.__init__; falls back to repr
        # if a caller hands us a custom DSO-like object.
        logger.info('attached library: %s',
                    getattr(library, '_name', None) or repr(library))
        return library

    def rebind_commands(self):
        """Walk the cache and bind every already-forced :class:`CommandSignature`."""
        if self._library is None:
            return 0
        bound = 0
        for value in self._cache.values():
            if isinstance(value, CommandSignature) and value._fn is None:
                if value.bind(self._library) is not None:
                    bound += 1
        logger.info('rebound %d command(s) against %s',
                    bound, getattr(self._library, '_name', None) or repr(self._library))
        return bound

    def __getattr__(self, name):
        # Only invoked on normal-lookup miss, so kernel attributes
        # (``_data``, ``_cache``, ``_library``) are never intercepted.
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._data:
            raise AttributeError(name)
        return self(name)

    def __dir__(self):
        base = set(super().__dir__())
        base.update(k for k in self._data if isinstance(k, str)
                    and k.isidentifier() and not k.startswith('_'))
        return sorted(base)


# ---------------------------------------------------------------------------
# Library construction
# ---------------------------------------------------------------------------

def _default_library_path():
    if sys.platform == 'win32':
        return 'vulkan-1.dll'
    if sys.platform == 'darwin':
        return 'libvulkan.1.dylib'
    return 'libvulkan.so.1'


def build_registry(source=None, *, library=None, refresh=False):
    """Construct a :class:`VkRegistry` from a Vulkan XML registry.

    Same signature and semantics as
    :func:`lazyregistry.vulkan_stdlib.build_registry`:

    - ``source`` is ``None`` (fetch from GitHub), a URL, a filesystem
      path, or an :class:`xml.etree.ElementTree.Element`.
    - ``library`` opens the Vulkan loader at construction time and
      enables lazy command binding.
    - ``refresh=True`` forces a redownload of cached XML.
    """
    from .xml_parser import parse_xml
    import time
    t0 = time.perf_counter()
    data = parse_xml(source, refresh=refresh)
    merged: dict = {**STDLIB, **data}
    registry = VkRegistry(merged, library=library)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info('built registry: %d entries in %.1f ms', len(registry), elapsed_ms)
    return registry


# ---------------------------------------------------------------------------
# Stdlib namespace — merged into the registry by :func:`build_registry`
# ---------------------------------------------------------------------------

STDLIB: dict = {}
STDLIB.update(PRIMITIVES)
STDLIB.update({
    'ref':              ref,
    'array':            array,
    'sizeof':           sizeof,
    'make_handle':      make_handle,
    'make_basetype':    make_basetype,
    'make_bitmask':     make_bitmask,
    'make_enum':        make_enum,
    'make_funcpointer': make_funcpointer,
    'make_struct':      make_struct,
    'make_union':       make_union,
    'make_command':     make_command,
})
