"""Stdlib factories the XML-derived thunk graph references at force time.

The simplified parallel to :mod:`lazyregistry.vulkan_stdlib`. Factories
are byte-for-byte identical between the two packages — they accept the
same arguments and produce the same ctypes types. The differences are
all in the host:

- :class:`VkRegistry` subclasses :class:`volkano.lazy.Lazy`
  (~50-line kernel) rather than :class:`lazyregistry.dsl.Registry`
  (~500-line template-evaluation language). Storage is mutable, no
  ``freeze()`` pass at construction.
- :func:`build_registry` has diverged: it takes volkano's own source
  vocabulary (see :func:`volkano.xml_source.resolve_source`), defers
  opening the Vulkan loader to the first command force, and records the
  provenance of the XML it parsed. It has no ``refresh`` parameter and
  no knowledge of the ``.pyi`` stub.
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

    ``_unbound`` records *why* binding didn't happen, set by
    :func:`make_command` which is the only place that knows. Three
    causes look identical from here but need very different fixes, and
    guessing at the wrong one sends people hunting for a
    ``vkGetInstanceProcAddr`` problem when their loader simply isn't
    installed.
    """

    __slots__ = ('name', 'rettype', 'params', 'attrs', '_fn', '_unbound')

    #: ``_unbound`` values. ``NO_LIBRARY``: the registry was built with
    #: ``library=None``, so nothing was ever opened. ``OPEN_FAILED``: a
    #: loader was requested but ``ctypes.CDLL`` refused it (a warning was
    #: already logged). ``NOT_EXPORTED``: the loader opened fine and
    #: simply doesn't export this symbol.
    NO_LIBRARY = 'no-library'
    OPEN_FAILED = 'open-failed'
    NOT_EXPORTED = 'not-exported'

    def __init__(self, name, rettype, params, attrs):
        self.name = name
        self.rettype = rettype
        self.params = params
        self.attrs = attrs
        self._fn = None
        self._unbound = self.NO_LIBRARY

    def bind(self, dll):
        try:
            fn = getattr(dll, self.name)
        except AttributeError:
            # Most "missing" commands are extension entry points the
            # loader doesn't surface — they need vkGetInstanceProcAddr /
            # vkGetDeviceProcAddr. Log at DEBUG so the noise is opt-in.
            logger.debug('command %s not exported by loader', self.name)
            self._unbound = self.NOT_EXPORTED
            return None
        fn.argtypes = tuple(friendly(p[1]) for p in self.params)
        fn.restype = friendly(self.rettype) if self.rettype is not None else None
        self._fn = fn
        logger.debug('bound %s', self.name)
        return fn

    def __call__(self, *args):
        if self._fn is None:
            raise RuntimeError(
                f"Command {self.name!r} is not bound: {self._reason()}")
        return self._fn(*args)

    def _reason(self):
        if self._unbound == self.NOT_EXPORTED:
            return ("the Vulkan loader doesn't export this entry point. That's "
                    "normal for extension commands - fetch it at runtime via "
                    "vkGetInstanceProcAddr / vkGetDeviceProcAddr.")
        if self._unbound == self.OPEN_FAILED:
            return ("the Vulkan loader could not be opened (a warning naming "
                    "the path was logged when it was first needed). Install a "
                    "Vulkan runtime, or point VOLKANO_LIBRARY at the loader.")
        return ("this registry was built with library=None, so no loader was "
                "ever opened. Use volkano.registry(..., library=True), or "
                "call attach_library(...) then rebind_commands().")

    def __repr__(self):
        state = 'bound' if self._fn is not None else 'unbound'
        return f'<CommandSignature {self.name} ({state})>'


def make_command(registry, name, rettype, params=(), attrs=None):
    """``<command>`` — produces a :class:`CommandSignature`.

    Binds against the registry's loader, opening it on the first command
    force — this is the point where a deferred ``library=`` spec is
    actually realised. Every subsequent force finds it already open, so
    *forcing a command = binding it once*.

    ``_ensure_library`` is reached through :func:`getattr` because the
    factories are also called with ``registry=None`` (and with plain
    :class:`Lazy` instances) in tests.
    """
    sig = CommandSignature(name, rettype, list(params), dict(attrs or {}))
    ensure = getattr(registry, '_ensure_library', None)
    lib = ensure() if ensure is not None else None
    if lib is not None:
        sig.bind(lib)
    elif getattr(registry, '_library_spec', None) is not None:
        # A loader was asked for and we failed to open it — distinct from
        # never having asked, and the warning is already in the log.
        sig._unbound = sig.OPEN_FAILED
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

    __slots__ = ('_library', '_library_spec', '_library_resolved', '_provenance')

    def __init__(self, data, *, library=None):
        super().__init__(data)
        self._library = None
        # The spec is held, not realised: opening the loader is deferred
        # to the first *command* force, so constants, enums and structs
        # resolve on a machine with no Vulkan driver installed. Building
        # a registry is a parsing job; it shouldn't need a GPU stack.
        self._library_spec = library
        self._library_resolved = library is None
        self._provenance = None

    def _open_library(self, spec):
        """Realise a library spec into an open DSO handle.

        ``spec`` may be ``True`` (autodetect by platform), a filename /
        path string (handed to :class:`ctypes.CDLL`), or an already-open
        CDLL-like object, which passes through untouched.
        """
        if spec is True:
            spec = _default_library_path()
        if isinstance(spec, str):
            spec = ctypes.CDLL(spec)
        return spec

    def _ensure_library(self):
        """Open the deferred library spec once; return the handle or ``None``.

        Never raises. A missing loader has to leave commands *resolvable
        but unbound*, because :func:`volkano.stub.write_stub` forces
        every key inside a blanket ``except Exception`` — raising here
        would quietly reduce every command in the generated stub to
        ``Any`` on any machine without a driver. The failure surfaces
        instead at call time, via :meth:`CommandSignature._reason`.
        """
        if self._library_resolved:
            return self._library
        with self._lock:
            if self._library_resolved:
                return self._library
            spec = self._library_spec
            try:
                library = self._open_library(spec)
            except OSError as exc:
                logger.warning(
                    'could not open the Vulkan loader (%s): %s - commands '
                    'will resolve unbound', spec, exc)
                library = None
            else:
                # ``_name`` is set by ctypes.CDLL.__init__; falls back to
                # repr if a caller hands us a custom DSO-like object.
                logger.info('attached library: %s',
                            getattr(library, '_name', None) or repr(library))
            self._library = library
            # Set last, and under the lock: a reader that sees this flag
            # must also see the handle, or it would bind against None.
            self._library_resolved = True
            return library

    def attach_library(self, library):
        """Attach a Vulkan loader DSO so subsequent command forces bind.

        ``library`` may be ``True`` (autodetect), a filename / path
        string (passed to :class:`ctypes.CDLL`), or an already-open
        :class:`ctypes.CDLL` instance. Unlike the deferred path this is
        eager and *does* raise on a bad path — it's an explicit act, so
        silence would be unhelpful. Attaching after commands have
        already been forced won't retroactively bind them — call
        :meth:`rebind_commands` in that case.
        """
        with self._lock:
            library = self._open_library(library)
            self._library = library
            self._library_spec = library
            self._library_resolved = True
        logger.info('attached library: %s',
                    getattr(library, '_name', None) or repr(library))
        return library

    def rebind_commands(self):
        """Walk the cache and bind every already-forced :class:`CommandSignature`."""
        library = self._ensure_library()
        if library is None:
            return 0
        bound = 0
        for value in self._cache.values():
            if isinstance(value, CommandSignature) and value._fn is None:
                if value.bind(library) is not None:
                    bound += 1
        logger.info('rebound %d command(s) against %s',
                    bound, getattr(library, '_name', None) or repr(library))
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


def build_registry(source=None, *, library=None, api='vulkan'):
    """Construct a :class:`VkRegistry` from a Vulkan XML registry.

    - ``source`` is ``None`` / ``'main'`` (Khronos main branch), a
      version like ``'1.4.359'``, a filesystem path, or an
      :class:`xml.etree.ElementTree.Element`. See
      :func:`volkano.xml_source.resolve_source`.
    - ``library`` is the loader spec — ``None`` (attach nothing),
      ``True`` (autodetect), a path, or an open CDLL. It is *held*, not
      opened: see :meth:`VkRegistry._ensure_library`.

    This module deliberately knows nothing about the ``.pyi`` stub.
    Keeping the stub out of the build path is what makes "a registry you
    construct yourself never writes to your source tree" a structural
    property rather than a flag a caller could get wrong — only the
    package facade syncs the stub, via :func:`volkano.stub.sync_stub`.
    """
    from .xml_parser import parse_registry
    import time
    t0 = time.perf_counter()
    data, provenance = parse_registry(source, api=api)
    merged: dict = {**STDLIB, **data}
    registry = VkRegistry(merged, library=library)
    registry._provenance = provenance
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
