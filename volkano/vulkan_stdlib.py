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
  URI or path (see :func:`volkano.xml_source.resolve_source`), defers
  opening the Vulkan loader to the first command *call*, and records the
  provenance of the XML it parsed. It has no ``refresh`` parameter and
  no knowledge of the ``.pyi`` stub.
"""

from __future__ import annotations

import ctypes
import logging
import sys

from .cbase import (
    Handle, Enum, FunctionPointer, Pointer, Array, Struct, Union, friendly,
    decoder,
    char, byte, ubyte,
    int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    float32, float64, size, ssize,
)
from .lazy import Lazy
from .xml_parser import VOID, VOID_P


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

#: C spelling → cbase type, for ``<type category="basetype">`` typedefs
#: (``typedef uint32_t VkSampleMask``), where the underlying name reaches
#: :func:`make_basetype` as a string rather than as a registry lookup.
#:
#: A lookup table, *not* a namespace: these keys are no longer merged
#: into the registry, because a member declared ``uint32_t`` is
#: translated to ``uint32`` by :data:`~volkano.xml_parser.CANONICAL_TYPE_NAMES`
#: before it is ever looked up. Publishing both spellings put two keys on
#: one class and stood ``vk.int`` / ``vk.bool`` / ``vk.float`` in front of
#: the builtins they shadow, for names no one types.
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


#: The same scalars under cbase's own names, so ``vk.float32`` resolves
#: beside ``vk.float``. :data:`PRIMITIVES` is keyed on C spellings
#: because that is what vk.xml writes; these are the names *volkano*
#: writes — what the generated stub annotates every struct field and
#: parameter with, and what :mod:`volkano.cbase` calls them. Someone who
#: has read either and reaches for ``vk.float32`` should not have to
#: discover that the registry files it under ``float``.
#:
#: Merged after PRIMITIVES and before the XML, so a name vk.xml defines
#: still wins — nothing here can shadow a Vulkan type.
SCALARS: dict = {
    'char':    char,
    'byte':    byte,
    'ubyte':   ubyte,
    'int8':    int8,
    'uint8':   uint8,
    'int16':   int16,
    'uint16':  uint16,
    'int32':   int32,
    'uint32':  uint32,
    'int64':   int64,
    'uint64':  uint64,
    'float32': float32,
    'float64': float64,
    'size':    size,
    'ssize':   ssize,
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


def enum_member(group, name, value):
    """One enumerant as a top-level entry — the member, not the bare int.

    ``vk.VK_SUCCESS`` and ``vk.VkResult.VK_SUCCESS`` used to be two
    different objects: the group built an ``IntEnum``, while the
    enumerant emitted alongside it was a plain ``int``. They compared
    equal, so the seam only showed up under ``is`` — which is exactly
    how one reaches for an enum member, and now that commands *return*
    members (see :func:`cbase.decoder`) the temptation is real.

    ``group`` is the forced enum class, so this is where an enumerant
    entry ends up depending on its group; ``value`` is the integer the
    XML computed, kept as the fallback for a name the class doesn't
    carry (an ``api``-filtered enumerant, an alias whose target didn't
    survive). The ``isinstance`` check is what distinguishes "member"
    from "attribute that merely exists" — an alias resolves through it
    to the canonical member, which is the point.
    """
    member = getattr(group, name, None)
    if isinstance(member, group):
        return member
    return value


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
    and configures ctypes argtypes/restype. Binding happens on the first
    ``__call__`` — that is the only operation that genuinely needs a
    loader, so resolving, introspecting or stubbing a command behaves
    like every other registry entry and touches no DSO. A call that
    can't bind raises a descriptive :class:`RuntimeError`.

    ``registry`` is the :class:`VkRegistry` to ask for a loader when
    that first call arrives; ``None`` (the default, and what the
    factories get in tests) means "never bindable". The reference is
    strong — registry and cache already form a cycle through every
    other forced value, and none of it defines ``__del__``.

    ``_unbound`` records *why* binding didn't happen, and stays ``None``
    until something actually tries — an untried signature is not a
    failed one. Three causes look identical from the call site but need
    very different fixes, and guessing at the wrong one sends people
    hunting for a ``vkGetInstanceProcAddr`` problem when their loader
    simply isn't installed.
    """

    __slots__ = ('name', 'rettype', 'params', 'attrs',
                 '_fn', '_unbound', '_registry')

    #: ``_unbound`` values. ``NO_LIBRARY``: the registry was built with
    #: ``library=None``, so there was nothing to open. ``OPEN_FAILED``: a
    #: loader was requested but ``ctypes.CDLL`` refused it (a warning was
    #: already logged). ``NOT_EXPORTED``: the loader opened fine and
    #: simply doesn't export this symbol. ``None``: no bind attempted yet.
    NO_LIBRARY = 'no-library'
    OPEN_FAILED = 'open-failed'
    NOT_EXPORTED = 'not-exported'

    def __init__(self, name, rettype, params, attrs, registry=None):
        self.name = name
        self.rettype = rettype
        self.params = params
        self.attrs = attrs
        self._fn = None
        self._unbound = None
        self._registry = registry

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
        # ``decoder`` wins where it applies (VkResult, and nothing else
        # in the registry today): ``friendly`` alone would hand back the
        # backing scalar, so vkCreateInstance's failure arrived as
        # ``int32(-3)`` rather than as
        # ``VkResult.VK_ERROR_INITIALIZATION_FAILED``. ctypes takes a
        # plain callable in the restype slot and feeds it the returned C
        # int, which costs a fraction of what the equivalent ``errcheck``
        # hop would, and skips building the scalar wrapper nobody wanted.
        fn.restype = decoder(self.rettype) or friendly(self.rettype)
        self._fn = fn
        logger.debug('bound %s', self.name)
        return fn

    def __call__(self, *args):
        # The bound path costs one attribute load and a branch — the
        # same check that used to only raise now also carries the
        # deferred bind, so nothing was added to the hot path.
        fn = self._fn
        if fn is None:
            fn = self._bind_now()
        return fn(*args)

    def _bind_now(self):
        """Cold path: realise the registry's loader and resolve the symbol.

        Reached once per command — after a successful bind ``_fn`` short-
        circuits it forever. Reached again on every call for a command
        that *failed* to bind, which is deliberate: a loader attached
        after the first failed call (or one that gained the export)
        binds on the next call, with no :meth:`VkRegistry.rebind_commands`
        needed.

        ``_ensure_library`` is reached through :func:`getattr` because a
        signature may hold no registry at all, or a plain :class:`Lazy`
        (both happen in tests).
        """
        registry = self._registry
        ensure = getattr(registry, '_ensure_library', None)
        library = ensure() if ensure is not None else None
        if library is None:
            # A loader that was asked for and couldn't be opened is a
            # different problem from one that was never asked for; the
            # warning naming the path is already in the log.
            self._unbound = (self.OPEN_FAILED
                             if getattr(registry, '_library_spec', None) is not None
                             else self.NO_LIBRARY)
        elif self.bind(library) is not None:
            return self._fn
        raise RuntimeError(
            f"Command {self.name!r} is not bound: {self._reason()}")

    def _reason(self):
        if self._unbound == self.NOT_EXPORTED:
            return ("the Vulkan loader doesn't export this entry point. That's "
                    "normal for extension commands - fetch it at runtime via "
                    "vkGetInstanceProcAddr / vkGetDeviceProcAddr.")
        if self._unbound == self.OPEN_FAILED:
            return ("the Vulkan loader could not be opened (a warning naming "
                    "the path was logged when it was first needed). Install a "
                    "Vulkan runtime, or point VOLKANO_LIBRARY at the loader.")
        return ("this registry was built with library=None, so there was no "
                "loader to bind against. Use volkano.registry(..., "
                "library=True), or call attach_library(...) and call again.")

    def __repr__(self):
        if self._fn is not None:
            state = 'bound'
        elif self._unbound is None:
            # Binding is deferred to the first call, so "unbound" here
            # would read as a diagnosis nothing has actually made yet.
            state = 'unbound, binds on call'
        else:
            state = f'unbound: {self._unbound}'
        return f'<CommandSignature {self.name} ({state})>'


def make_command(registry, name, rettype, params=(), attrs=None):
    """``<command>`` — produces a :class:`CommandSignature`.

    Pure data: the signature records the registry to ask for a loader
    later and opens nothing. Forcing a command is therefore exactly as
    cheap and as driver-independent as forcing a struct or an enum;
    :meth:`CommandSignature._bind_now` realises a deferred ``library=``
    spec on the first *call*, which is the only operation that needs
    one.
    """
    return CommandSignature(name, rettype, list(params), dict(attrs or {}),
                            registry)


# ---------------------------------------------------------------------------
# VkRegistry — attribute-style access + DSO handle
# ---------------------------------------------------------------------------

class VkRegistry(Lazy):
    """Vulkan-flavoured :class:`Lazy` with attribute access and lazy binding.

    Mirrors :class:`lazyregistry.vulkan_stdlib.VkRegistry` — same
    ``__getattr__`` semantics, same ``attach_library`` /
    ``rebind_commands`` surface, same ``.ptr`` / ``.ref`` instance
    shortcuts wired into every struct/union/handle class produced by
    the factories. Beyond the parent kernel, the one behavioural
    difference is *when* the loader opens: here nothing but an actual
    command call triggers it, so the loader plays no part in
    resolution.
    """

    __slots__ = ('_library', '_library_spec', '_library_resolved', '_provenance')

    def __init__(self, data, *, library=None):
        super().__init__(data)
        self._library = None
        # The spec is held, not realised: opening the loader is deferred
        # to the first *command call*, so the whole registry — commands
        # included — resolves on a machine with no Vulkan driver
        # installed. Building a registry is a parsing job; it shouldn't
        # need a GPU stack, and neither should reading a signature back.
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

        Never raises. Callers are :meth:`CommandSignature._bind_now` and
        :meth:`rebind_commands`, neither of which wants an ``OSError``
        from three frames down: the call site raises its own
        :class:`RuntimeError` carrying :meth:`CommandSignature._reason`,
        which names the actual fix, and the bulk rebind reports a count.
        The path that failed is in the warning logged here.
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
        """Attach a Vulkan loader DSO for subsequent command calls to bind against.

        ``library`` may be ``True`` (autodetect), a filename / path
        string (passed to :class:`ctypes.CDLL`), or an already-open
        :class:`ctypes.CDLL` instance. Unlike the deferred path this is
        eager and *does* raise on a bad path — it's an explicit act, so
        silence would be unhelpful. Attaching at any point before a
        command is called is enough, however many commands have been
        forced by then; :meth:`rebind_commands` only matters if you want
        them bound *now* rather than on first call.
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
        """Eagerly bind every already-forced :class:`CommandSignature`.

        Not required for correctness — each command binds itself on its
        first call. This is the bulk, up-front version: it opens the
        loader, walks the cache, and returns how many symbols resolved,
        which makes it the way to answer "how much of this registry can
        this loader actually dispatch?" without calling anything.
        """
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
        """Attribute access over the registry, forcing on first touch.

        Every failure leaves here as :class:`AttributeError`, including
        an entry that exists but cannot be *built*. vk.xml declares the
        window-system and video types — ``HINSTANCE``, ``wl_display``,
        ``StdVideoH264SequenceParameterSet`` — without defining them,
        leaving each to a platform header volkano does not read, so the
        structs referencing them raise ``KeyError`` from the kernel.
        Letting that out of attribute access breaks the protocol every
        introspecting tool depends on: ``hasattr`` raises instead of
        answering, ``getattr(vk, name, default)`` raises instead of
        returning the default, and anything walking :func:`dir` dies on
        the first one.

        Forcing through ``registry(name)`` is unchanged and still raises
        the ``KeyError`` naming the missing type, for callers who would
        rather have the failure than the absence.
        """
        # Only invoked on normal-lookup miss, so kernel attributes
        # (``_data``, ``_cache``, ``_library``) are never intercepted.
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._data:
            raise AttributeError(name)
        try:
            return self(name)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else '(unknown)'
            raise AttributeError(
                f'{name!r} exists in this registry but cannot be built: it '
                f'references {missing!r}, which is not defined here. vk.xml '
                f'leaves platform and video types to their own headers, so '
                f'the entries that use them are unreachable.') from exc

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

    - ``source`` is ``None`` (Khronos's main branch), an http(s) URL,
      a ``file://`` URI, a filesystem path, or an
      :class:`xml.etree.ElementTree.Element`. See
      :func:`volkano.xml_source.resolve_source`.
    - ``library`` is the loader spec — ``None`` (attach nothing),
      ``True`` (autodetect), a path, or an open CDLL. It is *held*, not
      opened: see :meth:`VkRegistry._ensure_library`.

    This module deliberately knows nothing about the ``.pyi`` stub.
    Keeping the stub out of the build path is what makes "building a
    registry never writes to your source tree" a structural property
    rather than a flag a caller could get wrong — the stub is written
    by ``python -m volkano update`` and by nothing else.
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
STDLIB.update(SCALARS)
STDLIB.update({
    # The two slots with no name a user would type. Underscore-prefixed
    # (see xml_parser.VOID / VOID_P) so the thunk graph resolves them
    # while attribute access, dir() and the stub pass them by: C ``void``
    # is the absence of a type, and ``void *`` is type-erased.
    VOID:               None,
    VOID_P:             Pointer[None],
})
STDLIB.update({
    'ref':              ref,
    'array':            array,
    'sizeof':           sizeof,
    'make_handle':      make_handle,
    'make_basetype':    make_basetype,
    'make_bitmask':     make_bitmask,
    'make_enum':        make_enum,
    'enum_member':      enum_member,
    'make_funcpointer': make_funcpointer,
    'make_struct':      make_struct,
    'make_union':       make_union,
    'make_command':     make_command,
})
