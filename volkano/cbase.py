import ctypes
import enum
import operator
import sys
import weakref
from _ctypes import Structure as CStructureBase
from _ctypes import Union as CUnionBase
from _ctypes import _Pointer as CPointerBase
from _ctypes import Array as CArrayBase
from _ctypes import _SimpleCData as CDataBase
from _ctypes import CFuncPtr as CFuncPtrBase
from typing import Generic, TypeVar

# Type parameters for :class:`Pointer` and :class:`Array`. Declared so
# ``Pointer[X]`` / ``Array[X]`` are valid generic aliases for static
# analysers (notably inside the generated ``__init__.pyi`` stub, where pyright
# enforces strict type-expression rules and would otherwise reject the
# subscript). At runtime the custom ``__class_getitem__`` on each class
# still governs and returns the concrete subclass — the ``Generic`` base
# only carries static intent.
_PtrT = TypeVar('_PtrT')
_ArrT = TypeVar('_ArrT')

class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class Mixin:
    _pointer_type = None
    def __repr__(self):
        return f"{self.__class__.__qualname__}"

    @classmethod
    def from_param(cls, *obj):
        """ctypes argument-coercion hook + multi-arg constructor shortcut.

        ctypes calls this with a single positional value when adapting
        a foreign-function argument — ``*obj`` captures it as a 1-tuple
        and the body forwards via ``cls(*obj)``. Direct callers can
        also spread positional args (``Cls.from_param(x, y, z)``) which
        flow straight into the constructor without any tuple-wrapping
        ceremony.
        """
        if len(obj) == 1 and isinstance(obj[0], cls):
            return obj[0]
        return cls(*obj)

    @property
    def ref(self):
        return ctypes.byref(self)

    @property
    def ptr(self):
        return Pointer.from_ref(self)

    @classproperty
    def size(self):
        return ctypes.sizeof(self)

    @classproperty
    def pointer_type(cls):
        # Read the cache via ``cls.__dict__`` rather than attribute
        # lookup. CPython 3.9–3.12 ctypes Union metaclass fails to
        # invalidate the type-attribute cache when a class attribute is
        # rewritten, so a write-then-attribute-read within the same
        # frame returns the stale value (verified on 3.9.20 and 3.10.1;
        # 3.7.16 and 3.13+ are unaffected). Dict access bypasses the
        # cache, and we return the local var to avoid relying on the
        # read-back at all.
        cached = cls.__dict__.get('_pointer_type')
        if cached is not None:
            return cached
        ptype = type(f"Pointer[{cls.__name__}]", (Pointer,), {"_type_": cls})
        cls._pointer_type = ptype  # write reaches __dict__ on all versions
        return ptype

    def reinterpret(self, new_type):
        """View this object's bytes as ``new_type`` (dereferencing reinterpret).

        Takes the address of ``self``, casts a pointer to
        ``Pointer[new_type]``, and dereferences — so the returned
        value sits on the same memory as ``self`` but is typed as
        ``new_type``. Closer to ``reinterpret_cast<NewType&>(obj)``
        in C++ than to a pure pointer cast: ``self`` must be
        accessible memory (NULL or freed buffers will segfault).

        Distinct from :meth:`Pointer.cast`, which is a pure
        pointer-type cast — no dereference, NULL-safe.
        """
        return ctypes.cast(self.ref, Pointer[new_type]).contents


class StructureMixin(Mixin):
    """Naive lazy ``pointer_type`` cache for Struct/Union subclasses.

    Inherits :attr:`Mixin.pointer_type` (which lazily creates and caches
    ``Pointer[cls]`` on first access). The only override here is
    ``__init_subclass__``: it gives every subclass its *own*
    ``_pointer_type = None`` slot so the cache isn't inherited from the
    parent (without this, ``B(A).pointer_type`` would silently return
    the cached ``Pointer[A]``).

    Pointer creation must stay lazy — eager creation in
    ``__init_subclass__`` would propagate into Pointer's own subclass
    hook and produce an unbounded chain of ``Pointer[Pointer[...]]``.

    Historical note: an earlier version stored the lazy cache via a
    generator (``pointer_yielder``) to avoid writing a ctypes-derived
    class onto a Union/Structure subclass — Union ``__setattr__``
    behaved inconsistently across CPython versions. The naive version
    below writes ``cls._pointer_type = Pointer[cls]`` on first access;
    the test at the bottom of this module checks whether that
    assignment persists on the current interpreter.
    """

    @classmethod
    def create_empty(cls, name):
        """Create a subclass with ``_fields_`` left unset.

        First half of the self-cycle-safe pattern: plant the empty
        class in your registry/cache *before* descending the field
        type graph, so a field that re-asks for the type being built
        (intrusive linked-list ``next``, parent pointers, the
        ``VkBaseInStructure.pNext`` shape) finds the in-progress class
        instead of re-entering this factory. Once the field types have
        resolved, call :meth:`set_fields` to finalise the layout.
        """
        return type(name, (cls,), {})

    @classmethod
    def set_fields(cls, target, fields):
        """Assign ``_anonymous_`` then ``_fields_`` on ``target``.

        ``fields`` is a sequence of entries in any of these shapes:

        - ``(name, type)`` — ordinary field.
        - ``(name, type, bits)`` with ``isinstance(bits, int)`` —
          C bitfield of ``bits`` width.
        - ``(name, type, '_anonymous_')`` — anonymous member; the
          referenced struct/union's fields are reachable directly on
          ``target``.
        - :class:`Param` — ``(param.name, param.type)`` shape.

        Each field type passes through :func:`friendly` so
        :class:`enum.IntEnum` / :class:`enum.IntFlag` are substituted
        with their backing scalar. ``_anonymous_`` is assigned *before*
        ``_fields_`` because ctypes wires the anonymous-member lookup
        path at the moment ``_fields_`` is set.
        """
        ctypes_fields: list = []
        anon_names: list[str] = []
        for entry in fields:
            if isinstance(entry, Param):
                ctypes_fields.append((entry.name, friendly(entry.type)))
                continue
            if len(entry) == 3:
                fname, ftype, third = entry
                ftype = friendly(ftype)
                if isinstance(third, int):
                    ctypes_fields.append((fname, ftype, third))
                else:
                    ctypes_fields.append((fname, ftype))
                    if third == '_anonymous_':
                        anon_names.append(fname)
            else:
                fname, ftype = entry
                ctypes_fields.append((fname, friendly(ftype)))
        if anon_names:
            target._anonymous_ = tuple(anon_names)
        target._fields_ = ctypes_fields
        target._install_charptr_setattr()
        return target

    @classmethod
    def create(cls, name, members):
        """One-shot factory: :meth:`create_empty` then :meth:`set_fields`.

        Use the two-step form directly when fields may reference the
        type being constructed; this shorthand is fine for ordinary
        non-self-referential structs.
        """
        return cls.set_fields(cls.create_empty(name), members)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Per-subclass slot — break attribute-inheritance of the cache.
        # Done with a plain None, not a ctypes class, to avoid any
        # metaclass-of-ctypes-Union/Structure quirk at definition time.
        cls._pointer_type = None
        # Body-defined structs (``class S(Struct): _fields_ = [...]``)
        # have their layout in place here; two-step builds finalise it in
        # :meth:`set_fields`. Both funnel through the same installer.
        if '_fields_' in cls.__dict__:
            cls._install_charptr_setattr()

    @classmethod
    def _install_charptr_setattr(cls):
        """Install the ``bytes`` → ``const char *`` ``__setattr__`` iff
        ``cls`` has at least one ``Pointer[char]`` field.

        ``bytes`` assigned to such a field — by attribute *or* constructor
        kwarg, since ctypes routes both through ``__setattr__`` — is
        copied into an owned NUL-terminated buffer and stored as the
        decayed ``char *``, the buffer kept alive by the object's ctypes
        ``_objects``. This mirrors :meth:`Pointer.from_param` for
        arguments, so ``bytes`` behaves as ``const char *`` everywhere.

        The override is a Python-level ``__setattr__`` (~5x ctypes'
        C-level field set), so it is installed *only* on classes that
        have such a field — a small, mostly init-time minority. Every
        other struct/union is left untouched and keeps the C fast path.
        """
        charptr = {}
        for entry in getattr(cls, '_fields_', ()) or ():
            fname, ftype = entry[0], entry[1]
            if isinstance(ftype, type) and issubclass(ftype, Pointer) \
                    and getattr(ftype, '_type_', None) is char:
                charptr[fname] = ftype
        if charptr:
            cls._charptr_fields = charptr
            cls.__setattr__ = _charptr_setattr


def _unbox(x):
    """Coerce an arithmetic operand to its underlying Python scalar.

    :class:`ArithmeticMixin` calls this on both operands of every
    binary op. Anything that exposes a ``value`` attribute (cbase
    scalars, ctypes scalars, pointers) contributes its wrapped
    value; everything else (plain ``int`` / ``float``, or any
    Python object that already implements numeric ops) is returned
    unchanged.

    Uses ``getattr(x, 'value', x)`` rather than ``try``/``except
    AttributeError``: the C-level default lookup is ~3.4x faster on
    the plain-``int`` operand path (no Python exception frame raised
    and caught for ``5`` in ``int32(x) + 5``) and measured identical
    on the cbase-scalar path — so there is no "fast path" cost to
    paying the attribute-presence check, contrary to an earlier
    assumption.
    """
    return getattr(x, 'value', x)


# Promotion rank classification by ctypes ``_type_`` struct code.
# These are CPython's documented struct-module codes that ctypes uses
# internally; defining them as module-level frozensets keeps the rank
# lookup at one dict-hash per operand. Codes:
#  - 'b' c_byte  'B' c_ubyte
#  - 'h' c_int16  'H' c_uint16
#  - 'i' c_int    'I' c_uint
#  - 'l' c_long   'L' c_ulong
#  - 'q' c_int64  'Q' c_uint64
#  - 'n' c_ssize_t  'N' c_size_t
#  - '?' c_bool
#  - 'f' c_float  'd' c_double  'g' c_longdouble
#  - 'P' c_void_p (and Handle subclasses thereof)
_UNSIGNED_INT_CODES = frozenset({'B', 'H', 'I', 'L', 'Q', 'N', '?'})
_FLOAT_CODES = frozenset({'f', 'd', 'g'})
_POINTER_CODES = frozenset({'P'})


def _rank(t):
    """Promotion rank for a ctypes scalar type ``t``.

    Returns a sortable ``(kind, size, unsigned)`` tuple where:

    - ``kind`` is ``3`` for pointer-like, ``2`` for float, ``1`` for
      integer (or unknown).
    - ``size`` is :func:`ctypes.sizeof` — wider operands win.
    - ``unsigned`` is ``1`` for unsigned integers else ``0``, so on
      a same-size integer tie the unsigned type wins (matching C's
      usual arithmetic conversions).

    Pointer-like operands ("kind 3") dominate all numerics — so
    ``Handle(0x1000) + int64(8)`` stays a :class:`Handle`, preserving
    pointer arithmetic identity. User-defined CData subclasses with
    non-standard ``_type_`` codes fall back to the signed-integer
    rank and can override binary ops on the subclass if a custom
    promotion rule is needed.
    """
    code = getattr(t, '_type_', None)
    size = ctypes.sizeof(t)
    if code in _POINTER_CODES:
        return (3, size, 0)
    if code in _FLOAT_CODES:
        return (2, size, 0)
    is_unsigned = code in _UNSIGNED_INT_CODES
    return (1, size, 1 if is_unsigned else 0)


def _result_type(self_type, other):
    """Pick the result type for a binary arithmetic op.

    If ``other`` is a :class:`CData` instance, promote to the
    higher-:func:`_rank` of the two declared types (C's usual
    arithmetic conversions: pointer over numeric, float over int,
    wider over narrower, unsigned over signed on tie). Otherwise
    return ``self_type`` unchanged — a plain Python int / float
    operand carries no declared C type to promote toward, so the
    cbase type on the other side governs the result.
    """
    if not isinstance(other, CData):
        return self_type
    other_type = type(other)
    if other_type is self_type:
        return self_type
    return self_type if _rank(self_type) >= _rank(other_type) else other_type


def _binop(op):
    """Forward binary-op factory — result wrapped in the promoted type.

    Promotion follows :func:`_result_type`; if neither operand is a
    cbase scalar (only possible via direct method invocation, since
    descriptor lookup ensures ``self`` is one of these on normal
    operator use) the result still wraps in ``type(self)``.
    """
    def method(self, other):
        rtype = _result_type(type(self), other)
        return rtype(op(_unbox(self), _unbox(other)))
    method.__name__ = f'__{op.__name__}__'
    method.__qualname__ = f'ArithmeticMixin.{method.__name__}'
    return method


def _rbinop(op):
    """Reflected binary-op factory — used when self is on the *right*.

    Same promotion as :func:`_binop` (the result type depends on the
    declared types, not the argument order), with the actual op
    applied in the original direction (``op(other_value, self_value)``)
    so subtraction / division / shifts don't silently transpose.
    """
    def method(self, other):
        rtype = _result_type(type(self), other)
        return rtype(op(_unbox(other), _unbox(self)))
    method.__name__ = f'__r{op.__name__}__'
    method.__qualname__ = f'ArithmeticMixin.{method.__name__}'
    return method


def _ibinop(op):
    """In-place binary-op factory — mutates ``self.value`` and returns self.

    Implements C's compound-assignment semantics:
    ``x op= y``  ≡  ``x = (typeof x)(x op y)``. The compute step
    follows Python's usual numeric promotion (so ``int32 += float32``
    computes in float), and the assignment step truncates the
    promoted value back to ``self``'s storage type — matching C's
    implicit conversion at the assignment boundary.

    For the lossy float→int path we apply ``int(result)`` (truncate
    toward zero) before handing to ctypes, since ``c_intXX.value =
    1.5`` raises whereas C's ``int32_t x; x += 1.5f`` silently
    yields ``x + 1``. Lossy int→int truncation is left to ctypes (it
    wraps signed/unsigned per the type's storage range), and the
    int→float path widens losslessly.
    """
    def method(self, other):
        result = op(_unbox(self), _unbox(other))
        if isinstance(result, float):
            code = getattr(type(self), '_type_', None)
            if code not in _FLOAT_CODES:
                result = int(result)
        self.value = result
        return self
    method.__name__ = f'__i{op.__name__}__'
    method.__qualname__ = f'ArithmeticMixin.{method.__name__}'
    return method


class ArithmeticMixin:
    """Numeric + bitwise operators returning a new ``type(self)`` instance.

    Mixed into :class:`CData` so every cbase scalar (and the
    pointer-like :class:`VoidPointer` / :class:`Handle`) supports
    Python's full arithmetic and bitwise operator protocol:

    - Forward binary ops (``+`` ``-`` ``*`` ``/`` ``//`` ``%`` ``**``
      ``<<`` ``>>`` ``&`` ``|`` ``^``) construct a fresh
      ``type(self)`` carrying the Python-arithmetic result.
    - Reflected ops (``__radd__`` etc.) preserve the cbase type when
      a plain Python primitive is on the left: ``5 + int32(3)``
      → ``int32(8)``.
    - In-place ops (``+=`` ``-=`` …) mutate ``self.value`` directly
      and return ``self`` — the cheap path for accumulators that
      avoid allocating a fresh wrapper per iteration.
    - Unary ``-`` / ``+`` / ``abs`` / ``~`` likewise rewrap.

    Operand coercion goes through :func:`_unbox`, so any of
    ``int32(5) + int32(3)``, ``int32(5) + 3``, and ``3 + int32(5)``
    produce the same ``int32(8)`` result.

    Edge case: when the underlying Python op promotes the result
    type (true division of two ints yields a float; negative-exponent
    ``**`` likewise), the wrap-up call to ``type(self)(value)`` hits
    the ctypes scalar's constructor with the promoted value. ctypes
    raises ``TypeError`` if the value isn't representable in the
    storage type (e.g. ``int32(5) / int32(2)`` because ``2.5`` can't
    live in a C int) — switch to :class:`float32` / :class:`float64`
    when the operation can produce a non-integer result.
    """

    __slots__ = ()

    __add__       = _binop(operator.add)
    __radd__      = _rbinop(operator.add)
    __iadd__      = _ibinop(operator.add)
    __sub__       = _binop(operator.sub)
    __rsub__      = _rbinop(operator.sub)
    __isub__      = _ibinop(operator.sub)
    __mul__       = _binop(operator.mul)
    __rmul__      = _rbinop(operator.mul)
    __imul__      = _ibinop(operator.mul)
    __truediv__   = _binop(operator.truediv)
    __rtruediv__  = _rbinop(operator.truediv)
    __itruediv__  = _ibinop(operator.truediv)
    __floordiv__  = _binop(operator.floordiv)
    __rfloordiv__ = _rbinop(operator.floordiv)
    __ifloordiv__ = _ibinop(operator.floordiv)
    __mod__       = _binop(operator.mod)
    __rmod__      = _rbinop(operator.mod)
    __imod__      = _ibinop(operator.mod)
    __pow__       = _binop(operator.pow)
    __rpow__      = _rbinop(operator.pow)
    __lshift__    = _binop(operator.lshift)
    __rlshift__   = _rbinop(operator.lshift)
    __ilshift__   = _ibinop(operator.lshift)
    __rshift__    = _binop(operator.rshift)
    __rrshift__   = _rbinop(operator.rshift)
    __irshift__   = _ibinop(operator.rshift)
    __and__       = _binop(operator.and_)
    __rand__      = _rbinop(operator.and_)
    __iand__      = _ibinop(operator.and_)
    __or__        = _binop(operator.or_)
    __ror__       = _rbinop(operator.or_)
    __ior__       = _ibinop(operator.or_)
    __xor__       = _binop(operator.xor)
    __rxor__      = _rbinop(operator.xor)
    __ixor__      = _ibinop(operator.xor)

    def __neg__(self):    return type(self)(-_unbox(self))
    def __pos__(self):    return type(self)(+_unbox(self))
    def __abs__(self):    return type(self)(abs(_unbox(self)))
    def __invert__(self): return type(self)(~_unbox(self))


class CData(CDataBase, Mixin, ArithmeticMixin):
    _type_ = "P"

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.value})"

    def __int__(self):
        """Auto-unbox to Python int via :attr:`value`.

        ctypes' simple-data classes unbox automatically when read as
        a struct field of their *exact* ctypes type but stop unboxing
        for subclasses — making struct-field reads on cbase scalars
        (:class:`int32`, :class:`uint32`, …) return wrapper instances
        rather than Python ints. Defining ``__int__`` lets the scalar
        slot into anywhere Python expects an :func:`int` coercion:
        arithmetic with plain ints, comparison, format strings.
        :func:`hex` / :func:`bin` / :func:`oct` and slicing/range use
        :meth:`__index__` instead — see below. ``None`` (uninitialised
        :class:`VoidPointer` / :class:`Handle` instances) coerces to
        ``0``, matching C's NULL=0 convention.
        """
        v = self.value
        return 0 if v is None else int(v)

    def __float__(self):
        """Auto-unbox to Python float via :attr:`value`.

        Mirror of :meth:`__int__` for the float-bearing scalars
        (:class:`float32`, :class:`float64`); same NULL→0.0 fallback
        path for pointer-like subclasses.
        """
        v = self.value
        return 0.0 if v is None else float(v)

    def __index__(self):
        """Integer-protocol coercion — enables :func:`hex`, slicing, etc.

        :func:`hex` / :func:`bin` / :func:`oct`, slice indices, and
        ``range()`` positional arguments all use :meth:`__index__`
        rather than :meth:`__int__`. For integer-bearing CData
        subclasses (:class:`int32`, :class:`uint32`, :class:`Handle`,
        …) the coercion is lossless as the protocol requires. For
        float subclasses (:class:`float32`, :class:`float64`) it
        falls through to ``int(self.value)`` — technically a
        violation of ``__index__``'s "no precision loss" rule, but
        defined on the shared base anyway to avoid splitting the
        scalar hierarchy for what is rarely a meaningful operation on
        a float in the first place.
        """
        v = self.value
        return 0 if v is None else int(v)

    # ----- Comparison + hash + bool protocol ----------------------------
    #
    # ctypes' simple-data classes auto-unbox to Python scalars on
    # *exact-type* struct-field reads, but stop unboxing for subclasses.
    # That means a cbase scalar field readback yields a wrapper instance
    # rather than ``int`` / ``float``. Defining the comparison /
    # ``__hash__`` / ``__bool__`` trio here makes the wrapper transparent
    # at every site where user code expects an unboxed scalar: ``flags &
    # 0x10 != 0`` works, ``info.x == 5`` works, ``if info.flag:`` works,
    # ``handle in some_set`` works, ``some_dict[handle] = …`` works.
    #
    # Together with :meth:`__int__` / :meth:`__float__` / :meth:`__index__`
    # and :class:`ArithmeticMixin`, the wrapper is operationally
    # indistinguishable from its unwrapped ``self.value`` for the
    # numeric protocols that Python idioms reach for. Identity (``is``)
    # and class-introspection (``type(x) is int``) still see the
    # wrapper — by design; the wrapper carries declared C-type
    # information that users do sometimes rely on (e.g. for
    # ``ctypes._fields_`` round-trips).

    def __eq__(self, other):
        if isinstance(other, CData):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, CData):
            return self.value != other.value
        return self.value != other

    def __lt__(self, other):
        return self.value < (other.value if isinstance(other, CData) else other)

    def __le__(self, other):
        return self.value <= (other.value if isinstance(other, CData) else other)

    def __gt__(self, other):
        return self.value > (other.value if isinstance(other, CData) else other)

    def __ge__(self, other):
        return self.value >= (other.value if isinstance(other, CData) else other)

    def __hash__(self):
        """Hash via :attr:`value` so wrappers and unboxed scalars are dict-equivalent.

        Required for :class:`dict` / :class:`set` membership to behave
        consistently with the ``__eq__`` we just defined: ``hash(a) ==
        hash(b)`` whenever ``a == b``. ``None``-valued pointer
        subclasses (uninitialised :class:`VoidPointer` / :class:`Handle`)
        hash as ``hash(0)`` to match their NULL=0 numeric coercion.
        """
        v = self.value
        return hash(0 if v is None else v)

    def __bool__(self):
        """Truthiness via :attr:`value` so ``if scalar:`` works as expected.

        ``bool(uint32(0))`` is ``False``, ``bool(uint32(5))`` is
        ``True`` — matches the unboxed-scalar reading the user would
        get from a raw ctypes type. Without this, Python's default
        ``__bool__`` returns ``True`` for any non-None instance and
        ``if info.flag:`` would incorrectly be true for a zero flag.
        """
        return bool(self.value)


# Cache of cbase ``Pointer`` subclasses built for *raw* ctypes element
# types (e.g. the blessed :data:`c_char_p`). Mixin elements cache their
# pointer on themselves via :attr:`Mixin.pointer_type`; raw ctypes types
# can't carry that slot, so they're keyed here instead — keeping pointer
# identity stable (``Pointer[c_char_p] is Pointer[c_char_p]``).
class Pointer(CPointerBase, Mixin, Generic[_PtrT]):
    _type_ = ctypes.c_void_p

    def __repr__(self):
        val = ctypes.cast(self, VoidPointer).value
        addr = 'NULL' if val is None else f'0x{val:X}'
        return f"{self.__class__.__qualname__}({addr})"

    def __class_getitem__(cls, item):
        """Build the typed Pointer subclass for ``item``.

        Accepts:

        - ``None`` → :class:`VoidPointer`, the canonical ``void *`` slot
          (spelled ``Pointer[None]`` everywhere outside this module).
        - A :class:`Mixin` subclass → its cached ``pointer_type`` (the
          common path — every type the registry produces is a cbase
          :class:`Mixin` type, so this is the branch that fires).
        - A plain ctypes type → :func:`ctypes.POINTER` (ctypes caches it,
          so identity holds). The registry never reaches this branch; it
          exists only for direct ctypes-interop use of ``Pointer[X]``.
        - An :class:`enum.IntEnum` / :class:`enum.IntFlag` → substituted
          via :func:`friendly` first, so ``Pointer[SomeEnum]`` becomes
          a pointer to the backing scalar.
        """
        if item is None:
            return VoidPointer
        item = friendly(item)
        if isinstance(item, type) and issubclass(item, Mixin):
            return item.pointer_type
        return ctypes.POINTER(item)

    @classmethod
    def from_param(cls, value):
        """Pointer-arg coercion — delegate to ctypes' metaclass.

        :meth:`Mixin.from_param` is variadic to support multi-arg
        constructor shortcuts, but for pointer argtypes ctypes'
        standard coercion is what callers actually need:

        - ``byref(struct)`` markers pass through as the canonical
          pointer representation.
        - ``None`` becomes a NULL pointer.
        - Instances of this Pointer type pass through unchanged.

        ``PyCPointerType.from_param`` (the metaclass-level method)
        implements all three at the C level. This override exposes
        it so that a cbase Pointer used as an ``argtype`` accepts
        the same value shapes as a raw :func:`ctypes.POINTER`; the
        Mixin one would otherwise win attribute lookup because
        ``_Pointer.from_param`` lives on the metaclass, not on the
        class itself.

        For ``Pointer[char]`` specifically, a ``bytes`` argument is
        treated as a ``const char *``: it is copied into an owned
        NUL-terminated buffer and the decayed pointer is returned. The
        buffer is kept alive by the returned pointer's ``_objects``, and
        ctypes keeps the ``from_param`` result alive for the call's
        duration — so ``vkFoo(b"name")`` is safe. This mirrors
        :func:`_charptr_setattr` for struct fields.
        """
        if cls._type_ is char and isinstance(value, (bytes, bytearray)):
            return ctypes.cast(_char_buffer(value), cls)
        return type(cls).from_param(cls, value)

    @classmethod
    def from_address(cls, address):
        return ctypes.cast(address, cls)

    @classmethod
    def from_ref(cls, item):
        return cls[item.__class__](item)

    @property
    def value(self):
        return ctypes.cast(self, VoidPointer).value

    def cast(self, new_pointer_type):
        """Cast this pointer to ``new_pointer_type``, same address.

        Pure pointer-type cast — no dereference, no copy, no read of
        the pointed-to bytes. Matches :func:`ctypes.cast` semantics
        and ``static_cast<NewPointerType>(ptr)`` in C++. NULL-safe
        (works on NULL pointers without faulting). Useful for
        assigning a typed pointer to a differently-typed pointer
        slot — :class:`VoidPointer` for type-erased fields like
        Vulkan's ``pNext``, or a typed pointer of unrelated element
        type for reinterpret-cast patterns.

        Distinct from :meth:`Mixin.reinterpret`, which dereferences:
        ``cast`` keeps the pointer-ness, only the declared target
        type changes. Accepts ``new_pointer_type`` as a cbase Pointer
        subclass, a ctypes pointer class, :class:`VoidPointer`, or
        any ctypes type :func:`ctypes.cast` accepts.
        """
        return ctypes.cast(self, new_pointer_type)


class VoidPointer(ctypes.c_void_p, CData):
    """The ``void *`` slot — the concrete ``c_void_p`` subclass that
    ``Pointer[None]`` resolves to.

    Internal to cbase: outside code (the registry, the generated stub,
    the RAII layer) never names ``VoidPointer`` — ``void *`` is always
    spelled ``Pointer[None]``. Its ``__name__`` / ``__qualname__`` are set
    to ``"Pointer[None]"`` just below (a ctypes metaclass ignores a
    class-body ``__name__``), so it *renders* as ``Pointer[None]``
    everywhere: the stub emits ``ct.__name__``; pointer types build their
    name from the element's ``__name__`` (so ``void **`` is
    ``Pointer[Pointer[None]]``); ``__repr__`` uses ``__qualname__``.
    """


VoidPointer.__name__ = "Pointer[None]"
VoidPointer.__qualname__ = "Pointer[None]"


class Handle(ctypes.c_void_p, CData):
    """Tagged opaque pointer — typedef'd ``c_void_p`` with class identity.

    Generic C-binding base for HANDLE-style types (Win32 HANDLE, X11
    XID, OpenGL object names, Vulkan handles). Each named handle is a
    distinct subclass so function-signature identity is preserved even
    though the storage is always pointer-sized opaque int.
    """

    @classmethod
    def create(cls, name):
        """Create a named ``Handle`` subclass — a C typedef equivalent."""
        return type(name, (cls,), {})


class Array(CArrayBase, Mixin, Generic[_ArrT]):
    _type_ = ctypes.c_void_p
    _length_ = 0

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(str(i) for i in self)})"

    def __class_getitem__(cls, args):
        """Build the fixed-size Array subclass for ``(basetype, length)``.

        Element-type handling mirrors :meth:`Pointer.__class_getitem__`:

        - ``basetype=None`` → :class:`VoidPointer` (array of opaque
          pointers; ``void[N]`` itself isn't well-defined in C, so we
          collapse to ``void *[N]``, the common practical case).
        - :class:`enum.IntEnum` / :class:`enum.IntFlag` → substituted
          via :func:`friendly` (ctypes can't store an enum class).
        - Plain ctypes types and :class:`Mixin` subclasses pass through.

        ``length`` accepts a :class:`CData` scalar (its ``.value`` is
        used) or anything ``int``-coercible.
        """
        basetype, length = args
        if basetype is None:
            basetype = VoidPointer
        basetype = friendly(basetype)
        if isinstance(length, CData):
            length = length.value
        length = int(length)
        name = f"{cls.__name__}[{getattr(basetype, '__name__', repr(basetype))}, {length}]"
        return type(name, (Array,), {"_type_": basetype, "_length_": length})

class Param:
    def __init__(self, name, type):
        self.name = name
        self.type = type

class FunctionPointer(CFuncPtrBase, Mixin):
    """C function-pointer factory.

    The base class is a namespace; :meth:`create` is the factory.
    The returned class is a :class:`Mixin`-bearing subclass of the
    ctypes ``_CFuncPtr`` class that :func:`ctypes.CFUNCTYPE` /
    :func:`ctypes.WINFUNCTYPE` produces — so every cbase
    function-pointer type carries the same surface as every other
    cbase type (``.ref`` / ``.size`` / ``Pointer[X]`` / ``Array[X, N]``).

    Calling convention selection and signature deduplication are
    delegated to the ctypes factory; this class just grafts Mixin
    onto the result so the API stays uniform with :class:`Handle`,
    :class:`Struct`, etc.
    """

    _argtypes_ = ()
    _restype_ = None
    _flags_ = 1

    @classmethod
    def create(cls, name, rettype, params, *, stdcall=False):
        """Build a typed function-pointer class.

        ``params`` accepts :class:`Param` objects (``.type`` is read)
        or ``(field_name, type)`` tuples/lists. Each arg type and
        ``rettype`` pass through :func:`friendly` for
        :class:`enum.IntEnum` / :class:`enum.IntFlag` substitution.

        ``stdcall=True`` routes through :func:`ctypes.WINFUNCTYPE` on
        win32 (stdcall on 32-bit x86, no-op on x86_64); on non-win32
        platforms it falls through to :func:`ctypes.CFUNCTYPE` since
        stdcall doesn't exist there.

        The result is ``type(name, (cached_ctypes_class, Mixin), {})``:
        each named typedef is its own Python class, but the ctypes
        factory's signature cache still dedupes the *underlying*
        ``_CFuncPtr`` so two function-pointer types with identical
        signatures share calling-convention machinery via inheritance.
        ``_CFuncPtr.from_param`` precedes ``Mixin.from_param`` in
        MRO, so ctypes' argument-coercion path stays in force — the
        same trick :class:`VoidPointer` / :class:`Handle` use.
        """
        argtypes = []
        for p in params:
            t = p.type if isinstance(p, Param) else p[1]
            argtypes.append(friendly(t))
        rt = friendly(rettype) if rettype is not None else None
        if stdcall and sys.platform == 'win32':
            factory = ctypes.WINFUNCTYPE
        else:
            factory = ctypes.CFUNCTYPE
        base = factory(rt, *argtypes)
        # ctypes' PyCFuncPtrType metaclass requires _flags_ in the
        # class's own namespace (it doesn't walk the MRO for it);
        # copy the three signature attributes from the cached base.
        return type(name, (base, Mixin), {
            '_flags_': base._flags_,
            '_restype_': base._restype_,
            '_argtypes_': base._argtypes_,
        })

    def __class_getitem__(cls, item):
        """Anonymous-typedef shorthand: ``FunctionPointer[ret, [args]]``.

        Type-expression form for one-off callbacks where the typedef
        doesn't need a stable Python name (stub generation, debugging
        identity, etc.). Mirrors :class:`Pointer` / :class:`Array`
        subscript syntax. ``item`` must be a 2-tuple ``(ret_type,
        args_seq)`` where ``args_seq`` is a list or tuple of types —
        the second element is *always* a sequence even for a single
        arg (``FunctionPointer[int32, [int32]]``) and for the no-args
        case (``FunctionPointer[int32, []]``); use :meth:`create`
        instead if you want :class:`Param` objects or stdcall.

        Returns a fresh :class:`Mixin`-bearing subclass of ctypes'
        cached ``_CFuncPtr`` for the signature; the generated class
        name reads as ``FunctionPointer[(arg, ...) -> ret]`` so it's
        legible in tracebacks. :func:`friendly` is applied to
        ``ret_type`` and each arg the same way :meth:`create` does.
        """
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"FunctionPointer subscript expects (ret, args_seq); "
                f"got {item!r}")
        ret, args = item
        if not isinstance(args, (list, tuple)):
            raise TypeError(
                f"FunctionPointer subscript's args element must be a "
                f"list/tuple of types; got {args!r}")
        argtypes = [friendly(a) for a in args]
        rt = friendly(ret) if ret is not None else None
        base = ctypes.CFUNCTYPE(rt, *argtypes)

        def _label(t):
            if t is None:
                return 'void'
            return getattr(t, '__name__', repr(t))

        name = (
            f"FunctionPointer["
            f"({', '.join(_label(a) for a in args)}) "
            f"-> {_label(ret)}]"
        )
        return type(name, (base, Mixin), {
            '_flags_': base._flags_,
            '_restype_': base._restype_,
            '_argtypes_': base._argtypes_,
        })


class Struct(CStructureBase, StructureMixin):
    pass

class Union(CUnionBase, StructureMixin):
    pass



class byte(ctypes.c_byte, CData): pass
class ubyte(ctypes.c_ubyte, CData): pass
class int8(ctypes.c_int8, CData): pass
class uint8(ctypes.c_uint8, CData): pass
class int16(ctypes.c_int16, CData): pass
class uint16(ctypes.c_uint16, CData): pass
class int32(ctypes.c_int32, CData): pass
class uint32(ctypes.c_uint32, CData): pass
class int64(ctypes.c_int64, CData): pass
class uint64(ctypes.c_uint64, CData): pass
class float32(ctypes.c_float, CData): pass
class float64(ctypes.c_double, CData): pass
class size(ctypes.c_size_t, CData): pass
class ssize(ctypes.c_ssize_t, CData): pass
# C ``char``. Kept distinct from the numeric byte wrappers so a
# ``char[N]`` struct field round-trips as Python ``bytes``: ctypes'
# char-array conversion keys off the element's ``'c'`` type code, which
# survives subclassing (unlike ``c_char_p``'s string marshalling, which
# does not — hence that one stays raw, see the re-export note below).
# Used for inline fixed-size strings such as ``deviceName[256]``.
class char(ctypes.c_char, CData): pass


def _char_buffer(data):
    """Owned, NUL-terminated ``char`` buffer holding ``data`` (bytes).

    A ``char[len+1]`` array with ``data`` copied in and the trailing NUL
    left zero. Backs the ``bytes`` → ``const char *`` coercion in both
    :meth:`Pointer.from_param` (arguments) and :func:`_charptr_setattr`
    (struct fields): the caller decays it to a ``Pointer[char]`` whose
    owner keeps it alive.
    """
    data = bytes(data)
    buf = (char * (len(data) + 1))()
    ctypes.memmove(buf, data, len(data))
    return buf


def _charptr_setattr(self, name, value):
    """``__setattr__`` installed on structs/unions with a ``Pointer[char]``
    field (see :meth:`StructureMixin._install_charptr_setattr`).

    A ``bytes`` value at such a field is copied into an owned buffer and
    stored as the decayed ``char *``; the owning object keeps the buffer
    alive via ctypes ``_objects``. Everything else takes the normal
    C-level field set — the ``bytes`` guard keeps that path branch-cheap.
    """
    if type(value) in (bytes, bytearray):
        fields = type(self).__dict__.get('_charptr_fields')
        ftype = fields.get(name) if fields else None
        if ftype is not None:
            value = ctypes.cast(_char_buffer(value), ftype)
    super(StructureMixin, self).__setattr__(name, value)


# ---------------------------------------------------------------------------
# Convenience re-exports — surfaces from :mod:`ctypes` that consumers
# commonly need, so that C-binding code can be written through cbase
# alone without reaching past it for one-off utility functions.
# ---------------------------------------------------------------------------

sizeof = ctypes.sizeof
addressof = ctypes.addressof
string_at = ctypes.string_at
byref = ctypes.byref
pointer = ctypes.pointer
cast = ctypes.cast

# Raw ctypes re-exports — low-level escape hatches only. The registry
# does *not* use any of them: it maps the C primitives to cbase types so
# that every type flowing through a struct field or command signature
# carries the :class:`Mixin` surface — ``char`` → :class:`char`, ``int``
# → :class:`int32`, ``void *`` → :class:`VoidPointer`, and ``char *`` →
# :class:`Pointer`-to-:class:`char` (supplied via :class:`string`, the
# owned char buffer). c_char_p's ``bytes`` ⇄ C-string auto-marshalling
# is intentionally *not* used as a field/arg type — it hides who owns
# the buffer; :class:`string` + ``Pointer[char]`` make that explicit. The
# fixed-width numeric scalars (``c_uint32`` etc.) have no re-export at
# all; use the Mixin wrapper of the same width (:class:`uint32`).
c_char = ctypes.c_char
c_char_p = ctypes.c_char_p
c_int = ctypes.c_int
c_bool = ctypes.c_bool
c_void_p = ctypes.c_void_p


# Registered backing scalar per enum class name. Populated by
# :meth:`Enum.create` and read by :func:`friendly`. Keyed on the enum
# class's ``__name__`` so two construction sites that build identical
# enum shapes share the same lookup result — and so consumers can
# pre-register a scalar for an externally-defined enum class by
# assigning ``_ENUM_SCALARS[name] = scalar`` before any field/argtype
# resolution runs.
_ENUM_SCALARS: dict[str, type] = {}


class Enum:
    """IntEnum / IntFlag factory with backing-scalar registration.

    :class:`enum.IntEnum` and :class:`enum.IntFlag` classes are not
    directly storable in ctypes ``_fields_`` or ``argtypes`` — ctypes
    needs the underlying integer type. This factory builds the enum
    class *and* registers the ctypes scalar that :func:`friendly`
    substitutes for it at field-/argtype-time. ``flag=True`` selects
    :class:`enum.IntFlag` backed by ``c_uint32`` (or ``c_uint64`` when
    ``bitwidth=64``); default is :class:`enum.IntEnum` backed by
    ``c_int32``.
    """

    @staticmethod
    def create(name, values=(), aliases=(), bitwidth=32, flag=False):
        """Build the enum class and register its backing scalar.

        ``values`` is a sequence of ``(member_name, int_value)`` pairs.
        ``aliases`` is a sequence of ``(alias_name, target)`` where
        ``target`` is either a member name already in ``values`` or a
        plain int. Duplicate values trigger the alternate construction
        path in :meth:`_build_aliased`, since :class:`enum.IntEnum`
        rejects duplicates at definition time.
        """
        base = enum.IntFlag if flag else enum.IntEnum
        items: dict[str, int] = {}
        for vname, vval in values:
            items[vname] = int(vval)
        for aname, atarget in aliases:
            if atarget in items:
                items[aname] = items[atarget]
            elif isinstance(atarget, int):
                items[aname] = atarget

        if items:
            try:
                ecls = base(name, items)
            except (TypeError, ValueError):
                ecls = Enum._build_aliased(name, items, base)
        else:
            ecls = base(name, [])

        # cbase wrappers (not raw ctypes scalars): every other numeric
        # type in the substrate is a cbase wrapper, so enum-typed
        # struct fields stay consistent with primitive-typed fields.
        # Wrapper readback goes through the comparison / arithmetic /
        # ``__int__`` protocol defined on :class:`CData`, so user code
        # like ``info.flags & 0x10 == 0x10`` works transparently.
        if flag:
            scalar = uint64 if bitwidth == 64 else uint32
        else:
            scalar = int32
        _ENUM_SCALARS[name] = scalar
        return ecls

    @staticmethod
    def _build_aliased(name, items, base):
        """IntEnum subclass tolerant of duplicate values via aliases.

        The functional :class:`enum.IntEnum` constructor rejects two
        members with the same value at definition time. Build with the
        first-seen value for each int as primaries, then assign the
        remaining duplicates via ``setattr`` after construction — they
        land as proper :class:`enum.Enum` aliases in ``__members__``.
        """
        seen: dict[int, str] = {}
        primary: dict[str, int] = {}
        aliases: list[tuple[str, int]] = []
        for k, v in items.items():
            if v in seen:
                aliases.append((k, v))
            else:
                primary[k] = v
                seen[v] = k
        ecls = base(name, primary)
        for alias_name, alias_value in aliases:
            setattr(ecls, alias_name, ecls(alias_value))
        return ecls


def friendly(ctype):
    """Substitute :class:`enum.IntEnum` / :class:`enum.IntFlag` for its scalar.

    ctypes can't store an enum class directly in ``_fields_`` or use
    it in ``argtypes``; it needs the underlying integer type. If
    ``ctype`` was registered via :meth:`Enum.create` (or pre-seeded in
    :data:`_ENUM_SCALARS`), return the recorded backing scalar;
    otherwise pass through. ``None`` (the C ``void`` placeholder) is
    returned unchanged — callers decide how to handle the void case.
    """
    if isinstance(ctype, type) and issubclass(ctype, enum.Enum):
        scalar = _ENUM_SCALARS.get(ctype.__name__)
        if scalar is not None:
            return scalar
    return ctype


# Memoised int -> member decoders, keyed weakly on the enum class so a
# discarded registry's classes stay collectable. Keyed on the class and
# not (like :data:`_ENUM_SCALARS`) on its name, because the answer here
# depends on the members: two registries built from different vk.xml
# revisions have two different VkResult tables, and handing one's
# decoder to the other would mistranslate every code the older revision
# didn't have.
_ENUM_DECODERS: "weakref.WeakKeyDictionary[type, object]" = weakref.WeakKeyDictionary()


def decoder(ctype):
    """:func:`friendly`'s inverse for a *return* value: int -> enum member.

    :func:`friendly` erases an enum to its backing scalar. That is
    exactly right going in - an :class:`enum.IntEnum` handed to a
    ctypes ``argtype`` already *is* an int - and lossy coming back,
    where the scalar is all the caller ever sees. This builds the
    callable that puts the member back, for ctypes' callable-``restype``
    slot; ``None`` means "not decodable, leave the scalar alone".

    ctypes feeds a callable ``restype`` the C ``int`` the function
    returned, so only an enum backed by :class:`int32` qualifies. That
    is a statement about *signed*, int-sized storage and not just its
    width: a ``uint32``-backed :class:`enum.IntFlag` is the same size,
    but ctypes would sign-extend it and any member with bit 31 set
    would arrive negative and silently fail to match.

    An integer with no member - a result code from an extension newer
    than the parsed XML - passes through as a plain :class:`int`. A
    driver is entitled to return one, and a ``ValueError`` raised from
    inside the call would be the least useful available reaction to it.
    """
    if not (isinstance(ctype, type) and issubclass(ctype, enum.Enum)):
        return None
    if friendly(ctype) is not int32:
        return None
    decode = _ENUM_DECODERS.get(ctype)
    if decode is None:
        # Iterating the class yields canonical members only, so an
        # aliased code decodes to its primary spelling.
        table = {int(member): member for member in ctype}

        def decode(value, _get=table.get):
            member = _get(value)
            return value if member is None else member

        decode.__name__ = f'decode_{ctype.__name__}'
        _ENUM_DECODERS[ctype] = decode
    return decode


# ---------------------------------------------------------------------------
# Cross-version sanity test for the lazy pointer_type cache
# ---------------------------------------------------------------------------
#
# Run with ``python -m volkano.cbase`` on each interpreter you
# care about. ``Mixin.pointer_type`` uses ``cls.__dict__.get`` to read
# the cache and returns a local — this dodges a CPython 3.9–3.12
# type-attribute-cache invalidation bug in the ctypes Union metaclass
# where ``cls._pointer_type = X; return cls._pointer_type`` would
# return the stale pre-write value. Test verifies on every interpreter
# we run it against. A failure here would mean a fresh regression — at
# that point switch the read path to a module-level WeakValueDictionary
# keyed on ``cls`` (writes nothing to the class at all).

if __name__ == '__main__':
    import sys

    print(f'python: {sys.version.splitlines()[0]}')
    print('-' * 60)

    def _probe(kind: str, cls):
        def _label(v):
            if v is None:
                return 'None'
            if isinstance(v, str):
                return repr(v)
            return getattr(v, '__name__', repr(v))

        own_before = cls.__dict__.get('_pointer_type', '<missing>')
        ptype = cls.pointer_type
        own_after = cls.__dict__.get('_pointer_type', '<missing>')
        # Bug condition: pointer_type returned None even though the
        # classproperty body just assigned a fresh Pointer subclass.
        returned_ok = ptype is not None
        persisted_ok = returned_ok and own_after is ptype
        print(f'{kind} {cls.__name__}:')
        print(f'  _pointer_type before access: {_label(own_before)}')
        print(f'  pointer_type access returned: {_label(ptype)}')
        print(f'  _pointer_type after access:  {_label(own_after)}')
        print(f'  -> assignment persisted:     {persisted_ok}')
        return ptype, persisted_ok

    # --- Structure ---
    class _S(Struct):
        _fields_ = [('x', int32), ('y', int32)]

    s_ptype, s_ok = _probe('Struct', _S)

    # --- Union ---
    class _U(Union):
        _fields_ = [('a', int32), ('b', float32)]

    u_ptype, u_ok = _probe('Union ', _U)

    # --- Subclass isolation: child must not inherit parent's cached ptype.
    class _S2(_S):
        pass

    class _U2(_U):
        pass

    # Same-class post-mortem: after the classproperty body wrote the
    # value, what do attribute reads see now that we're back in the
    # outer frame? Diagnoses whether the type-attribute cache is stale.
    print('-' * 60)
    print('After the classproperty call, re-reading _U._pointer_type:')
    print(f'  _U.__dict__["_pointer_type"]:    '
          f'{_U.__dict__.get("_pointer_type")!r}')
    print(f'  getattr(_U, "_pointer_type"):    '
          f'{getattr(_U, "_pointer_type", "<missing>")!r}')
    print(f'  _U.pointer_type (2nd call):      '
          f'{_U.pointer_type!r}')

    s2_ptype = _S2.pointer_type
    u2_ptype = _U2.pointer_type
    print('-' * 60)
    print(f'Struct subclass has its own ptype: '
          f'{s_ptype is not None and s2_ptype is not None and s2_ptype is not s_ptype}')
    print(f'Union  subclass has its own ptype: '
          f'{u_ptype is not None and u2_ptype is not None and u2_ptype is not u_ptype}')

    # Direct setattr probe — bypass the classproperty so we see the raw
    # metaclass __setattr__ + __getattribute__ behavior on each kind.
    print('-' * 60)
    print('Direct setattr probe (no classproperty involved):')

    class _US(Union):
        _fields_ = [('a', int32), ('b', int32)]
    setattr(_US, '_pointer_type', 'MARKER')
    via_dict_u = _US.__dict__.get('_pointer_type', '<missing>')
    via_attr_u = getattr(_US, '_pointer_type', '<missing>')
    print(f'  Union  via __dict__:  {via_dict_u!r}')
    print(f'  Union  via getattr:   {via_attr_u!r}')
    print(f'  Union  read matches:  {via_dict_u == via_attr_u}')

    class _SS(Struct):
        _fields_ = [('x', int32), ('y', int32)]
    setattr(_SS, '_pointer_type', 'MARKER')
    via_dict_s = _SS.__dict__.get('_pointer_type', '<missing>')
    via_attr_s = getattr(_SS, '_pointer_type', '<missing>')
    print(f'  Struct via __dict__:  {via_dict_s!r}')
    print(f'  Struct via getattr:   {via_attr_s!r}')
    print(f'  Struct read matches:  {via_dict_s == via_attr_s}')

    print('-' * 60)
    if s_ok and u_ok:
        print('OK - lazy assignment persists on this interpreter.')
    else:
        print('FAIL - cls._pointer_type write-then-read returned stale. '
              'CPython type-attribute-cache regression; switch the cache '
              'to a module-level WeakValueDictionary keyed on cls.')
