"""Tests for the cbase ctypes-substrate kernel.

Covers the Mixin surface, scalar unboxing, the Pointer/Array generic
APIs, Handle/Enum/friendly, Struct two-step + bitfield + anonymous
layouts, FunctionPointer factories, and the ArithmeticMixin's
C-style promotion and compound-assignment rules.
"""
from __future__ import annotations

import ctypes
import enum
import sys
import unittest

from volkano.cbase import (
    Mixin,
    CData, Pointer, Handle, Array,
    Param, FunctionPointer, Struct, Union,
    char,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float32, float64,
    Enum, friendly, decoder, _ENUM_SCALARS,
)


class MixinSurfaceTests(unittest.TestCase):
    """Universal Mixin contract: ref/pointer/size and lazy pointer_type."""

    def test_size_classproperty(self):
        self.assertEqual(int32.size, 4)
        self.assertEqual(uint64.size, 8)

    def test_ref_returns_byref_handle(self):
        # ctypes.byref returns a _CArgObject; we can't deeply introspect it
        # but it must be non-None and roundtrip through a ctypes call.
        self.assertIsNotNone(int32(5).ref)

    def test_pointer_type_cached_per_class(self):
        self.assertIs(int32.pointer_type, int32.pointer_type)

    def test_pointer_type_distinct_across_classes(self):
        self.assertIsNot(int32.pointer_type, uint32.pointer_type)

    def test_struct_subclass_has_its_own_cache(self):
        # Cross-version cache regression: a subclass must not inherit the
        # parent's cached Pointer[Parent] — see cbase.py's __main__ probe.
        class A(Struct):
            _fields_ = [('x', int32)]
        class B(A):
            pass
        self.assertIsNot(A.pointer_type, B.pointer_type)


class CDataUnboxingTests(unittest.TestCase):
    """__int__, __float__, __index__ on scalar wrappers."""

    def test_int_on_int_scalar(self):
        self.assertEqual(int(int32(42)), 42)

    def test_int_on_float_scalar_truncates_toward_zero(self):
        self.assertEqual(int(float32(3.99)), 3)
        self.assertEqual(int(float32(-3.99)), -3)

    def test_float_on_int_scalar_widens(self):
        self.assertEqual(float(int32(42)), 42.0)

    def test_float_on_float_scalar(self):
        self.assertAlmostEqual(float(float32(3.14)), 3.14, places=5)

    def test_hex_via_index(self):
        self.assertEqual(hex(uint32(0xBEEF)), '0xbeef')

    def test_bin_via_index(self):
        self.assertEqual(bin(int32(5)), '0b101')

    def test_slice_via_index(self):
        self.assertEqual(['a', 'b', 'c', 'd'][:int32(2)], ['a', 'b'])

    def test_range_via_index(self):
        self.assertEqual(list(range(uint32(3))), [0, 1, 2])

    def test_null_voidpointer_unboxes_to_zero(self):
        self.assertEqual(int(Pointer[None]()), 0)
        self.assertEqual(float(Pointer[None]()), 0.0)

    def test_null_handle_unboxes_to_zero(self):
        H = Handle.create('NullHandleTest')
        self.assertEqual(int(H()), 0)


class PointerTests(unittest.TestCase):

    def test_none_returns_void_slot(self):
        # Pointer[None] is the canonical void* slot: a stable singleton
        # c_void_p subclass, spelled (and named) "Pointer[None]".
        vp = Pointer[None]
        self.assertIs(vp, Pointer[None])
        self.assertTrue(issubclass(vp, ctypes.c_void_p))
        self.assertEqual(vp.__name__, 'Pointer[None]')

    def test_mixin_type_returns_cached_pointer_type(self):
        p = Pointer[int32]
        self.assertIs(p, int32.pointer_type)
        self.assertTrue(issubclass(p, Pointer))

    def test_plain_ctypes_type_returns_ctypes_pointer(self):
        # A raw (non-cbase) ctypes element falls back to ctypes.POINTER,
        # which ctypes itself caches — so identity still holds. The
        # registry never reaches this branch (every element is a cbase
        # Mixin type); it exists only for direct ctypes-interop.
        p = Pointer[ctypes.c_int]
        self.assertTrue(issubclass(p, ctypes._Pointer))
        self.assertIs(p, Pointer[ctypes.c_int])

    def test_intenum_substituted_via_friendly(self):
        E = Enum.create('PointerEnumTest', values=[('R', 0), ('G', 1)])
        p = Pointer[E]
        self.assertIs(p._type_, int32)

    def test_from_address_returns_typed_pointer(self):
        p = Pointer[int32].from_address(0)
        self.assertIsInstance(p, Pointer[int32])

    def test_cast_to_voidpointer_preserves_address(self):
        x = int32(42)
        typed_ptr = x.ptr  # Pointer[int32]
        void_ptr = typed_ptr.cast(Pointer[None])
        self.assertIsInstance(void_ptr, Pointer[None])
        self.assertEqual(void_ptr.value, ctypes.addressof(x))

    def test_cast_to_typed_pointer(self):
        x = int32(42)
        typed_ptr = x.ptr
        # Cast to a pointer of a different ctype with same width.
        cast = typed_ptr.cast(Pointer[uint32])
        self.assertIsInstance(cast, Pointer[uint32])
        # Same address means the new pointer dereferences to the same bytes.
        # ``cast[0]`` returns a cbase uint32 wrapper — read .value to compare.
        self.assertEqual(cast[0].value, 42)

    def test_cast_does_not_dereference(self):
        # ``Pointer.cast`` keeps the pointer; ``Mixin.reinterpret``
        # dereferences. Verify they're operationally distinct: cast must
        # not touch the pointed-to bytes (no dereference), so it can be
        # called on a NULL pointer without segfaulting.
        null = Pointer[int32].from_address(0)
        result = null.cast(Pointer[None])
        self.assertEqual(result.value, None)  # NULL preserved


class ArrayTests(unittest.TestCase):

    def test_none_collapses_to_voidpointer(self):
        a = Array[None, 4]
        self.assertIs(a._type_, Pointer[None])
        self.assertEqual(a._length_, 4)

    def test_intenum_substituted_via_friendly(self):
        E = Enum.create('ArrayEnumTest', values=[('A', 0)])
        a = Array[E, 8]
        self.assertIs(a._type_, int32)
        self.assertEqual(a._length_, 8)

    def test_size_matches_ctypes(self):
        a = Array[int32, 4]
        self.assertEqual(ctypes.sizeof(a), 4 * ctypes.sizeof(int32))

    def test_length_from_cdata_scalar(self):
        a = Array[int32, uint32(8)]
        self.assertEqual(a._length_, 8)

    def test_plain_ctypes_basetype(self):
        a = Array[ctypes.c_int, 3]
        self.assertEqual(a._length_, 3)


class CharPtrBytesTests(unittest.TestCase):
    """``bytes`` behaves as ``const char *`` at ``Pointer[char]`` slots."""

    def test_from_param_bytes_decays_to_char_ptr(self):
        # Argument coercion: bytes -> owned buffer -> decayed char*.
        p = Pointer[char].from_param(b'hello')
        self.assertTrue(issubclass(type(p), Pointer))
        self.assertEqual(ctypes.string_at(p), b'hello')

    def test_from_param_non_char_pointer_unaffected(self):
        # Only Pointer[char] coerces bytes; other element types delegate.
        with self.assertRaises(Exception):
            Pointer[int32].from_param(b'hello')

    def test_field_assignment_bytes_to_char_ptr(self):
        class S(Struct):
            _fields_ = [('name', Pointer[char])]
        s = S()
        s.name = b'application'           # attribute assignment
        self.assertEqual(ctypes.string_at(s.name), b'application')

    def test_constructor_kwarg_bytes_to_char_ptr(self):
        class S(Struct):
            _fields_ = [('a', uint32), ('name', Pointer[char])]
        s = S(a=7, name=b'world')          # ctypes routes ctor kwargs via __setattr__
        self.assertEqual(s.a, 7)
        self.assertEqual(ctypes.string_at(s.name), b'world')

    def test_buffer_kept_alive_by_struct(self):
        import gc
        class S(Struct):
            _fields_ = [('name', Pointer[char])]
        s = S(name=b'survives-gc')          # no external ref to the buffer
        gc.collect()
        self.assertEqual(ctypes.string_at(s.name), b'survives-gc')

    def test_override_installed_only_when_char_ptr_field_present(self):
        from volkano.cbase import _charptr_setattr
        class HasCharPtr(Struct):
            _fields_ = [('name', Pointer[char])]
        class NoCharPtr(Struct):
            _fields_ = [('a', uint32), ('b', uint32)]
        self.assertIs(HasCharPtr.__setattr__, _charptr_setattr)
        self.assertIsNot(NoCharPtr.__setattr__, _charptr_setattr)
        self.assertIn('_charptr_fields', HasCharPtr.__dict__)
        self.assertNotIn('_charptr_fields', NoCharPtr.__dict__)

    def test_inline_char_array_field_still_takes_bytes(self):
        # char[N] (Array) is inline storage, not a pointer; bytes still
        # round-trip natively and the override must not interfere.
        class S(Struct):
            _fields_ = [('buf', Array[char, 8])]
        s = S(buf=b'hi')
        self.assertEqual(s.buf, b'hi')


class HandleTests(unittest.TestCase):

    def test_create_returns_distinct_subclasses(self):
        A = Handle.create('HandleA')
        B = Handle.create('HandleB')
        self.assertIsNot(A, B)
        self.assertEqual(A.__name__, 'HandleA')
        self.assertEqual(B.__name__, 'HandleB')

    def test_handle_is_pointer_sized(self):
        H = Handle.create('PtrSizedHandle')
        self.assertEqual(ctypes.sizeof(H), ctypes.sizeof(ctypes.c_void_p))

    def test_handle_carries_int_address(self):
        H = Handle.create('AddrHandle')
        self.assertEqual(H(0x1234).value, 0x1234)

    def test_handle_from_param_accepts_int(self):
        H = Handle.create('FromParamHandle')
        # No assertion on the returned _CArgObject adapter — we just need
        # the coercion path not to raise.
        H.from_param(0x1000)


class EnumTests(unittest.TestCase):

    def test_intenum_by_default(self):
        E = Enum.create('Default1', values=[('A', 1), ('B', 2)])
        self.assertTrue(issubclass(E, enum.IntEnum))
        self.assertEqual(E.A, 1)
        self.assertEqual(E.B, 2)

    def test_intflag_with_flag_kwarg(self):
        F = Enum.create('FlagDefault', values=[('X', 1), ('Y', 2)], flag=True)
        self.assertTrue(issubclass(F, enum.IntFlag))
        self.assertEqual(F.X | F.Y, 3)

    def test_alias_to_member_name(self):
        E = Enum.create('AliasName', values=[('NEW', 5)], aliases=[('OLD', 'NEW')])
        self.assertEqual(E.OLD, E.NEW)

    def test_alias_to_int_literal(self):
        E = Enum.create('AliasInt', values=[('X', 1)], aliases=[('LITERAL', 42)])
        self.assertEqual(E.LITERAL, 42)

    def test_duplicate_value_becomes_alias(self):
        E = Enum.create('Dup', values=[('A', 1), ('B', 1), ('C', 2)])
        self.assertIs(E.B, E.A)
        self.assertNotEqual(E.C, E.A)

    def test_scalar_registered_for_intenum(self):
        Enum.create('ScalarReg1', values=[('A', 0)])
        self.assertIs(_ENUM_SCALARS['ScalarReg1'], int32)

    def test_scalar_registered_for_intflag_32(self):
        Enum.create('ScalarReg32', values=[('A', 1)], flag=True)
        self.assertIs(_ENUM_SCALARS['ScalarReg32'], uint32)

    def test_scalar_registered_for_intflag_64(self):
        Enum.create('ScalarReg64', values=[('A', 1)], flag=True, bitwidth=64)
        self.assertIs(_ENUM_SCALARS['ScalarReg64'], uint64)

    def test_empty_values_constructs(self):
        E = Enum.create('Empty', values=[])
        self.assertTrue(issubclass(E, enum.IntEnum))


class FriendlyTests(unittest.TestCase):

    def test_registered_enum_substituted(self):
        E = Enum.create('FriendlyReg', values=[('A', 0)])
        self.assertIs(friendly(E), int32)

    def test_unregistered_enum_passes_through(self):
        class U(enum.IntEnum):
            X = 0
        self.assertIs(friendly(U), U)

    def test_ctypes_type_passes_through(self):
        self.assertIs(friendly(ctypes.c_int), ctypes.c_int)

    def test_none_passes_through(self):
        self.assertIsNone(friendly(None))


class DecoderTests(unittest.TestCase):

    def test_int_backed_enum_decodes_to_member(self):
        E = Enum.create('DecReg', values=[('A', 0), ('B', -3)])
        self.assertIs(decoder(E)(-3), E.B)

    def test_unknown_value_passes_through_as_int(self):
        E = Enum.create('DecUnknown', values=[('A', 0)])
        got = decoder(E)(999)
        self.assertEqual(got, 999)
        self.assertNotIsInstance(got, enum.Enum)

    def test_alias_decodes_to_primary_member(self):
        E = Enum.create('DecAlias', values=[('A', 0)], aliases=[('A_KHR', 'A')])
        self.assertIs(decoder(E)(0), E.A)

    def test_decoder_is_memoised_per_class(self):
        E = Enum.create('DecMemo', values=[('A', 0)])
        self.assertIs(decoder(E), decoder(E))

    def test_unsigned_flag_is_not_decodable(self):
        # uint32-backed: ctypes would sign-extend a bit-31 member.
        F = Enum.create('DecFlag', values=[('X', 1)], flag=True)
        self.assertIsNone(decoder(F))

    def test_unregistered_enum_is_not_decodable(self):
        class U(enum.IntEnum):
            X = 0
        self.assertIsNone(decoder(U))

    def test_non_enum_is_not_decodable(self):
        self.assertIsNone(decoder(int32))
        self.assertIsNone(decoder(ctypes.c_int))
        self.assertIsNone(decoder(None))


class StructLayoutTests(unittest.TestCase):

    def test_create_empty_returns_named_subclass(self):
        S = Struct.create_empty('EmptyStruct')
        self.assertEqual(S.__name__, 'EmptyStruct')
        self.assertTrue(issubclass(S, ctypes.Structure))

    def test_set_fields_assigns_layout(self):
        S = Struct.create_empty('LayoutS')
        Struct.set_fields(S, [('x', int32), ('y', int32)])
        self.assertEqual(ctypes.sizeof(S), 8)
        s = S()
        s.x = 1
        s.y = 2
        self.assertEqual(s.x.value, 1)
        self.assertEqual(s.y.value, 2)

    def test_self_cycle_via_pointer(self):
        Node = Struct.create_empty('SelfCycleNode')
        Struct.set_fields(Node, [('val', int32), ('next', Pointer[Node])])
        self.assertIs(Node._fields_[1][1]._type_, Node)

    def test_one_shot_create(self):
        S = Struct.create('OneShot', [('a', int32), ('b', uint64)])
        self.assertEqual(ctypes.sizeof(S), 16)

    def test_bitfield_three_tuple(self):
        S = Struct.create('Bitfield', [
            ('a', ctypes.c_uint32, 4),
            ('b', ctypes.c_uint32, 28),
        ])
        self.assertEqual(ctypes.sizeof(S), 4)
        s = S()
        s.a = 0xF
        s.b = 0xABCDEF1
        self.assertEqual(s.a, 0xF)
        self.assertEqual(s.b, 0xABCDEF1)

    def test_anonymous_member(self):
        Inner = Union.create('AnonInner', [('i', int32), ('f', float32)])
        Outer = Struct.create('AnonOuter', [
            ('tag', int32),
            ('u', Inner, '_anonymous_'),
        ])
        self.assertEqual(Outer._anonymous_, ('u',))
        o = Outer()
        o.i = 42  # Reachable directly because 'u' is anonymous.
        self.assertEqual(o.u.i.value, 42)

    def test_param_entries(self):
        S = Struct.create('PMS', [Param('x', int32), Param('y', uint64)])
        self.assertEqual([f[0] for f in S._fields_], ['x', 'y'])
        self.assertEqual(ctypes.sizeof(S), 16)

    def test_friendly_substitution_in_fields(self):
        E = Enum.create('StructFieldEnum', values=[('A', 0)])
        S = Struct.create('WithEnumField', [('c', E)])
        # IntEnum substituted with cbase backing scalar.
        self.assertIs(S._fields_[0][1], int32)


class UnionTests(unittest.TestCase):

    def test_overlapping_fields(self):
        U = Union.create('TestUnion', [('i', int32), ('f', float32)])
        self.assertEqual(ctypes.sizeof(U), 4)
        u = U()
        u.i = 0x40490FDB  # bit pattern for ~3.14159
        self.assertAlmostEqual(u.f.value, 3.1415925, places=5)


class FunctionPointerTests(unittest.TestCase):

    def test_tuple_param_form(self):
        F = FunctionPointer.create('FP_tuple', int32, [('x', int32), ('y', uint32)])
        self.assertEqual(F._argtypes_, (int32, uint32))

    def test_param_object_form(self):
        F = FunctionPointer.create(
            'FP_param', None,
            [Param('a', int32), Param('b', float32)])
        self.assertEqual(F._argtypes_, (int32, float32))

    def test_void_rettype(self):
        F = FunctionPointer.create('FP_void', None, [])
        self.assertIsNone(F._restype_)

    def test_friendly_applied_to_argtypes(self):
        E = Enum.create('FPArgEnum', values=[('A', 0)])
        F = FunctionPointer.create('FP_enum_arg', int32, [('e', E)])
        self.assertIs(F._argtypes_[0], int32)

    def test_friendly_applied_to_rettype(self):
        E = Enum.create('FPRetEnum', values=[('A', 0)])
        F = FunctionPointer.create('FP_enum_ret', E, [])
        self.assertIs(F._restype_, int32)

    def test_each_typedef_gets_its_own_subclass(self):
        # Unique signature shape so this test owns its cache slot
        # regardless of test-method execution order — ctypes'
        # WINFUNCTYPE/CFUNCTYPE caches are process-global.
        F1 = FunctionPointer.create(
            'FP_typedef_a', int16,
            [('a', int8), ('b', int8), ('c', int8)])
        F2 = FunctionPointer.create(
            'FP_typedef_b', int16,
            [('a', int8), ('b', int8), ('c', int8)])
        # Each typedef is its own Mixin-bearing subclass; identical
        # signatures no longer collapse to one Python class.
        self.assertIsNot(F1, F2)
        self.assertEqual(F1.__name__, 'FP_typedef_a')
        self.assertEqual(F2.__name__, 'FP_typedef_b')
        # ctypes' signature cache still dedupes the underlying
        # _CFuncPtr class — both subclasses share it via MRO.
        cfuncptr_names = ('CFunctionType', 'WinFunctionType')
        base1 = next(c for c in F1.__mro__ if c.__name__ in cfuncptr_names)
        base2 = next(c for c in F2.__mro__ if c.__name__ in cfuncptr_names)
        self.assertIs(base1, base2)

    def test_result_is_mixin_subclass(self):
        F = FunctionPointer.create('FP_mixin', int32, [('x', int32)])
        self.assertTrue(issubclass(F, Mixin))

    @unittest.skipUnless(sys.platform == 'win32', 'stdcall only on win32')
    def test_stdcall_routes_to_winfunctype(self):
        F = FunctionPointer.create('FP_std', int32, [('x', int32)], stdcall=True)
        # WINFUNCTYPE sets _flags_ = 0 (CFUNCTYPE uses 1).
        self.assertEqual(F._flags_, 0)

    def test_default_cdecl_flags(self):
        F = FunctionPointer.create('FP_cdecl', int32, [('x', int32)])
        self.assertEqual(F._flags_, 1)


class FunctionPointerSubscriptTests(unittest.TestCase):
    """Type-expression form: FunctionPointer[ret, [args]]."""

    def test_basic_subscript(self):
        F = FunctionPointer[None, [int32, int32]]
        self.assertIsNone(F._restype_)
        self.assertEqual(F._argtypes_, (int32, int32))
        self.assertTrue(issubclass(F, Mixin))

    def test_subscript_with_typed_return(self):
        F = FunctionPointer[int32, [int32]]
        self.assertIs(F._restype_, int32)
        self.assertEqual(F._argtypes_, (int32,))

    def test_subscript_no_args(self):
        F = FunctionPointer[int32, []]
        self.assertEqual(F._argtypes_, ())
        self.assertIs(F._restype_, int32)

    def test_subscript_tuple_args_form(self):
        # Both list and tuple are accepted for the args sequence.
        F = FunctionPointer[None, (int32, uint32)]
        self.assertEqual(F._argtypes_, (int32, uint32))

    def test_subscript_friendly_applied(self):
        E = Enum.create('FPSubEnum', values=[('A', 0)])
        F = FunctionPointer[E, [E]]
        self.assertIs(F._restype_, int32)
        self.assertIs(F._argtypes_[0], int32)

    def test_subscript_generates_readable_name(self):
        F = FunctionPointer[None, [int32, int32]]
        # Generated name reads like a function type for legibility in
        # tracebacks; exact shape is "FunctionPointer[(args) -> ret]".
        self.assertIn('int32', F.__name__)
        self.assertIn('void', F.__name__)
        self.assertIn('->', F.__name__)

    def test_subscript_equivalent_to_create(self):
        # Same underlying signature → both forms produce classes
        # that share the cached ctypes _CFuncPtr base.
        Fc = FunctionPointer.create('FP_via_create', int16, [('x', uint16)])
        Fs = FunctionPointer[int16, [uint16]]
        cfuncptr_names = ('CFunctionType', 'WinFunctionType')
        base_c = next(c for c in Fc.__mro__ if c.__name__ in cfuncptr_names)
        base_s = next(c for c in Fs.__mro__ if c.__name__ in cfuncptr_names)
        self.assertIs(base_c, base_s)

    def test_subscript_rejects_single_arg(self):
        with self.assertRaises(TypeError):
            FunctionPointer[int32]

    def test_subscript_rejects_non_sequence_args(self):
        with self.assertRaises(TypeError):
            # args must be a list/tuple — passing a bare type is wrong.
            FunctionPointer[None, int32]


class ArithmeticBasicTests(unittest.TestCase):
    """Forward, reflected, in-place, bitwise, unary."""

    def test_forward_add_same_type(self):
        r = int32(5) + int32(3)
        self.assertEqual(r.value, 8)
        self.assertIs(type(r), int32)

    def test_forward_add_with_plain_int(self):
        r = int32(5) + 3
        self.assertEqual(r.value, 8)
        self.assertIs(type(r), int32)

    def test_reflected_add_with_plain_int(self):
        r = 5 + int32(3)
        self.assertEqual(r.value, 8)
        self.assertIs(type(r), int32)

    def test_inplace_preserves_identity(self):
        x = int32(10)
        xid = id(x)
        x += int32(5)
        self.assertEqual(x.value, 15)
        self.assertEqual(id(x), xid)

    def test_bitwise_or(self):
        self.assertEqual((uint32(0xF0) | uint32(0x0F)).value, 0xFF)

    def test_bitwise_and(self):
        self.assertEqual((uint32(0xFF) & uint32(0x0F)).value, 0x0F)

    def test_bitwise_xor(self):
        self.assertEqual((uint32(0xFF) ^ uint32(0x0F)).value, 0xF0)

    def test_shift_left(self):
        self.assertEqual((uint32(1) << uint32(8)).value, 0x100)

    def test_unary_neg(self):
        self.assertEqual((-int32(5)).value, -5)

    def test_unary_invert(self):
        self.assertEqual((~uint32(0)).value, 0xFFFFFFFF)

    def test_unary_abs(self):
        self.assertEqual(abs(int32(-9)).value, 9)


class ArithmeticPromotionTests(unittest.TestCase):
    """C-style usual arithmetic conversions on cbase scalars."""

    def test_wider_int_wins(self):
        self.assertIs(type(int32(5) + int64(3)), int64)

    def test_wider_int_wins_either_side(self):
        self.assertIs(type(int64(5) + int32(3)), int64)

    def test_unsigned_wins_on_tie(self):
        self.assertIs(type(int32(5) + uint32(3)), uint32)

    def test_unsigned_wins_on_tie_either_side(self):
        self.assertIs(type(uint32(5) + int32(3)), uint32)

    def test_unsigned_wins_on_64bit_tie(self):
        self.assertIs(type(int64(5) + uint64(3)), uint64)

    def test_float_dominates_int(self):
        self.assertIs(type(int32(5) + float32(1.5)), float32)

    def test_float_dominates_int_either_side(self):
        self.assertIs(type(float32(1.5) + int64(5)), float32)

    def test_wider_float_wins(self):
        self.assertIs(type(float32(1.0) + float64(2.0)), float64)

    def test_pointer_dominates_int(self):
        H = Handle.create('PointerDomH')
        self.assertIs(type(H(0x1000) + int64(8)), H)

    def test_pointer_dominates_either_side(self):
        H = Handle.create('PointerDomH2')
        self.assertIs(type(int64(8) + H(0x1000)), H)

    def test_plain_int_does_not_promote(self):
        self.assertIs(type(int32(5) + 3), int32)

    def test_same_size_signed_signed_no_int_promotion(self):
        # Deliberately *not* C's integer-promotion rule (int8 + int8 → int).
        self.assertIs(type(int8(10) + int8(5)), int8)


class ArithmeticCompoundAssignmentTests(unittest.TestCase):
    """C compound-assignment: promote, compute, truncate-back."""

    def test_float_into_int_truncates_toward_zero(self):
        x = int32(100)
        x += float32(1.5)
        self.assertEqual(x.value, 101)

    def test_float_into_int_negative_truncates_toward_zero(self):
        x = int32(0)
        x += float32(-1.5)
        # int(-1.5) is -1 (toward zero), not -2 (floor).
        self.assertEqual(x.value, -1)

    def test_repeated_small_float_increment_stays_zero(self):
        # Canonical C gotcha: each compound step truncates independently.
        x = int32(0)
        for _ in range(3):
            x += float32(0.7)
        self.assertEqual(x.value, 0)

    def test_int_into_float_widens_losslessly(self):
        x = float32(2.0)
        xid = id(x)
        x += int32(1)
        self.assertAlmostEqual(x.value, 3.0)
        self.assertEqual(id(x), xid)

    def test_wider_int_into_narrower_wraps_via_ctypes(self):
        x = int32(0)
        x += int64(0xFFFFFFFFFF)  # 40 bits
        # ctypes truncates to bottom 32 bits → 0xFFFFFFFF → -1 as int32.
        self.assertEqual(x.value, -1)

    def test_unsigned_compound_assignment_wraps(self):
        x = uint32(0xFFFFFFFE)
        x += uint32(5)
        self.assertEqual(x.value, 3)


class ArithmeticReflectedTests(unittest.TestCase):
    """Reflected ops preserve the same promotion regardless of side."""

    def test_radd_cross_type_promotes(self):
        self.assertIs(type(int64(3).__radd__(int32(5))), int64)

    def test_rsub_subtraction_order_preserved(self):
        # 5 - int32(3) → int32(2), NOT int32(-2)
        self.assertEqual((5 - int32(3)).value, 2)


class ConcreteSizesTests(unittest.TestCase):
    """The scalar wrappers expose correct ctypes sizes."""

    def test_concrete_widths(self):
        self.assertEqual(ctypes.sizeof(int8), 1)
        self.assertEqual(ctypes.sizeof(int16), 2)
        self.assertEqual(ctypes.sizeof(int32), 4)
        self.assertEqual(ctypes.sizeof(int64), 8)
        self.assertEqual(ctypes.sizeof(uint8), 1)
        self.assertEqual(ctypes.sizeof(uint16), 2)
        self.assertEqual(ctypes.sizeof(uint32), 4)
        self.assertEqual(ctypes.sizeof(uint64), 8)
        self.assertEqual(ctypes.sizeof(float32), 4)
        self.assertEqual(ctypes.sizeof(float64), 8)


if __name__ == '__main__':
    unittest.main()
