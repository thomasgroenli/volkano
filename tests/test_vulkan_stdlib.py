"""Tests for the factories and VkRegistry in volkano.vulkan_stdlib."""
from __future__ import annotations

import enum
import unittest

from volkano import cbase
from volkano.lazy import _Thunk, _Raw, SELF, Force
from volkano.vulkan_stdlib import (
    STDLIB, VkRegistry, CommandSignature,
    ref, array,
    make_handle, make_basetype, make_bitmask, make_enum,
    make_funcpointer, make_struct, make_union, make_command,
    enum_member,
)


def _registry(extra=None):
    data = dict(STDLIB)
    if extra:
        data.update(extra)
    return VkRegistry(data)


class RefArrayTests(unittest.TestCase):

    def test_ref_char_returns_pointer_to_char(self):
        self.assertIs(ref(cbase.char), cbase.Pointer[cbase.char])

    def test_ref_none_collapses_to_voidpointer(self):
        self.assertIs(ref(None), cbase.Pointer[None])

    def test_ref_generic_returns_POINTER(self):
        self.assertIs(ref(cbase.uint32), cbase.Pointer[cbase.uint32])

    def test_array_fixed_size(self):
        arr = array(cbase.uint32, 4)
        self.assertTrue(issubclass(arr, cbase.Array))
        self.assertIs(arr._type_, cbase.uint32)
        self.assertEqual(arr._length_, 4)

    def test_array_none_target_uses_voidpointer(self):
        arr = array(None, 3)
        self.assertTrue(issubclass(arr, cbase.Array))
        self.assertIs(arr._type_, cbase.Pointer[None])
        self.assertEqual(arr._length_, 3)


class HandleTests(unittest.TestCase):

    def test_handle_is_void_p_subclass_with_metadata(self):
        H = make_handle(None, 'VkInstance', dispatchable=True,
                        parent='VkX', objtypeenum='VK_OBJECT_TYPE_INSTANCE')
        self.assertTrue(issubclass(H, cbase.c_void_p))
        self.assertEqual(H.__name__, 'VkInstance')
        self.assertEqual(H._vk_kind, 'handle')
        self.assertIs(H._vk_dispatchable, True)
        self.assertEqual(H._vk_parent, 'VkX')
        self.assertEqual(H._vk_objtypeenum, 'VK_OBJECT_TYPE_INSTANCE')

    def test_handle_instance_has_ptr_and_ref(self):
        H = make_handle(None, 'VkFoo')
        inst = H()
        self.assertIsNotNone(inst.ptr)
        self.assertIsNotNone(inst.ref)


class BasetypeBitmaskTests(unittest.TestCase):

    def test_basetype_known_underlying_returns_primitive(self):
        self.assertIs(make_basetype(None, 'VkBool32', 'uint32_t'),
                      cbase.uint32)

    def test_basetype_unknown_underlying_falls_back_to_voidpointer(self):
        self.assertIs(make_basetype(None, 'VkUnknown', 'no_such_t'),
                      cbase.Pointer[None])

    def test_bitmask_widths(self):
        self.assertIs(make_bitmask(None, 'F', bitwidth=32), cbase.uint32)
        self.assertIs(make_bitmask(None, 'F', bitwidth=64), cbase.uint64)


class EnumTests(unittest.TestCase):

    def test_enum_intenum_default(self):
        E = make_enum(None, 'E', kind='enum',
                      values=(('A', 0), ('B', 1)))
        self.assertTrue(issubclass(E, enum.IntEnum))
        self.assertEqual(E.A, 0)
        self.assertEqual(E.B, 1)

    def test_enum_bitmask_becomes_intflag(self):
        E = make_enum(None, 'F', kind='bitmask',
                      values=(('X', 1), ('Y', 2)))
        self.assertTrue(issubclass(E, enum.IntFlag))
        self.assertEqual(int(E.X | E.Y), 3)

    def test_enum_aliases_point_to_primary(self):
        E = make_enum(None, 'E', kind='enum',
                      values=(('A', 0),),
                      aliases=(('A_KHR', 'A'),))
        self.assertEqual(E.A_KHR, E.A)

    def test_enum_duplicate_values_become_aliases(self):
        E = make_enum(None, 'E', kind='enum',
                      values=(('A', 0), ('A_ALIAS', 0)))
        self.assertEqual(E.A, 0)
        self.assertEqual(E.A_ALIAS, E.A)


class EnumMemberTests(unittest.TestCase):

    def setUp(self):
        self.group = make_enum(None, 'VkFoo', 'enum', 32,
                               (('VK_FOO_A', 0),), (('VK_FOO_A_KHR', 'VK_FOO_A'),))

    def test_returns_the_member(self):
        self.assertIs(enum_member(self.group, 'VK_FOO_A', 0),
                      self.group.VK_FOO_A)

    def test_alias_returns_the_canonical_member(self):
        self.assertIs(enum_member(self.group, 'VK_FOO_A_KHR', 0),
                      self.group.VK_FOO_A)

    def test_name_absent_from_the_group_falls_back_to_the_value(self):
        self.assertEqual(enum_member(self.group, 'VK_FOO_MISSING', 7), 7)

    def test_non_member_attribute_falls_back_to_the_value(self):
        # 'name' resolves on the class but is not a member of it.
        self.assertEqual(enum_member(self.group, 'name', 7), 7)


class FuncpointerTests(unittest.TestCase):

    def test_funcpointer_name_assigned_to_typedef(self):
        PFN = make_funcpointer(None, 'PFN_Foo', rettype=None,
                               params=(('x', cbase.c_int),))
        self.assertEqual(PFN.__name__, 'PFN_Foo')


class StructUnionTests(unittest.TestCase):

    def test_struct_basic_fields(self):
        r = _registry({
            'S': _Thunk(make_struct, SELF, 'S', _Raw([
                ['a', Force('uint32_t')],
                ['b', Force('int32_t')],
            ])),
        })
        S = r('S')
        self.assertTrue(issubclass(S, cbase.Struct))
        self.assertEqual([f[0] for f in S._fields_], ['a', 'b'])
        self.assertEqual([f[1] for f in S._fields_],
                         [cbase.uint32, cbase.int32])

    def test_struct_self_pointer_does_not_deadlock(self):
        from volkano.cbase import Pointer
        r = _registry({
            'Node': _Thunk(make_struct, SELF, 'Node', _Raw([
                ['next', _Thunk(ref, Force('Node'))],
                ['value', Force('uint32_t')],
            ])),
        })
        Node = r('Node')
        # ``ref()`` now returns a cbase Pointer (which is itself a
        # cbase.Pointer subclass); the field is the cached
        # Pointer[Node] for the in-progress struct.
        self.assertIs(Node._fields_[0][1], Pointer[Node])

    def test_union_basic(self):
        r = _registry({
            'U': _Thunk(make_union, SELF, 'U', _Raw([
                ['a', Force('uint32_t')],
                ['b', Force('float')],
            ])),
        })
        U = r('U')
        self.assertTrue(issubclass(U, cbase.Union))

    def test_struct_substitutes_enum_with_scalar(self):
        """Struct fields typed as an IntEnum must use the underlying scalar."""
        r = _registry({
            'E': _Thunk(make_enum, SELF, 'E', 'enum', 32,
                        _Raw((('A', 0),)), _Raw(())),
            'S': _Thunk(make_struct, SELF, 'S', _Raw([
                ['e', Force('E')],
            ])),
        })
        S = r('S')
        self.assertIs(S._fields_[0][1], cbase.int32)


class _FakeDll:
    """Stand-in for the Vulkan loader DSO — getattr returns or raises like the real thing."""

    def __init__(self, names):
        self._fns = {}
        for name in names:
            def _fn(*a, _n=name): return _n
            _fn.argtypes = ()
            _fn.restype = None
            self._fns[name] = _fn

    def __getattr__(self, name):
        try:
            return self._fns[name]
        except KeyError as e:
            raise AttributeError(name) from e


class CommandTests(unittest.TestCase):

    def test_unbound_command_call_raises(self):
        sig = make_command(VkRegistry({}), 'vkFoo', None, [], {})
        self.assertIsInstance(sig, CommandSignature)
        with self.assertRaisesRegex(RuntimeError, 'not bound'):
            sig()

    def test_command_binds_when_symbol_present(self):
        sig = CommandSignature('vkBar', None, [], {})
        dll = _FakeDll(['vkBar'])
        self.assertIsNotNone(sig.bind(dll))
        self.assertEqual(sig(), 'vkBar')

    def test_command_bind_missing_symbol_returns_none(self):
        sig = CommandSignature('vkAbsent', None, [], {})
        self.assertIsNone(sig.bind(_FakeDll([])))
        self.assertIsNone(sig._fn)

    def test_enum_return_binds_a_decoding_restype(self):
        R = make_enum(None, 'VkResultLike', 'enum', 32,
                      (('VK_SUCCESS', 0), ('VK_ERROR_X', -3)), ())
        sig = CommandSignature('vkEnumRet', R, [], {})
        sig.bind(_FakeDll(['vkEnumRet']))
        self.assertIs(sig._fn.restype(-3), R.VK_ERROR_X)

    def test_scalar_return_binds_the_scalar_itself(self):
        sig = CommandSignature('vkScalarRet', cbase.uint32, [], {})
        sig.bind(_FakeDll(['vkScalarRet']))
        self.assertIs(sig._fn.restype, cbase.uint32)

    def test_void_return_binds_none(self):
        sig = CommandSignature('vkVoidRet', None, [], {})
        sig.bind(_FakeDll(['vkVoidRet']))
        self.assertIsNone(sig._fn.restype)

    def test_command_binds_on_first_call_not_at_force(self):
        dll = _FakeDll(['vkAuto'])
        r = VkRegistry({**STDLIB,
                        'vkAuto': _Thunk(make_command, SELF, 'vkAuto', None,
                                         _Raw(()), _Raw({}))},
                       library=dll)
        sig = r('vkAuto')
        self.assertIsNone(sig._fn)
        self.assertEqual(sig(), 'vkAuto')
        self.assertIs(sig._fn, dll._fns['vkAuto'])

    def test_rebind_commands_binds_only_unbound_in_cache(self):
        sig = CommandSignature('vkLater', None, [], {})
        r = VkRegistry({})
        r._cache['vkLater'] = sig
        self.assertEqual(r.rebind_commands(), 0)  # no library

        r.attach_library(_FakeDll(['vkLater']))
        self.assertEqual(r.rebind_commands(), 1)
        self.assertIsNotNone(sig._fn)

        self.assertEqual(r.rebind_commands(), 0)  # already bound


class RegistryAttrTests(unittest.TestCase):

    def test_registry_getattr_forces_key(self):
        r = _registry({
            'VkBool32': _Thunk(make_basetype, SELF, 'VkBool32', 'uint32_t'),
        })
        self.assertIs(r.VkBool32, cbase.uint32)

    def test_registry_getattr_missing_raises_attributeerror(self):
        with self.assertRaises(AttributeError):
            _registry().NopeMissing

    def test_registry_underscore_attrs_not_intercepted(self):
        r = _registry()
        self.assertIsNone(r._library)
        with self.assertRaises(AttributeError):
            r._something_undefined

    def test_registry_dir_includes_data_keys(self):
        r = _registry({
            'VkBool32': _Thunk(make_basetype, SELF, 'VkBool32', 'uint32_t'),
        })
        names = dir(r)
        self.assertIn('VkBool32', names)
        self.assertIn('uint32_t', names)


if __name__ == '__main__':
    unittest.main()
