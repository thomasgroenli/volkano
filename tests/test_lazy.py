"""Tests for the thunk-graph kernel in volkano.lazy."""
from __future__ import annotations

import unittest

from volkano.lazy import (
    Lazy, Force, Call, Quote, SELF,
    _Thunk, _Raw, _Expr,
)


class LazyKernelTests(unittest.TestCase):

    def test_literal_passes_through(self):
        r = Lazy({'x': 42})
        self.assertEqual(r('x'), 42)

    def test_missing_key_raises_keyerror(self):
        with self.assertRaises(KeyError):
            Lazy({})('nope')

    def test_force_memoises(self):
        calls = {'n': 0}
        def factory():
            calls['n'] += 1
            return object()
        r = Lazy({'x': _Thunk(factory)})
        a = r('x')
        b = r('x')
        self.assertIs(a, b)
        self.assertEqual(calls['n'], 1)

    def test_self_resolves_to_registry(self):
        r = Lazy({'me': _Thunk(lambda reg: reg, SELF)})
        self.assertIs(r('me'), r)

    def test_thunk_self_does_cached_lookup(self):
        r = Lazy({'a': 7, 'b': _Thunk(SELF, 'a')})
        self.assertEqual(r('b'), 7)

    def test_raw_value_not_descended(self):
        payload = [_Thunk(lambda: 1 / 0)]  # would explode if descended
        r = Lazy({'x': _Thunk(lambda v: v, _Raw(payload))})
        self.assertIs(r('x'), payload)

    def test_list_and_dict_descend_elementwise(self):
        r = Lazy({
            'a': 1,
            'shape': [_Thunk(SELF, 'a'), {'k': _Thunk(SELF, 'a')}],
        })
        self.assertEqual(r('shape'), [1, {'k': 1}])

    def test_nested_thunks_evaluate_inside_out(self):
        r = Lazy({'x': _Thunk(lambda y: y + 1, _Thunk(lambda: 10))})
        self.assertEqual(r('x'), 11)

    def test_cycle_detection_raises(self):
        r = Lazy({
            'a': _Thunk(SELF, 'b'),
            'b': _Thunk(SELF, 'a'),
        })
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            r('a')

    def test_exception_clears_resolving_marker(self):
        """A failing force must not leave the key stuck in _RESOLVING."""
        def boom():
            raise RuntimeError('boom')
        r = Lazy({'x': _Thunk(boom)})
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            r('x')
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            r('x')

    def test_mapping_protocol(self):
        r = Lazy({'a': 1, 'b': 2})
        self.assertEqual(len(r), 2)
        self.assertIn('a', r)
        self.assertNotIn('missing', r)
        self.assertEqual(sorted(r), ['a', 'b'])

    def test_getitem_returns_raw_unforced(self):
        raw = _Thunk(lambda: 99)
        r = Lazy({'x': raw})
        self.assertIs(r['x'], raw)
        self.assertEqual(r._cache, {})


class BuilderHelperTests(unittest.TestCase):

    def test_force_helper(self):
        r = Lazy({'a': 7, 'b': Force('a')})
        self.assertEqual(r('b'), 7)

    def test_call_helper_resolves_factory_by_name(self):
        r = Lazy({
            'mul': lambda a, b: a * b,
            'a': 3,
            'out': Call('mul', [Force('a'), 4]),
        })
        self.assertEqual(r('out'), 12)

    def test_call_with_registry_prepends_self(self):
        r = Lazy({
            'echo': lambda reg, name: (reg, name),
            'r': Call('echo', ['hi'], with_registry=True),
        })
        got_reg, got_name = r('r')
        self.assertIs(got_reg, r)
        self.assertEqual(got_name, 'hi')

    def test_quote_skips_descent(self):
        inner = _Thunk(lambda: 1 / 0)
        seen: list = []
        def factory(payload):
            seen.append(payload)
            return payload
        r = Lazy({'x': _Thunk(factory, Quote([inner]))})
        out = r('x')
        self.assertEqual(out, [inner])
        self.assertIs(seen[0][0], inner)


class ExprTests(unittest.TestCase):
    """Python-eval expression nodes with the registry as locals."""

    def test_literal_expression(self):
        r = Lazy({'x': _Expr('42')})
        self.assertEqual(r('x'), 42)

    def test_arithmetic_expression(self):
        r = Lazy({'x': _Expr('(2 + 3) * 4')})
        self.assertEqual(r('x'), 20)

    def test_resolves_identifiers_via_registry(self):
        r = Lazy({'A': 10, 'B': 20, 'sum': _Expr('A + B')})
        self.assertEqual(r('sum'), 30)

    def test_bitshift_and_or_with_precedence(self):
        r = Lazy({'mask': _Expr('(1 << 5) | (1 << 3)')})
        self.assertEqual(r('mask'), 0b101000)

    def test_calls_callable_resolved_from_registry(self):
        r = Lazy({'double': lambda x: x * 2, 'v': _Expr('double(7)')})
        self.assertEqual(r('v'), 14)

    def test_python_keywords_are_not_resolved(self):
        # 'True' / 'and' should pass through to Python without registry lookup.
        r = Lazy({'flag': _Expr('True and 1 or 0')})
        self.assertEqual(r('flag'), 1)


if __name__ == '__main__':
    unittest.main()
