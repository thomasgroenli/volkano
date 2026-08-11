"""Tests for the thunk-graph kernel in volkano.lazy."""
from __future__ import annotations

import pathlib
import threading
import time
import unittest

from volkano import lazy
from volkano.lazy import (
    Lazy, Force, Call, Quote, SELF,
    _Thunk, _Raw,
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


class NoInterpreterTests(unittest.TestCase):
    """The resolution path holds no expression form and no eval().

    The kernel once carried one. The Vulkan parser never emitted a
    single node of it, so it was pure attack surface: registry XML is
    third-party input, and it should not be able to reach an
    interpreter. Arithmetic vk.xml genuinely contains is folded by
    hand in xml_parser instead.
    """

    def test_the_kernel_exposes_no_expression_node(self):
        self.assertFalse([n for n in dir(lazy) if 'Expr' in n])

    def test_no_eval_in_the_module(self):
        source = pathlib.Path(lazy.__file__).read_text(encoding='utf-8')
        self.assertNotIn('eval(', source)

    def test_a_string_value_stays_a_string(self):
        # Not parsed, not evaluated -- forced literals pass through.
        r = Lazy({'x': '(2 + 3) * 4'})
        self.assertEqual(r('x'), '(2 + 3) * 4')


class ConcurrentForceTests(unittest.TestCase):
    """Forcing is memoised *per registry*, not per thread.

    Vulkan type identity depends on it: if two threads racing one key
    each built a value, the loser's ctypes class would be a different
    object with the same name, and a struct field typed by one would
    reject an instance of the other.
    """

    THREADS = 16

    def _race(self, registry, key):
        """Force ``key`` from many threads at once; return their results.

        Failures are collected rather than left to die in the worker, so
        a thread that raises makes the test fail loudly. Without the
        lock, latecomers observe another thread's ``_RESOLVING`` sentinel
        and raise a spurious "cyclic key reference" — invisible here if
        only the successes were counted.
        """
        results: list = []
        errors: list = []
        barrier = threading.Barrier(self.THREADS)

        def run():
            barrier.wait()
            try:
                results.append(registry(key))
            except BaseException as exc:      # noqa: BLE001 - reported below
                errors.append(exc)

        workers = [threading.Thread(target=run) for _ in range(self.THREADS)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), self.THREADS)
        return results

    def test_racing_threads_share_one_value(self):
        calls = []

        def slow_factory():
            # Guarantees an overlap window rather than relying on luck.
            calls.append(1)
            time.sleep(0.05)
            return object()

        registry = Lazy({'x': _Thunk(slow_factory)})
        results = self._race(registry, 'x')
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(set(map(id, results))), 1)

    def test_racing_threads_see_a_nested_graph_consistently(self):
        def slow(value):
            time.sleep(0.02)
            return [value]

        registry = Lazy({
            'leaf': _Thunk(lambda: object()),
            'mid': _Thunk(slow, Force('leaf')),
            'top': _Thunk(slow, Force('mid')),
        })
        results = self._race(registry, 'top')
        self.assertEqual(len(set(map(id, results))), 1)

    def test_a_real_cycle_still_raises_rather_than_deadlocking(self):
        # The lock is re-entrant, so an owner re-entering its own frame
        # reaches the _RESOLVING sentinel instead of blocking forever.
        registry = Lazy({'a': Force('b'), 'b': Force('a')})
        with self.assertRaises(ValueError):
            registry('a')

    def test_self_reference_raises(self):
        registry = Lazy({'a': Force('a')})
        with self.assertRaises(ValueError):
            registry('a')

    def test_a_failed_force_does_not_poison_the_cache(self):
        attempts = {'n': 0}

        def flaky():
            attempts['n'] += 1
            if attempts['n'] == 1:
                raise RuntimeError('first attempt fails')
            return 'ok'

        registry = Lazy({'x': _Thunk(flaky)})
        with self.assertRaises(RuntimeError):
            registry('x')
        self.assertEqual(registry('x'), 'ok')


if __name__ == '__main__':
    unittest.main()
