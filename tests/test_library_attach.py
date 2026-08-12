"""Tests for deferred loader attachment.

The loader spec is held, not opened, until the first command is
*called*. Resolution never needs it — a command signature is data like
any other registry entry, so the whole registry (commands included)
forces on a machine with no Vulkan driver, which is what stub
generation walks. Only dispatch genuinely needs a DSO.
"""
from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from volkano.lazy import SELF, _Raw, _Thunk
from volkano.vulkan_stdlib import (
    STDLIB, CommandSignature, VkRegistry, make_command,
)


class _FakeDll:
    """Stands in for an open CDLL: attribute access yields a callable."""

    def __init__(self, names):
        self._fns = {name: (lambda n=name: (lambda *a: n))() for name in names}

    def __getattr__(self, name):
        try:
            return self._fns[name]
        except KeyError:
            raise AttributeError(name)


def _command(name):
    return _Thunk(make_command, SELF, name, None, _Raw(()), _Raw({}))


def _registry(library=None, names=('vkAlpha', 'vkBeta')):
    data = {**STDLIB, 'VK_FOO': 7}
    data.update({name: _command(name) for name in names})
    return VkRegistry(data, library=library)


class DeferralTests(unittest.TestCase):

    def test_resolution_never_opens_the_loader(self):
        registry = _registry(library='/nonexistent/libvulkan.so.1')
        with patch.object(VkRegistry, '_open_library',
                          side_effect=AssertionError('must not open')):
            self.assertEqual(registry.VK_FOO, 7)
            sig = registry.vkAlpha
            # Reading the signature back is inspection, not dispatch.
            self.assertIsNone(sig._fn)
            self.assertEqual(sig.name, 'vkAlpha')
            repr(sig)
        self.assertFalse(registry._library_resolved)

    def test_first_command_call_opens_the_loader(self):
        dll = _FakeDll(['vkAlpha'])
        registry = _registry(library=dll)
        sig = registry.vkAlpha
        self.assertFalse(registry._library_resolved)
        self.assertEqual(sig(), 'vkAlpha')
        self.assertTrue(registry._library_resolved)
        self.assertIsNotNone(sig._fn)

    def test_the_loader_is_opened_exactly_once(self):
        dll = _FakeDll(['vkAlpha', 'vkBeta'])
        registry = _registry(library=dll)
        with patch.object(VkRegistry, '_open_library',
                          return_value=dll) as opener:
            registry.vkAlpha()
            registry.vkAlpha()
            registry.vkBeta()
        opener.assert_called_once()

    def test_a_bound_command_does_not_re_enter_the_bind_path(self):
        registry = _registry(library=_FakeDll(['vkAlpha']))
        sig = registry.vkAlpha
        sig()
        with patch.object(CommandSignature, '_bind_now',
                          side_effect=AssertionError('already bound')):
            self.assertEqual(sig(), 'vkAlpha')

    def test_library_none_opens_nothing(self):
        registry = _registry(library=None)
        with patch.object(VkRegistry, '_open_library',
                          side_effect=AssertionError('nothing to open')):
            with self.assertRaises(RuntimeError):
                registry.vkAlpha()


class OpenFailureTests(unittest.TestCase):
    """A missing loader must never break resolution — only calling."""

    def test_forcing_a_command_does_not_even_warn(self):
        registry = _registry(library='/nonexistent/libvulkan.so.1')
        with patch.object(VkRegistry, '_open_library',
                          side_effect=AssertionError('must not open')):
            registry.vkAlpha
            registry.vkBeta

    def test_calls_warn_once_and_raise(self):
        registry = _registry(library='/nonexistent/libvulkan.so.1')
        with self.assertLogs('volkano.stdlib', level='WARNING') as logs:
            with self.assertRaises(RuntimeError):
                registry.vkAlpha()
            with self.assertRaises(RuntimeError):
                registry.vkBeta()
        warnings = [r for r in logs.records if r.levelname == 'WARNING']
        self.assertEqual(len(warnings), 1)

    def test_a_failed_open_is_not_retried(self):
        registry = _registry(library='/nonexistent/libvulkan.so.1')
        with self.assertLogs('volkano.stdlib', level='WARNING'):
            with self.assertRaises(RuntimeError):
                registry.vkAlpha()
        with patch.object(VkRegistry, '_open_library',
                          side_effect=AssertionError('must not retry')):
            with self.assertRaises(RuntimeError):
                registry.vkBeta()

    def test_forcing_everything_survives_without_a_driver(self):
        # The property stub generation depends on: write_stub forces
        # every key inside a blanket except-Exception, so a raise here
        # would silently reduce every command to `Any`. With binding
        # deferred to the call, the loader isn't even consulted.
        registry = _registry(library='/nonexistent/libvulkan.so.1')
        with patch.object(VkRegistry, '_open_library',
                          side_effect=AssertionError('must not open')):
            forced = [registry(key) for key in registry
                      if not key.startswith('_')]
        self.assertTrue(any(isinstance(v, CommandSignature) for v in forced))


class UnboundReasonTests(unittest.TestCase):
    """Three causes look identical from the call site; the message must not."""

    def test_no_library(self):
        sig = _registry(library=None).vkAlpha
        with self.assertRaisesRegex(RuntimeError, 'built with library=None'):
            sig()

    def test_open_failed(self):
        registry = _registry(library='/nonexistent/libvulkan.so.1')
        sig = registry.vkAlpha
        with self.assertLogs('volkano.stdlib', level='WARNING'):
            with self.assertRaisesRegex(RuntimeError, 'could not be opened'):
                sig()

    def test_not_exported(self):
        # The loader opened fine; this symbol simply isn't in it.
        sig = _registry(library=_FakeDll([])).vkAlpha
        with self.assertRaisesRegex(RuntimeError, "doesn't export"):
            sig()

    def test_repr_does_not_diagnose_before_a_bind_is_tried(self):
        sig = _registry(library=_FakeDll([])).vkAlpha
        self.assertIn('binds on call', repr(sig))
        with self.assertRaises(RuntimeError):
            sig()
        self.assertIn('not-exported', repr(sig))


class AttachLibraryTests(unittest.TestCase):
    """The explicit call stays eager and raising — it's a deliberate act."""

    def test_attach_raises_on_a_bad_path(self):
        registry = _registry(library=None)
        with self.assertRaises(OSError):
            registry.attach_library('/nonexistent/libvulkan.so.1')

    def test_attach_after_a_force_needs_no_rebind(self):
        registry = _registry(library=None)
        sig = registry.vkAlpha
        self.assertIsNone(sig._fn)
        registry.attach_library(_FakeDll(['vkAlpha']))
        self.assertEqual(sig(), 'vkAlpha')

    def test_attach_after_a_failed_call_binds_on_the_next_call(self):
        registry = _registry(library=None)
        sig = registry.vkAlpha
        with self.assertRaises(RuntimeError):
            sig()
        registry.attach_library(_FakeDll(['vkAlpha']))
        self.assertEqual(sig(), 'vkAlpha')

    def test_attach_then_rebind_binds_forced_commands(self):
        registry = _registry(library=None)
        sig = registry.vkAlpha
        self.assertIsNone(sig._fn)
        registry.attach_library(_FakeDll(['vkAlpha']))
        self.assertEqual(registry.rebind_commands(), 1)
        self.assertIsNotNone(sig._fn)

    def test_rebind_realises_a_deferred_spec(self):
        dll = _FakeDll(['vkAlpha'])
        registry = _registry(library=dll)
        registry._cache['vkAlpha'] = CommandSignature('vkAlpha', None, [], {})
        self.assertEqual(registry.rebind_commands(), 1)


class ConcurrencyTests(unittest.TestCase):
    """Deferral must not hand one thread a command bound against None."""

    def test_racing_command_calls_all_dispatch(self):
        names = tuple(f'vkCmd{i}' for i in range(16))
        dll = _FakeDll(list(names))
        registry = _registry(library=dll, names=names)
        sigs = [registry(name) for name in names]

        release = threading.Event()
        opens = []
        real_open = VkRegistry._open_library

        def slow_open(self, spec):
            # Widen the window between "resolution started" and "handle
            # published" — the exact interval the lock has to cover. A
            # reader that saw the flag without the handle would bind
            # against None and produce a silently unusable command.
            opens.append(spec)
            release.wait(0.5)
            return real_open(self, spec)

        results: list = []
        barrier = threading.Barrier(16)

        def call(i):
            barrier.wait()
            results.append(sigs[i]())

        with patch.object(VkRegistry, '_open_library', slow_open):
            threads = [threading.Thread(target=call, args=(i,))
                       for i in range(16)]
            for t in threads:
                t.start()
            release.set()
            for t in threads:
                t.join()

        self.assertCountEqual(results, names)
        self.assertTrue(all(sig._fn is not None for sig in sigs))
        self.assertEqual(len(opens), 1)


if __name__ == '__main__':
    unittest.main()
