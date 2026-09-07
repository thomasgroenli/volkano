"""Tests for the volkano package facade.

The facade is mode 2 (:func:`volkano.registry`) applied to environment
variables, plus a module global and a stub *check*. These tests pin
that relationship: the environment reaches the registry *as arguments*,
nothing else in the package consults it, and importing never writes.
"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import volkano as vk
from volkano import stub


class _FakeReg:
    VK_FOO = 42

    def __dir__(self):
        return ['VkInstance', 'VkPhysicalDevice']


_URI_A = ('https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'
          '/refs/tags/v1.4.359/xml/vk.xml')
_URI_B = 'file:///work/vk.xml'


def _patch_build(factory):
    return patch('volkano.vulkan_stdlib.build_registry', factory)


def _reset():
    """Clear the singleton and anything :func:`volkano.use` set."""
    vk._registry = None
    vk._source_override = vk._UNSET
    vk._library_override = vk._UNSET


class FacadeTests(unittest.TestCase):
    """The package keeps a module-global registry; reset around every test."""

    def setUp(self):
        _reset()
        self.addCleanup(_reset)

    def test_attr_access_triggers_lazy_build_once(self):
        calls = {'n': 0}

        def fake_build(source=None, **kw):
            calls['n'] += 1
            return _FakeReg()

        with _patch_build(fake_build):
            self.assertEqual(calls['n'], 0)
            self.assertEqual(vk.VK_FOO, 42)
            self.assertEqual(vk.VK_FOO, 42)
            self.assertEqual(calls['n'], 1)

    def test_underscore_attrs_do_not_trigger_build(self):
        def fake_build(*a, **kw):
            raise AssertionError('build should not run for underscore attrs')

        with _patch_build(fake_build):
            with self.assertRaises(AttributeError):
                vk.__some_private_attr__

    def test_dir_merges_registry_names(self):
        with _patch_build(lambda *a, **kw: _FakeReg()):
            names = dir(vk)
        self.assertTrue({'registry', 'get_registry',
                         'VkInstance', 'VkPhysicalDevice'} <= set(names))

    def test_get_registry_caches(self):
        sentinel = _FakeReg()
        with _patch_build(lambda *a, **kw: sentinel):
            a = vk.get_registry()
            b = vk.get_registry()
        self.assertIs(a, b)

    def test_import_never_writes_the_stub(self):
        # Generating it forces every entry and writes a megabyte into
        # the installed package; `python -m volkano update` owns that.
        with _patch_build(lambda *a, **kw: _FakeReg()):
            with patch.object(stub, 'sync_stub',
                              side_effect=AssertionError('must not sync')):
                with patch.object(stub, 'write_stub',
                                  side_effect=AssertionError('must not write')):
                    vk.VK_FOO
                    vk.VK_FOO

    def test_a_stale_stub_is_reported_once_and_names_the_command(self):
        seen = {}

        def fake_stale(registry, **kw):
            # Checked after publishing, so a re-entrant attribute access
            # from anything it touches finds the singleton rather than
            # starting a second build.
            seen['published'] = vk._registry is registry
            return True

        with _patch_build(lambda *a, **kw: _FakeReg()):
            with patch.object(stub, 'stub_is_stale', fake_stale):
                with self.assertLogs('volkano', level='INFO') as logs:
                    vk.VK_FOO
                    vk.VK_FOO
        self.assertTrue(seen['published'])
        self.assertEqual(len(logs.records), 1)
        self.assertIn('python -m volkano update', logs.output[0])

    def test_a_broken_stub_check_never_sinks_the_import(self):
        with _patch_build(lambda *a, **kw: _FakeReg()):
            with patch.object(stub, 'stub_is_stale',
                              side_effect=OSError('read-only')):
                self.assertEqual(vk.VK_FOO, 42)


class EnvironmentTests(unittest.TestCase):
    """``get_registry`` is the only reader of VOLKANO_XML / VOLKANO_LIBRARY."""

    def setUp(self):
        _reset()
        self.addCleanup(_reset)

    def _captured_kwargs(self, env):
        """Build the singleton under exactly ``env`` and report the call.

        ``clear=True`` so a real VOLKANO_* in the developer's shell can't
        make these pass or fail by accident; the build itself is mocked,
        so nothing here needs the rest of the environment.
        """
        captured = {}

        def fake_build(source=None, *, library=None, **kw):
            captured.update(source=source, library=library)
            return _FakeReg()

        vk._registry = None
        with patch.dict('os.environ', env, clear=True):
            with _patch_build(fake_build):
                vk.get_registry()
        return captured

    def test_unset_means_main_and_autodetect(self):
        captured = self._captured_kwargs({})
        self.assertIsNone(captured['source'])       # -> resolve_source -> DEFAULT_URI
        self.assertIs(captured['library'], True)    # -> autodetect

    def test_blank_is_treated_as_unset(self):
        # `export VOLKANO_XML=` and a declared-but-empty CI variable are
        # the usual sources of this, and they must not read as values.
        captured = self._captured_kwargs({'VOLKANO_XML': '',
                                          'VOLKANO_LIBRARY': '  '})
        self.assertIsNone(captured['source'])
        self.assertIs(captured['library'], True)

    def test_xml_var_is_passed_through_as_the_source(self):
        captured = self._captured_kwargs({'VOLKANO_XML': _URI_A})
        self.assertEqual(captured['source'], _URI_A)

    def test_library_var_is_passed_through_as_a_path(self):
        captured = self._captured_kwargs(
            {'VOLKANO_LIBRARY': '/opt/vulkan/libvulkan.so.1'})
        self.assertEqual(captured['library'], '/opt/vulkan/libvulkan.so.1')

    def test_library_none_means_attach_nothing(self):
        captured = self._captured_kwargs({'VOLKANO_LIBRARY': 'none'})
        self.assertIsNone(captured['library'])

    def test_environment_is_read_at_build_time_not_import_time(self):
        # Both values are set long after `import volkano` ran, and both
        # take effect — so there is no "configure before first access"
        # ordering rule to trip over.
        first = self._captured_kwargs({'VOLKANO_XML': _URI_B})
        second = self._captured_kwargs({'VOLKANO_XML': _URI_A})
        self.assertEqual(first['source'], _URI_B)
        self.assertEqual(second['source'], _URI_A)


class ModeTwoIsolationTests(unittest.TestCase):
    """``registry()`` has no ambient inputs and no side effects."""

    EMPTY_XML = ET.fromstring('<registry/>')

    def setUp(self):
        _reset()
        self.addCleanup(_reset)

    def test_registry_does_not_populate_the_singleton(self):
        built = vk.registry(self.EMPTY_XML)
        self.assertIsNotNone(built)
        self.assertIsNone(vk._registry)

    def test_registry_never_writes_the_stub(self):
        with patch.object(stub, 'write_stub',
                          side_effect=AssertionError('mode 2 must not write')):
            with patch.object(stub, 'sync_stub',
                              side_effect=AssertionError('mode 2 must not sync')):
                vk.registry(self.EMPTY_XML)

    def test_registry_ignores_the_environment(self):
        with patch.dict('os.environ', {'VOLKANO_XML': _URI_A}, clear=False):
            captured = {}

            def fake_build(source=None, *, library=None, **kw):
                captured.update(source=source, library=library)
                return _FakeReg()

            with _patch_build(fake_build):
                vk.registry()
        self.assertIsNone(captured['source'])
        self.assertIsNone(captured['library'])


class UseTests(unittest.TestCase):
    """``use()`` is the in-process equivalent of the environment."""

    def setUp(self):
        _reset()
        self.addCleanup(_reset)

    def _captured(self, env=None):
        captured = {}

        def fake_build(source=None, *, library=None, **kw):
            captured.update(source=source, library=library)
            return _FakeReg()

        with patch.dict('os.environ', env or {}, clear=True):
            with _patch_build(fake_build):
                vk.get_registry()
        return captured

    def test_use_supplies_the_source(self):
        vk.use(_URI_A)
        self.assertEqual(self._captured()['source'], _URI_A)

    def test_use_beats_the_environment(self):
        vk.use(_URI_A)
        self.assertEqual(self._captured({'VOLKANO_XML': _URI_B})['source'],
                         _URI_A)

    def test_use_none_means_the_default_and_still_beats_the_environment(self):
        # An explicit None is a choice, not an absence.
        vk.use(None)
        self.assertIsNone(self._captured({'VOLKANO_XML': _URI_B})['source'])

    def test_an_unspecified_argument_is_left_alone(self):
        vk.use(library=None)
        captured = self._captured({'VOLKANO_XML': _URI_B})
        self.assertEqual(captured['source'], _URI_B)   # env still consulted
        self.assertIsNone(captured['library'])

    def test_calling_after_the_build_raises_rather_than_no_ops(self):
        with _patch_build(lambda *a, **kw: _FakeReg()):
            vk.VK_FOO
        with self.assertRaisesRegex(RuntimeError, 'already been built'):
            vk.use(_URI_A)

    def test_registry_still_ignores_use(self):
        # Mode 2 has no ambient inputs, and use() is an ambient input.
        vk.use(_URI_A)
        captured = {}

        def fake_build(source=None, *, library=None, **kw):
            captured.update(source=source)
            return _FakeReg()

        with _patch_build(fake_build):
            vk.registry()
        self.assertIsNone(captured['source'])


if __name__ == '__main__':
    unittest.main()
