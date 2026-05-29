"""Tests for the volkano.vk module facade."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from volkano import vk


class _FakeReg:
    VK_FOO = 42

    def __dir__(self):
        return ['VkInstance', 'VkPhysicalDevice']


def _patch_build(factory):
    return patch('volkano.vulkan_stdlib.build_registry', factory)


class FacadeTests(unittest.TestCase):
    """vk.py keeps a module-global registry; reset around every test."""

    def setUp(self):
        vk._registry = None
        vk._build_kwargs = {'library': True}

    def tearDown(self):
        vk._registry = None
        vk._build_kwargs = {'library': True}

    def test_attr_access_triggers_lazy_build_once(self):
        calls = {'n': 0}

        def fake_build(**kw):
            calls['n'] += 1
            return _FakeReg()

        with _patch_build(fake_build):
            self.assertEqual(calls['n'], 0)
            self.assertEqual(vk.VK_FOO, 42)
            self.assertEqual(vk.VK_FOO, 42)
            self.assertEqual(calls['n'], 1)

    def test_underscore_attrs_do_not_trigger_build(self):
        def fake_build(**kw):
            raise AssertionError('build should not run for underscore attrs')

        with _patch_build(fake_build):
            with self.assertRaises(AttributeError):
                vk.__some_private_attr__

    def test_configure_locked_after_first_access(self):
        with _patch_build(lambda **kw: _FakeReg()):
            _ = vk.VK_FOO
        with self.assertRaisesRegex(RuntimeError, 'already been built'):
            vk.configure(source='foo')

    def test_configure_reset_clears_singleton(self):
        with _patch_build(lambda **kw: _FakeReg()):
            _ = vk.VK_FOO
        vk.configure(source='other.xml', reset=True)
        self.assertIsNone(vk._registry)
        self.assertEqual(
            vk._build_kwargs,
            {'source': 'other.xml', 'library': True, 'refresh': False})

    def test_dir_merges_registry_names(self):
        with _patch_build(lambda **kw: _FakeReg()):
            names = dir(vk)
        self.assertTrue({'configure', 'get_registry',
                         'VkInstance', 'VkPhysicalDevice'} <= set(names))

    def test_get_registry_caches(self):
        sentinel = object()
        with _patch_build(lambda **kw: sentinel):
            a = vk.get_registry()
            b = vk.get_registry()
        self.assertIs(a, b)


if __name__ == '__main__':
    unittest.main()
