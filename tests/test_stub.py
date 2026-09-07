"""Tests for stub generation and provenance-driven regeneration."""
from __future__ import annotations

import ast
import os
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from volkano import cbase, stub, vulkan_stdlib
from volkano.xml_source import Provenance


_SHA_A = 'a' * 64
_SHA_B = 'b' * 64

_URI_A = ('https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'
          '/refs/heads/main/xml/vk.xml')
_URI_B = 'file:///work/vk.xml'


class _FakeRegistry:
    """Minimal stand-in for VkRegistry: iterable of keys, callable to force."""

    def __init__(self, data, provenance=None):
        self._data = dict(data)
        self._provenance = provenance

    def __iter__(self):
        return iter(self._data)

    def __call__(self, key):
        return self._data[key]


class _StubPathCase(unittest.TestCase):
    """Gives each test a scratch path to write a stub to."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(tmp.name, '__init__.pyi')

    def body(self):
        return pathlib.Path(self.path).read_text(encoding='utf-8')


class WriteStubTests(_StubPathCase):

    def test_writes_declarations_from_explicit_registry(self):
        registry = _FakeRegistry({'VK_FOO': 42, '_private': 1})
        count = stub.write_stub(self.path, registry=registry)
        self.assertEqual(count, 1)                      # '_private' skipped
        self.assertIn('VK_FOO: int = 42', self.body())
        self.assertNotIn('_private', self.body())

    def test_explicit_registry_bypasses_the_singleton(self):
        with patch('volkano.get_registry',
                   side_effect=AssertionError('should not be consulted')):
            stub.write_stub(self.path, registry=_FakeRegistry({'VK_FOO': 1}))

    def test_stamp_round_trips(self):
        provenance = Provenance(_SHA_A, _URI_A, 359)
        stub.write_stub(self.path, registry=_FakeRegistry({'VK_FOO': 1}),
                        provenance=provenance)
        self.assertEqual(stub.read_stub_provenance(self.path),
                         (_SHA_A, stub._GENERATOR, _URI_A))

    def test_unstamped_stub_reads_as_no_provenance(self):
        stub.write_stub(self.path, registry=_FakeRegistry({'VK_FOO': 1}))
        self.assertIsNone(stub.read_stub_provenance(self.path))

    def test_missing_file_reads_as_no_provenance(self):
        self.assertIsNone(stub.read_stub_provenance(self.path))

    def test_failure_mid_emit_leaves_the_old_stub_intact(self):
        stub.write_stub(self.path, registry=_FakeRegistry({'VK_FOO': 1}))
        original = self.body()
        with patch.object(stub, '_emit_constant', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                stub.write_stub(self.path, registry=_FakeRegistry({'VK_BAR': 2}))
        self.assertEqual(self.body(), original)
        # ...and no temp file was left lying next to it.
        self.assertEqual(os.listdir(self.dir), ['__init__.pyi'])


class EnumerantStubTests(_StubPathCase):
    """An enumerant is an int, so it has to be classified before ints."""

    def setUp(self):
        super().setUp()
        self.group = cbase.Enum.create(
            'VkFoo', values=[('VK_FOO_A', 0)], aliases=[('VK_FOO_A_KHR', 'VK_FOO_A')])

    def test_member_classifies_as_enum_member_not_int(self):
        self.assertEqual(stub._classify(self.group.VK_FOO_A), 'enum_member')

    def test_member_renders_as_the_group_attribute(self):
        registry = _FakeRegistry({'VkFoo': self.group,
                                  'VK_FOO_A': self.group.VK_FOO_A})
        stub.write_stub(self.path, registry=registry)
        self.assertIn('VK_FOO_A: VkFoo = VkFoo.VK_FOO_A', self.body())

    def test_alias_renders_as_the_canonical_member(self):
        registry = _FakeRegistry({'VkFoo': self.group,
                                  'VK_FOO_A_KHR': self.group.VK_FOO_A_KHR})
        stub.write_stub(self.path, registry=registry)
        self.assertIn('VK_FOO_A_KHR: VkFoo = VkFoo.VK_FOO_A', self.body())

    def test_group_is_declared_before_the_enumerants_that_name_it(self):
        registry = _FakeRegistry({'VkFoo': self.group,
                                  'VK_FOO_A': self.group.VK_FOO_A})
        stub.write_stub(self.path, registry=registry)
        body = self.body()
        self.assertLess(body.index('class VkFoo('), body.index('VK_FOO_A: VkFoo'))

    def test_stub_is_valid_python(self):
        registry = _FakeRegistry({'VkFoo': self.group,
                                  'VK_FOO_A': self.group.VK_FOO_A})
        stub.write_stub(self.path, registry=registry)
        ast.parse(self.body())

    def test_composite_flag_value_falls_back_to_int(self):
        flags = cbase.Enum.create('VkBarFlagBits',
                                  values=[('VK_BAR_A', 1), ('VK_BAR_B', 2)],
                                  flag=True)
        # Nameless on 3.10, named 'VK_BAR_A|VK_BAR_B' from 3.11 on -
        # neither spells as a single attribute access, so both render int.
        composite = flags.VK_BAR_A | flags.VK_BAR_B
        self.assertEqual(stub._classify(composite), 'int')

    def test_declared_composite_enumerant_is_still_a_member(self):
        # vk.xml does declare these (VK_CULL_MODE_FRONT_AND_BACK = 3);
        # a *declared* composite is a real member and keeps its name.
        flags = cbase.Enum.create('VkBazFlagBits',
                                  values=[('VK_BAZ_A', 1), ('VK_BAZ_B', 2),
                                          ('VK_BAZ_BOTH', 3)],
                                  flag=True)
        self.assertEqual(stub._classify(flags.VK_BAZ_BOTH), 'enum_member')


class SyncStubTests(_StubPathCase):
    """The stamp, not "did we just download", is what triggers a rewrite."""

    def _registry(self, sha, uri=_URI_A):
        return _FakeRegistry({'VK_FOO': 1},
                             provenance=Provenance(sha, uri, 359))

    def test_missing_stub_is_written(self):
        self.assertTrue(stub.sync_stub(self._registry(_SHA_A), path=self.path))
        self.assertIn('Auto-generated stub', self.body())

    def test_matching_stamp_is_a_no_op(self):
        stub.sync_stub(self._registry(_SHA_A), path=self.path)
        self.assertFalse(stub.sync_stub(self._registry(_SHA_A), path=self.path))

    def test_changed_sha_rewrites(self):
        stub.sync_stub(self._registry(_SHA_A), path=self.path)
        self.assertTrue(stub.sync_stub(self._registry(_SHA_B), path=self.path))
        self.assertEqual(stub.read_stub_provenance(self.path),
                         (_SHA_B, stub._GENERATOR, _URI_A))

    def test_changed_uri_rewrites_even_when_bytes_match(self):
        stub.sync_stub(self._registry(_SHA_A, _URI_A), path=self.path)
        self.assertTrue(
            stub.sync_stub(self._registry(_SHA_A, _URI_B), path=self.path))

    def test_bumped_generator_rewrites_unchanged_xml(self):
        # Improving the parser or an emitter changes the stub's content
        # without changing the XML. Keying only on the digest would
        # leave the stub contradicting the runtime.
        stub.sync_stub(self._registry(_SHA_A), path=self.path)
        with patch.object(stub, '_GENERATOR', stub._GENERATOR + 1):
            self.assertTrue(stub.sync_stub(self._registry(_SHA_A), path=self.path))

    def test_stub_predating_the_stamp_format_is_rewritten(self):
        pathlib.Path(self.path).write_text('# volkano-stub: sha256=%s '
                                           'header-version=359 source=%s\n'
                                           % (_SHA_A, _URI_A), encoding='utf-8')
        self.assertIsNone(stub.read_stub_provenance(self.path))
        self.assertTrue(stub.sync_stub(self._registry(_SHA_A), path=self.path))

    def test_registry_without_provenance_never_writes(self):
        self.assertFalse(stub.sync_stub(_FakeRegistry({'VK_FOO': 1}),
                                        path=self.path))
        self.assertFalse(os.path.exists(self.path))

    def test_write_failure_does_not_raise(self):
        with patch.object(stub, 'write_stub',
                          side_effect=OSError('read-only file system')):
            with self.assertLogs('volkano.stub', level='WARNING'):
                wrote = stub.sync_stub(self._registry(_SHA_A), path=self.path)
        self.assertFalse(wrote)

    def test_non_oserror_failure_is_also_swallowed(self):
        # os.replace onto a .pyi held open by a language server raises
        # PermissionError on Windows; an emitter bug raises anything.
        with patch.object(stub, 'write_stub', side_effect=RuntimeError('boom')):
            with self.assertLogs('volkano.stub', level='WARNING'):
                self.assertFalse(
                    stub.sync_stub(self._registry(_SHA_A), path=self.path))


class StubIsStaleTests(_StubPathCase):
    """The cheap check the facade runs on every build."""

    def _registry(self, sha, uri=_URI_A):
        return _FakeRegistry({'VK_FOO': 1},
                             provenance=Provenance(sha, uri, 359))

    def test_a_missing_stub_is_stale(self):
        self.assertTrue(stub.stub_is_stale(self._registry(_SHA_A),
                                           path=self.path))

    def test_a_freshly_written_stub_is_not_stale(self):
        stub.sync_stub(self._registry(_SHA_A), path=self.path)
        self.assertFalse(stub.stub_is_stale(self._registry(_SHA_A),
                                            path=self.path))

    def test_changed_bytes_or_uri_go_stale(self):
        stub.sync_stub(self._registry(_SHA_A), path=self.path)
        self.assertTrue(stub.stub_is_stale(self._registry(_SHA_B),
                                           path=self.path))
        self.assertTrue(stub.stub_is_stale(self._registry(_SHA_A, _URI_B),
                                           path=self.path))

    def test_no_provenance_is_never_stale(self):
        # An Element fixture has no identity to be out of date with.
        self.assertFalse(stub.stub_is_stale(_FakeRegistry({'VK_FOO': 1}),
                                            path=self.path))

    def test_checking_never_writes(self):
        with patch.object(stub, 'write_stub',
                          side_effect=AssertionError('must not write')):
            stub.stub_is_stale(self._registry(_SHA_A), path=self.path)
        self.assertFalse(os.path.exists(self.path))


class BuildRegistryLeavesStubAloneTests(_StubPathCase):
    """Nothing in the build path may reach for the stub."""

    #: Parsing an Element skips the network entirely, so these builds are
    #: cheap and offline while still exercising the real code path.
    EMPTY_XML = ET.fromstring('<registry/>')

    def test_build_never_writes_a_stub(self):
        with patch.object(stub, 'write_stub',
                          side_effect=AssertionError('build must not write')):
            vulkan_stdlib.build_registry(self.EMPTY_XML)

    def test_element_build_has_no_provenance(self):
        registry = vulkan_stdlib.build_registry(self.EMPTY_XML)
        self.assertIsNone(registry._provenance)
        # ...which is precisely what makes sync_stub a no-op for fixtures.
        self.assertFalse(stub.sync_stub(registry, path=self.path))


if __name__ == '__main__':
    unittest.main()
