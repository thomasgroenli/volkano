"""Tests for the source URI vocabulary, fetch and cache.

All offline: nothing here reaches the network. The download seam is
:func:`volkano.xml_source._download`, which every test patches.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from volkano import xml_source as xp


_VALID = b'<registry/>'

#: A tag URL, which volkano has no special knowledge of — it is just a
#: URI whose path happens to contain a ref. Used to check that pinning
#: needs nothing from the library beyond fetching what it is given.
_TAG_URI = ('https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'
            '/refs/tags/v1.4.359/xml/vk.xml')

#: An old tag, whose vk.xml lives at the pre-v1.2.131 path. Reachable
#: for exactly the same reason and with no fallback logic: the caller
#: names the file, so there is nothing for volkano to guess wrong.
_OLD_TAG_URI = ('https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'
                '/refs/tags/v1.0.33-core/src/spec/vk.xml')


class _CacheDirCase(unittest.TestCase):
    """Redirects the cache into a scratch directory for the test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = pathlib.Path(tmp.name)
        patcher = patch.object(xp, 'cache_dir', lambda: self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)


class ResolveSourceTests(unittest.TestCase):
    """A source is a URI or a path, and resolving reads no ambient state."""

    def test_none_is_the_default_uri(self):
        src = xp.resolve_source(None)
        self.assertEqual(src.uri, xp.DEFAULT_URI)
        self.assertFalse(src.local)

    def test_blank_is_the_default_not_the_current_directory(self):
        # Path('') is '.', which fails much later with a PermissionError.
        for blank in ('', '   ', '\t'):
            with self.subTest(blank=repr(blank)):
                self.assertEqual(xp.resolve_source(blank).uri, xp.DEFAULT_URI)

    def test_an_http_uri_is_kept_verbatim(self):
        for uri in (_TAG_URI, _OLD_TAG_URI, 'http://mirror.invalid/vk.xml'):
            with self.subTest(uri=uri):
                src = xp.resolve_source(uri)
                self.assertEqual(src.uri, uri)
                self.assertFalse(src.local)

    def test_any_host_is_acceptable(self):
        # Nothing pins volkano to Khronos but the default; a fork, a
        # mirror or an internal artifact server is the same code path.
        src = xp.resolve_source('https://intranet.example/vulkan/vk.xml')
        self.assertEqual(src.uri, 'https://intranet.example/vulkan/vk.xml')

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(xp.resolve_source(f'  {_TAG_URI} \n').uri, _TAG_URI)

    def test_a_bare_path_becomes_an_absolute_file_uri(self):
        src = xp.resolve_source('./xml/vk.xml')
        self.assertTrue(src.local)
        self.assertTrue(src.uri.startswith('file:'))
        self.assertTrue(src.path.is_absolute())

    def test_a_relative_path_does_not_depend_on_the_working_directory(self):
        # The URI lands in the stub's provenance stamp. A relative one
        # would mean something different after a cd, regenerating the
        # stub for no reason at all.
        here = pathlib.Path.cwd()
        self.assertEqual(xp.resolve_source('vk.xml').uri,
                         (here / 'vk.xml').as_uri())

    def test_a_file_uri_round_trips_to_the_same_path(self):
        target = pathlib.Path.cwd() / 'xml' / 'vk.xml'
        self.assertEqual(xp.resolve_source(target.as_uri()).path, target)

    def test_a_windows_drive_letter_is_not_a_scheme(self):
        # 'C' is one character; a scheme needs two or more.
        src = xp.resolve_source(r'C:\work\xml\vk.xml')
        self.assertTrue(src.local)
        self.assertEqual(src.uri, 'file:///C:/work/xml/vk.xml')

    def test_an_unfetchable_scheme_is_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, "'ftp' scheme"):
            xp.resolve_source('ftp://example.invalid/vk.xml')

    def test_a_remote_file_uri_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'local filesystem only'):
            xp.resolve_source('file://fileserver/share/vk.xml')

    def test_localhost_is_accepted_as_a_file_host(self):
        self.assertTrue(xp.resolve_source('file://localhost/tmp/vk.xml').local)

    def test_environment_is_ignored(self):
        # Mode 2's whole point. Only volkano.get_registry reads this.
        with patch.dict(os.environ, {xp.ENV_VAR: _TAG_URI}, clear=False):
            self.assertEqual(xp.resolve_source(None).uri, xp.DEFAULT_URI)

    def test_resolving_touches_no_filesystem(self):
        missing = pathlib.Path(tempfile.gettempdir()) / 'definitely-absent.xml'
        self.assertEqual(xp.resolve_source(str(missing)).path, missing)


class CacheNameTests(unittest.TestCase):
    """Legible names, but never at the cost of two URIs colliding."""

    def _name(self, uri):
        return xp.resolve_source(uri).path.name

    def test_the_name_recalls_the_uri(self):
        self.assertIn('refs-tags-v1.4.359', self._name(_TAG_URI))

    def test_two_layouts_of_one_ref_do_not_collide(self):
        self.assertNotEqual(self._name(_TAG_URI), self._name(_OLD_TAG_URI))

    def test_uris_differing_only_in_unsafe_characters_do_not_collide(self):
        # Sanitising alone would fold these together and silently serve
        # the wrong XML; the digest is what keeps them apart.
        a = self._name('https://e.invalid/a/b/vk.xml')
        b = self._name('https://e.invalid/a-b/vk.xml')
        self.assertNotEqual(a, b)

    def test_the_name_stays_a_sane_length(self):
        long_uri = 'https://e.invalid/' + 'segment/' * 60 + 'vk.xml'
        self.assertLessEqual(len(self._name(long_uri)), 96)

    def test_the_scheme_does_not_leak_into_the_name(self):
        self.assertNotIn('https', self._name('https://e.invalid/x/vk.xml'))

    def test_local_files_get_no_cache_entry(self):
        src = xp.resolve_source('./xml/vk.xml')
        self.assertEqual(src.path, pathlib.Path.cwd() / 'xml' / 'vk.xml')


class CacheTests(_CacheDirCase):

    def test_first_fetch_downloads_and_caches(self):
        src = xp.resolve_source(None)
        with patch.object(xp, '_download', return_value=_VALID) as dl:
            self.assertEqual(xp.ensure_cached(src), src.path)
        dl.assert_called_once_with(xp.DEFAULT_URI)
        self.assertEqual(src.path.read_bytes(), _VALID)

    def test_second_use_reads_the_cache_with_no_network(self):
        src = xp.resolve_source(None)
        with patch.object(xp, '_download', return_value=_VALID):
            xp.ensure_cached(src)
        with patch.object(xp, '_download',
                          side_effect=AssertionError('must not refetch')):
            xp.ensure_cached(src)

    def test_local_paths_are_never_cached(self):
        local = self.cache / 'hand-written.xml'
        local.write_bytes(_VALID)
        src = xp.resolve_source(str(local))
        with patch.object(xp, '_download',
                          side_effect=AssertionError('must not download')):
            self.assertEqual(xp.ensure_cached(src), local)

    def test_a_missing_local_path_says_so_and_explains_the_vocabulary(self):
        src = xp.resolve_source(str(self.cache / 'absent.xml'))
        with self.assertRaisesRegex(FileNotFoundError, 'http.s./file URI'):
            xp.ensure_cached(src)

    def test_distinct_uris_get_distinct_entries(self):
        with patch.object(xp, '_download', return_value=_VALID):
            for uri in (None, _TAG_URI, _OLD_TAG_URI):
                xp.ensure_cached(xp.resolve_source(uri))
        self.assertEqual(len(list(self.cache.iterdir())), 3)

    def test_write_is_atomic_on_failure(self):
        target = self.cache / 'entry.xml'
        target.write_bytes(b'<old/>')
        with patch('os.replace', side_effect=OSError('nope')):
            with self.assertRaises(OSError):
                xp._write_atomic(target, b'<new/>')
        self.assertEqual(target.read_bytes(), b'<old/>')
        self.assertEqual([p.name for p in self.cache.iterdir()], ['entry.xml'])

    def test_a_download_error_propagates_untouched(self):
        # There is no candidate-list fallback any more: the caller named
        # the file, so a 404 is an answer rather than a hint to guess.
        import urllib.error
        src = xp.resolve_source(_TAG_URI)
        err = urllib.error.HTTPError(_TAG_URI, 404, 'Not Found', {}, None)
        with patch.object(xp, '_download', side_effect=err) as dl:
            with self.assertRaises(urllib.error.HTTPError):
                xp.ensure_cached(src)
        dl.assert_called_once()


class RefreshTests(_CacheDirCase):

    def test_a_cold_cache_reports_fetched(self):
        with patch.object(xp, '_download', return_value=_VALID):
            self.assertEqual(xp.refresh_source(xp.resolve_source(None)),
                             'fetched')

    def test_unchanged_and_updated_are_distinguished(self):
        src = xp.resolve_source(None)
        with patch.object(xp, '_download', return_value=_VALID):
            self.assertEqual(xp.refresh_source(src), 'fetched')
            self.assertEqual(xp.refresh_source(src), 'unchanged')
        with patch.object(xp, '_download', return_value=b'<registry x="1"/>'):
            self.assertEqual(xp.refresh_source(src), 'updated')

    def test_every_remote_uri_refetches_unconditionally(self):
        # A tag URL and a branch URL are the same shape, so there is
        # nothing to infer immutability from. Explicit means explicit.
        src = xp.resolve_source(_TAG_URI)
        with patch.object(xp, '_download', return_value=_VALID) as dl:
            xp.refresh_source(src)
            xp.refresh_source(src)
        self.assertEqual(dl.call_count, 2)

    def test_local_path_reports_local(self):
        self.assertEqual(
            xp.refresh_source(xp.resolve_source('/x/vk.xml')), 'local')


class LoadXmlTests(_CacheDirCase):

    def test_element_bypasses_everything_and_has_no_provenance(self):
        element = ET.fromstring('<registry/>')
        root, provenance = xp.load_xml(element)
        self.assertIs(root, element)
        self.assertIsNone(provenance)

    def test_provenance_records_the_bytes_and_the_uri(self):
        with patch.object(xp, '_download', return_value=_VALID):
            _, provenance = xp.load_xml(_TAG_URI)
        self.assertEqual(provenance.sha256, hashlib.sha256(_VALID).hexdigest())
        self.assertEqual(provenance.uri, _TAG_URI)
        # Filled in by the parser, which is what knows the schema.
        self.assertIsNone(provenance.header_version)

    def test_provenance_for_a_local_file_is_an_absolute_uri(self):
        local = self.cache / 'hand-written.xml'
        local.write_bytes(_VALID)
        _, provenance = xp.load_xml(str(local))
        self.assertEqual(provenance.uri, local.resolve().as_uri())

    def test_corrupt_cache_entry_refetches_exactly_once(self):
        src = xp.resolve_source(None)
        src.path.parent.mkdir(parents=True, exist_ok=True)
        src.path.write_bytes(b'<truncated')
        with patch.object(xp, '_download', return_value=_VALID) as dl:
            with self.assertLogs('volkano.source', level='WARNING'):
                root, _ = xp.load_xml(None)
        self.assertEqual(root.tag, 'registry')
        dl.assert_called_once()

    def test_a_second_parse_failure_propagates(self):
        src = xp.resolve_source(None)
        src.path.write_bytes(b'<truncated')
        with patch.object(xp, '_download', return_value=b'<still broken'):
            with self.assertLogs('volkano.source', level='WARNING'):
                with self.assertRaises(ET.ParseError):
                    xp.load_xml(None)

    def test_a_corrupt_local_file_is_not_touched(self):
        local = self.cache / 'hand-written.xml'
        local.write_bytes(b'<truncated')
        with patch.object(xp, '_download',
                          side_effect=AssertionError('never fetch for a path')):
            with self.assertRaises(ET.ParseError):
                xp.load_xml(str(local))
        self.assertEqual(local.read_bytes(), b'<truncated')


if __name__ == '__main__':
    unittest.main()
