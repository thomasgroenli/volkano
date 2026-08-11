"""Tests for the source vocabulary, fetch and cache.

All offline: nothing here reaches the network. The download seam is
:func:`volkano.xml_source._download`, which every test patches.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from unittest.mock import patch

from volkano import xml_source as xp


_VALID = b'<registry/>'


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
    """The vocabulary is closed, prefix-tagged, and reads no ambient state."""

    def test_none_is_the_default_branch(self):
        src = xp.resolve_source(None)
        self.assertEqual(src.kind, 'branch')
        self.assertEqual(src.label, 'main')
        self.assertEqual(src.url, xp.MAIN_URL)
        self.assertTrue(src.mutable)

    def test_bare_main_is_the_default_branch(self):
        self.assertEqual(xp.resolve_source('main'), xp.resolve_source(None))

    def test_blank_is_main_not_the_current_directory(self):
        # Path('') is '.', which fails much later with a PermissionError.
        for blank in ('', '   ', '\t'):
            with self.subTest(blank=repr(blank)):
                self.assertEqual(xp.resolve_source(blank).kind, 'branch')

    def test_branch_prefix(self):
        src = xp.resolve_source('branch:sc_main')
        self.assertEqual((src.kind, src.ref), ('branch', 'sc_main'))
        self.assertIn('/refs/heads/sc_main/', src.url)
        self.assertTrue(src.mutable)

    def test_tag_prefix_is_immutable(self):
        src = xp.resolve_source('tag:v1.4.359')
        self.assertEqual((src.kind, src.ref), ('tag', 'v1.4.359'))
        self.assertIn('/refs/tags/v1.4.359/', src.url)
        self.assertFalse(src.mutable)

    def test_tag_names_pass_through_verbatim(self):
        # Plain vX.Y.Z only starts at v1.1.70; 89 of 382 published tags
        # use these older shapes, and v1.0.33 has never existed. Any
        # scheme that rebuilds a tag from a parsed version 404s on them.
        for ref in ('v1.0.33-core', 'v1.0-core-20161025',
                    'v1.0-core+wsi-20160216', 'v1.4.359'):
            with self.subTest(ref=ref):
                src = xp.resolve_source('tag:' + ref)
                self.assertEqual(src.ref, ref)
                self.assertIn('/refs/tags/' + ref + '/', src.url)

    def test_empty_ref_after_a_prefix_is_rejected(self):
        for text in ('tag:', 'branch:', 'tag:  '):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, 'no ref'):
                    xp.resolve_source(text)

    def test_anything_else_is_a_local_path(self):
        src = xp.resolve_source('./xml/vk.xml')
        self.assertEqual(src.kind, 'path')
        self.assertIsNone(src.url)
        self.assertIsNone(src.ref)

    def test_a_windows_path_is_not_mistaken_for_a_prefix(self):
        src = xp.resolve_source('C:\\work\\xml\\vk.xml')
        self.assertEqual(src.kind, 'path')

    def test_urls_are_rejected_with_guidance(self):
        with self.assertRaisesRegex(ValueError, 'not an accepted source'):
            xp.resolve_source('https://example.invalid/vk.xml')

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(xp.resolve_source('  tag:v1.4.359 \n').ref, 'v1.4.359')

    def test_environment_is_ignored(self):
        # Mode 2's whole point. Only volkano.get_registry reads this.
        with patch.dict(os.environ, {xp.ENV_VAR: 'tag:v1.4.359'}, clear=False):
            self.assertEqual(xp.resolve_source(None).kind, 'branch')


class CacheNameTests(unittest.TestCase):
    """Legible names, but never at the cost of two refs colliding."""

    def test_kind_is_part_of_the_name(self):
        # A tag and a branch may share a name; they must not share a file.
        self.assertNotEqual(xp.resolve_source('tag:main').path,
                            xp.resolve_source('branch:main').path)

    def test_well_behaved_refs_keep_a_clean_name(self):
        self.assertEqual(xp.resolve_source('tag:v1.4.359').path.name,
                         'tag-v1.4.359.xml')
        self.assertEqual(xp.resolve_source('branch:main').path.name,
                         'branch-main.xml')

    def test_plus_is_kept(self):
        self.assertEqual(
            xp.resolve_source('tag:v1.0-core+wsi-20160216').path.name,
            'tag-v1.0-core+wsi-20160216.xml')

    def test_path_separators_are_sanitised_without_colliding(self):
        slashed = xp.resolve_source('branch:feature/foo').path.name
        dashed = xp.resolve_source('branch:feature-foo').path.name
        self.assertNotIn('/', slashed)
        self.assertNotEqual(slashed, dashed)


class CacheTests(_CacheDirCase):

    def test_first_fetch_downloads_and_caches(self):
        src = xp.resolve_source('main')
        with patch.object(xp, '_download', return_value=_VALID) as dl:
            self.assertEqual(xp.ensure_cached(src),
                             self.cache / 'branch-main.xml')
        dl.assert_called_once_with(xp.MAIN_URL)
        self.assertEqual((self.cache / 'branch-main.xml').read_bytes(), _VALID)

    def test_second_use_reads_the_cache_with_no_network(self):
        src = xp.resolve_source('main')
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

    def test_a_missing_local_path_says_so(self):
        src = xp.resolve_source(str(self.cache / 'absent.xml'))
        with self.assertRaises(FileNotFoundError):
            xp.ensure_cached(src)

    def test_a_bare_version_gets_a_tag_hint(self):
        # The old vocabulary accepted '1.4.359'; it is now a path, so
        # the error should point at what replaced it.
        with self.assertRaisesRegex(FileNotFoundError, 'tag:v1\\.4\\.359'):
            xp.ensure_cached(xp.resolve_source('1.4.359'))

    def test_refs_get_their_own_entries(self):
        with patch.object(xp, '_download', return_value=_VALID):
            xp.ensure_cached(xp.resolve_source('main'))
            xp.ensure_cached(xp.resolve_source('tag:v1.4.359'))
            xp.ensure_cached(xp.resolve_source('tag:v1.0.33-core'))
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()),
                         ['branch-main.xml', 'tag-v1.0.33-core.xml',
                          'tag-v1.4.359.xml'])

    def test_write_is_atomic_on_failure(self):
        target = self.cache / 'branch-main.xml'
        target.write_bytes(b'<old/>')
        with patch('os.replace', side_effect=OSError('nope')):
            with self.assertRaises(OSError):
                xp._write_atomic(target, b'<new/>')
        self.assertEqual(target.read_bytes(), b'<old/>')
        self.assertEqual([p.name for p in self.cache.iterdir()],
                         ['branch-main.xml'])


class RepoLayoutFallbackTests(_CacheDirCase):
    """Khronos moved vk.xml from src/spec/ to xml/ between v1.1.70 and v1.2.131."""

    def _http_error(self, code):
        return urllib.error.HTTPError('u', code, 'msg', {}, None)

    def test_modern_layout_is_tried_first(self):
        src = xp.resolve_source('tag:v1.4.359')
        self.assertTrue(src.urls[0].endswith('/xml/vk.xml'))
        self.assertTrue(src.urls[1].endswith('/src/spec/vk.xml'))

    def test_a_404_falls_back_to_the_older_layout(self):
        src = xp.resolve_source('tag:v1.0.33-core')
        with patch.object(xp, '_download',
                          side_effect=[self._http_error(404), _VALID]) as dl:
            self.assertEqual(xp._download_any(src.urls), _VALID)
        self.assertEqual([c.args[0] for c in dl.call_args_list], list(src.urls))

    def test_a_non_404_does_not_fall_back(self):
        # Offline, 500, timeout: about the request, not the path. Trying
        # a second URL would only produce a more confusing error.
        src = xp.resolve_source('tag:v1.4.359')
        with patch.object(xp, '_download',
                          side_effect=self._http_error(500)) as dl:
            with self.assertRaises(urllib.error.HTTPError):
                xp._download_any(src.urls)
        dl.assert_called_once()

    def test_a_404_on_every_candidate_propagates(self):
        src = xp.resolve_source('tag:nope')
        with patch.object(xp, '_download', side_effect=self._http_error(404)):
            with self.assertRaises(urllib.error.HTTPError):
                xp._download_any(src.urls)


class RefreshTests(_CacheDirCase):

    def test_a_tag_is_a_no_op_once_present(self):
        src = xp.resolve_source('tag:v1.4.359')
        with patch.object(xp, '_download', return_value=_VALID):
            xp.ensure_cached(src)
        with patch.object(xp, '_download',
                          side_effect=AssertionError('tags never move')):
            self.assertEqual(xp.refresh_source(src), 'pinned')

    def test_a_cold_tag_cache_still_fetches(self):
        # "No revalidation" is not "never fetch".
        with patch.object(xp, '_download', return_value=_VALID):
            self.assertEqual(
                xp.refresh_source(xp.resolve_source('tag:v1.4.359')), 'fetched')

    def test_a_branch_reports_unchanged_and_updated(self):
        src = xp.resolve_source('main')
        with patch.object(xp, '_download', return_value=_VALID):
            self.assertEqual(xp.refresh_source(src), 'fetched')
            self.assertEqual(xp.refresh_source(src), 'unchanged')
        with patch.object(xp, '_download', return_value=b'<registry x="1"/>'):
            self.assertEqual(xp.refresh_source(src), 'updated')

    def test_a_non_default_branch_is_also_refreshable(self):
        src = xp.resolve_source('branch:sc_main')
        with patch.object(xp, '_download', return_value=_VALID):
            self.assertEqual(xp.refresh_source(src), 'fetched')
            self.assertEqual(xp.refresh_source(src), 'unchanged')

    def test_local_path_reports_local(self):
        self.assertEqual(xp.refresh_source(xp.resolve_source('/x/vk.xml')),
                         'local')


class LoadXmlTests(_CacheDirCase):

    def test_element_bypasses_everything_and_has_no_provenance(self):
        element = ET.fromstring('<registry/>')
        root, provenance = xp.load_xml(element)
        self.assertIs(root, element)
        self.assertIsNone(provenance)

    def test_provenance_hashes_the_bytes(self):
        with patch.object(xp, '_download', return_value=_VALID):
            _, provenance = xp.load_xml('main')
        self.assertEqual(provenance.sha256, hashlib.sha256(_VALID).hexdigest())
        self.assertEqual(provenance.label, 'main')

    def test_corrupt_cache_entry_refetches_exactly_once(self):
        (self.cache / 'branch-main.xml').write_bytes(b'<truncated')
        with patch.object(xp, '_download', return_value=_VALID) as dl:
            with self.assertLogs('volkano.source', level='WARNING'):
                root, _ = xp.load_xml('main')
        self.assertEqual(root.tag, 'registry')
        dl.assert_called_once()

    def test_a_second_parse_failure_propagates(self):
        (self.cache / 'branch-main.xml').write_bytes(b'<truncated')
        with patch.object(xp, '_download', return_value=b'<still broken'):
            with self.assertLogs('volkano.source', level='WARNING'):
                with self.assertRaises(ET.ParseError):
                    xp.load_xml('main')

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
