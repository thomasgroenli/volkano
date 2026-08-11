"""Tests for the vk.xml -> thunk-graph walker.

Schema-side only: literal and expression folding, the enum dialects
vk.xml has used over the years, and the parse entry point. Acquisition
lives in :mod:`tests.test_xml_source`.
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from volkano import xml_parser as xp
from volkano import xml_source
from volkano.lazy import _Thunk


class _CacheDirCase(unittest.TestCase):
    """Redirects the XML cache into a scratch directory for the test.

    parse_registry composes both halves, so the few tests that exercise
    a tag source still need somewhere for the fetch to land.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = pathlib.Path(tmp.name)
        patcher = patch.object(xml_source, 'cache_dir', lambda: self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)


class ParseIntTests(unittest.TestCase):
    """C integer suffixes are ``[uUlL]`` only — ``f``/``F`` is a hex digit."""

    def test_hex_ending_in_f_keeps_every_digit(self):
        # The regression: rstrip('uUlLfF') turned 0x1F into 0x1, so
        # VK_SHADER_STAGE_ALL_GRAPHICS resolved to 1 instead of 31 and
        # covered only the vertex stage.
        self.assertEqual(xp.parse_int('0x1F'), 31)
        self.assertEqual(xp.parse_int('0xFF'), 255)
        self.assertEqual(xp.parse_int('0x7FFFFFFF'), 0x7FFFFFFF)

    def test_suffixes_are_still_stripped(self):
        for text, want in [('64u', 64), ('12L', 12), ('0xFULL', 15),
                           ('0x7FFFFFFFU', 0x7FFFFFFF),
                           ('0xdeadbeefUL', 0xdeadbeef),
                           ('0xFFFFFFFFFFFFFFFFULL', 2 ** 64 - 1)]:
            with self.subTest(text=text):
                self.assertEqual(xp.parse_int(text), want)

    def test_other_bases_and_signs(self):
        self.assertEqual(xp.parse_int('0b1010'), 10)
        self.assertEqual(xp.parse_int('-1'), -1)
        self.assertEqual(xp.parse_int('  (42)  '), 42)

    def test_garbage_still_raises(self):
        with self.assertRaises(ValueError):
            xp.parse_int('not_a_number')


class FoldIntExprTests(unittest.TestCase):
    """``(~0U-1)`` and friends, folded by hand rather than by eval()."""

    def test_the_sentinel_idioms(self):
        # vk.xml spelled these (~0U-1)/(~0U-2) up to ~v1.2.131 and
        # (~1U)/(~2U) after; both must land on the same number.
        self.assertEqual(xp._fold_int_expr('(~0U-1)'), -2)
        self.assertEqual(xp._fold_int_expr('(~0U-2)'), -3)
        self.assertEqual(xp._fold_int_expr('(~1U)'), -2)
        self.assertEqual(xp._fold_int_expr('(~2U)'), -3)

    def test_signedness_matches_the_uint32_bit_pattern(self):
        for text in ('(~0U-1)', '(~1U)'):
            with self.subTest(text=text):
                self.assertEqual(xp._fold_int_expr(text) & 0xFFFFFFFF,
                                 0xFFFFFFFE)

    def test_plain_and_inverted_terms(self):
        self.assertEqual(xp._fold_int_expr('~0U'), -1)
        self.assertEqual(xp._fold_int_expr('7'), 7)
        self.assertEqual(xp._fold_int_expr('0x10 + 1'), 17)
        self.assertEqual(xp._fold_int_expr('(0x10-0x2)'), 14)

    def test_anything_unmodelled_returns_none_rather_than_guessing(self):
        for text in ('VK_SOMETHING', '0U*2', '', 'a-b', '1 2', '(~0U-)'):
            with self.subTest(text=text):
                self.assertIsNone(xp._fold_int_expr(text))


class LegacyEnumDialectTests(unittest.TestCase):
    """Spellings vk.xml used before the modern attributes existed."""

    def _enum(self, **attrib):
        return ET.Element('enum', attrib)

    def test_expression_values_are_folded(self):
        self.assertEqual(
            xp.enum_value(self._enum(name='VK_QUEUE_FAMILY_EXTERNAL',
                                     value='(~0U-1)')),
            (-2, None))

    def test_a_bare_name_becomes_an_alias(self):
        # Pre-`alias=` style: the target is named in `value`. It may not
        # be parsed yet, so it has to defer rather than resolve here.
        self.assertEqual(
            xp.enum_value(self._enum(name='VK_COLORSPACE_SRGB_NONLINEAR_KHR',
                                     value='VK_COLOR_SPACE_SRGB_NONLINEAR_KHR')),
            (None, 'VK_COLOR_SPACE_SRGB_NONLINEAR_KHR'))

    def test_quoted_strings_still_win_over_the_identifier_rule(self):
        self.assertEqual(
            xp.enum_value(self._enum(name='VK_EXT_NAME', value='"VK_EXT_foo"')),
            ('VK_EXT_foo', None))

    def test_floats_are_untouched(self):
        self.assertEqual(
            xp.enum_value(self._enum(name='VK_LOD_CLAMP_NONE', value='1000.0f')),
            (1000.0, None))

    def test_genuinely_uninterpretable_values_raise(self):
        with self.assertRaisesRegex(ValueError, 'cannot interpret'):
            xp.enum_value(self._enum(name='VK_WEIRD', value='0U*2'))

    def test_an_alias_outside_the_group_is_deferred_not_stringified(self):
        # Reaching a define or another block's constant: emitting the
        # bare name would hand make_enum a str where an int belongs.
        reader = xp.XmlReader(ET.fromstring(
            '<registry><enums name="VkFoo" type="enum">'
            '<enum name="VK_FOO_A" alias="SOMETHING_ELSEWHERE"/>'
            '</enums></registry>'))
        reader.run()
        aliases = reader.output['VkFoo'].args[-1]
        self.assertEqual([entry[0] for entry in aliases], ['VK_FOO_A'])
        target = aliases[0][1]
        self.assertIsInstance(target, _Thunk)          # i.e. Force(...)
        self.assertEqual(target.args, ('SOMETHING_ELSEWHERE',))


class IntegrityCheckTests(_CacheDirCase):
    """A semver tag's patch component is its VK_HEADER_VERSION."""

    XML = (b'<registry><types><type category="define">#define '
           b'<name>VK_HEADER_VERSION</name> {}</type></types></registry>')

    def _serving(self, header_version):
        return patch.object(
            xml_source, '_download',
            return_value=self.XML.replace(b'{}', header_version))

    def test_matching_header_version_passes(self):
        with self._serving(b'359'):
            data, provenance = xp.parse_registry('tag:v1.4.359')
        self.assertEqual(data['VK_HEADER_VERSION'], 359)
        self.assertEqual(provenance.header_version, 359)

    def test_mismatched_header_version_raises(self):
        with self._serving(b'42'):
            with self.assertRaisesRegex(ValueError, 'VK_HEADER_VERSION'):
                xp.parse_registry('tag:v1.4.359')

    def test_a_branch_is_not_checked(self):
        # A branch's version is whatever it says; nothing to compare to.
        with self._serving(b'42'):
            data, _ = xp.parse_registry('main')
        self.assertEqual(data['VK_HEADER_VERSION'], 42)

    def test_non_semver_tags_are_not_checked(self):
        # The '-core' and dated families carry no comparable component.
        for ref in ('v1.0.33-core', 'v1.0-core+wsi-20160216'):
            with self.subTest(ref=ref):
                with self._serving(b'42'):
                    xp.parse_registry('tag:' + ref)

    def test_non_vulkan_api_is_not_checked(self):
        # vulkansc carries an unrelated VK_HEADER_VERSION.
        with self._serving(b'22'):
            xp.parse_registry('tag:v1.4.359', api='vulkansc')

if __name__ == '__main__':
    unittest.main()
