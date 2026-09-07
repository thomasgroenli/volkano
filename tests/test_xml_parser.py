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
from volkano.vulkan_stdlib import STDLIB, VkRegistry
from volkano import cbase


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


class CanonicalTypeNameTests(unittest.TestCase):
    """A type is named once, in cbase's spelling, wherever it appears."""

    def _form(self, xml):
        return xp.type_form(ET.fromstring(xml))

    def test_a_c_spelling_resolves_the_cbase_key(self):
        form = self._form('<member><type>uint32_t</type><name>x</name></member>')
        self.assertEqual(form.args, ('uint32',))

    def test_a_vulkan_type_passes_through_untouched(self):
        form = self._form('<member><type>VkInstance</type><name>x</name></member>')
        self.assertEqual(form.args, ('VkInstance',))

    def test_void_pointer_takes_the_type_erased_slot(self):
        form = self._form('<member><type>void</type>* <name>p</name></member>')
        self.assertEqual(form.args, (xp.VOID_P,))

    def test_a_missing_type_element_takes_the_same_slot(self):
        self.assertEqual(self._form('<member><name>p</name></member>').args,
                         (xp.VOID_P,))

    def test_bare_void_takes_the_absence_of_a_type_slot(self):
        self.assertEqual(self._form('<proto><type>void</type><name>f</name></proto>').args,
                         (xp.VOID,))

    def test_a_pointer_to_a_c_spelling_wraps_the_canonical_key(self):
        form = self._form('<member><type>float</type>* <name>p</name></member>')
        # Call('ref', [Force('float32')])
        self.assertEqual(form.args[0].args, ('float32',))

    def test_translation_reaches_the_registry(self):
        reader = xp.XmlReader(ET.fromstring(
            '<registry><types>'
            '<type category="struct" name="S">'
            '<member><type>uint32_t</type><name>a</name></member>'
            '</type></types></registry>'))
        r = VkRegistry({**STDLIB, **reader.run()})
        self.assertIs(dict(r('S')._fields_)['a'], cbase.uint32)


class EnumerantEntryTests(unittest.TestCase):
    """Enumerants resolve through their group, not to a bare int."""

    XML = ('<registry><enums name="VkFoo" type="enum">'
           '<enum name="VK_FOO_A" value="0"/>'
           '<enum name="VK_FOO_B" value="-3"/>'
           '<enum name="VK_FOO_B_KHR" alias="VK_FOO_B"/>'
           '</enums></registry>')

    def _data(self):
        return xp.XmlReader(ET.fromstring(self.XML)).run()

    def test_enumerant_emits_a_group_referencing_call(self):
        entry = self._data()['VK_FOO_B']
        self.assertIsInstance(entry, _Thunk)
        # (Force('enum_member'), Force('VkFoo'), name, value)
        self.assertEqual(entry.fn.args, ('enum_member',))
        self.assertEqual(entry.args[0].args, ('VkFoo',))
        self.assertEqual(entry.args[1:], ('VK_FOO_B', -3))

    def test_computed_value_rides_along_as_the_fallback(self):
        self.assertEqual(self._data()['VK_FOO_B_KHR'].args[-1], -3)

    def test_forcing_yields_the_member_itself(self):
        r = VkRegistry({**STDLIB, **self._data()})
        self.assertIs(r('VK_FOO_A'), r('VkFoo').VK_FOO_A)

    def test_alias_forces_to_the_canonical_member(self):
        r = VkRegistry({**STDLIB, **self._data()})
        self.assertIs(r('VK_FOO_B_KHR'), r('VkFoo').VK_FOO_B)

    def test_no_cycle_between_group_and_its_enumerants(self):
        # The group and its members reference each other's entries; a
        # cycle here would surface as the kernel's ValueError.
        r = VkRegistry({**STDLIB, **self._data()})
        self.assertEqual(int(r('VK_FOO_B')), -3)
        self.assertEqual(len(r('VkFoo').__members__), 3)


class ProvenanceTests(_CacheDirCase):
    """The parser records VK_HEADER_VERSION; it no longer adjudicates it."""

    URI = ('https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'
           '/refs/tags/v1.4.359/xml/vk.xml')

    XML = (b'<registry><types><type category="define">#define '
           b'<name>VK_HEADER_VERSION</name> {}</type></types></registry>')

    def _serving(self, header_version):
        return patch.object(
            xml_source, '_download',
            return_value=self.XML.replace(b'{}', header_version))

    def test_header_version_is_recorded(self):
        with self._serving(b'359'):
            data, provenance = xp.parse_registry(self.URI)
        self.assertEqual(data['VK_HEADER_VERSION'], 359)
        self.assertEqual(provenance.header_version, 359)

    def test_provenance_carries_the_uri_it_was_fetched_from(self):
        with self._serving(b'359'):
            _, provenance = xp.parse_registry(self.URI)
        self.assertEqual(provenance.uri, self.URI)

    def test_the_uri_is_never_second_guessed(self):
        # There was once a cross-check here: a semver-shaped tag name
        # implied a VK_HEADER_VERSION, and disagreeing bytes raised. A
        # URI-keyed cache has no name/content indirection left to guard
        # — the entry is derived from the URI it came from — so the
        # bytes are simply believed.
        with self._serving(b'42'):
            data, provenance = xp.parse_registry(self.URI)
        self.assertEqual(data['VK_HEADER_VERSION'], 42)
        self.assertEqual(provenance.header_version, 42)

    def test_every_api_is_treated_alike(self):
        # vulkansc carries an unrelated VK_HEADER_VERSION. With nothing
        # to compare it against, it needs no special case.
        with self._serving(b'22'):
            _, provenance = xp.parse_registry(self.URI, api='vulkansc')
        self.assertEqual(provenance.header_version, 22)

    def test_an_element_still_has_no_provenance(self):
        import xml.etree.ElementTree as ET
        data, provenance = xp.parse_registry(
            ET.fromstring(self.XML.replace(b'{}', b'359').decode()))
        self.assertEqual(data['VK_HEADER_VERSION'], 359)
        self.assertIsNone(provenance)


if __name__ == '__main__':
    unittest.main()
