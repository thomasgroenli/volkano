"""vk.xml → thunk-graph converter for :mod:`volkano.lazy`.

Same XML walker as :mod:`lazyregistry.xml_parser` but emits
:class:`_Thunk` / :class:`_Raw` objects directly instead of the DSL's
form-shaped lists. The two parsers are line-for-line near-identical
apart from the form builders — :func:`Force`, :func:`Call`, :func:`Quote`
are re-exported from :mod:`volkano.lazy` and produce thunk
objects rather than marker-prefixed lists.

This module is the *schema* half. Everything about where the bytes come
from — URIs, downloads, the on-disk cache, provenance — lives in
:mod:`volkano.xml_source`, which knows how to fetch bytes and nothing
about Vulkan's schema. The split keeps this file testable
against an in-memory Element with no notion of caching, and keeps that
one testable with no notion of what a ``<type>`` means.

The dict this module returns is *not* JSON-serialisable: its values are
Python objects with embedded callables and sentinels. That's the
trade-off for losing the DSL's bookkeeping overhead — the thunk graph
goes straight from XML parse into the runtime registry with no
intermediate serialisation step.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

from .lazy import Force, Call, Quote
from .xml_source import Provenance, load_xml


# Parser-side logger. One DEBUG record per resolved fragment; the
# acquisition side logs separately under ``volkano.source``.
logger = logging.getLogger('volkano.parser')


# ---------------------------------------------------------------------------
# Type-fragment → thunk graph
# ---------------------------------------------------------------------------

#: Registry keys for the two slots with no name a user would ever type:
#: C ``void`` (the absence of a type) and volkano's own ``void *`` slot,
#: which is not a C spelling at all. Underscore-prefixed so the thunk
#: graph can still resolve them while attribute access, ``dir()`` and
#: the stub leave them alone.
VOID = '_void'
VOID_P = '_void_p'


def _bracket_array_length(name_tail: str) -> str | None:
    """Extract ``[N]`` from the text trailing a ``<name>`` element."""
    m = re.search(r'\[([^\]]+)\]', name_tail or '')
    return m.group(1).strip() if m else None


#: The C spellings vk.xml writes, mapped to the name volkano publishes
#: each type under. Applied wherever a type is *named* — which is
#: :func:`type_form` and the ``alias=`` passthrough, and nowhere else —
#: so a member declared ``uint32_t`` resolves the very entry a reader
#: reaches for as ``vk.uint32``. One type, one name, rather than a
#: registry key per spelling the XML happens to use.
#:
#: Several spellings share a target (``bool`` and ``_Bool``; ``intptr_t``
#: and ``ptrdiff_t``), which is the point: the collapsing happens here,
#: where it is a table anyone can read, instead of surfacing as five
#: registry entries for one class.
#:
#: Strings on both sides, deliberately. The parser must not import
#: :mod:`volkano.vulkan_stdlib` — that module imports *this* one — and a
#: spelling map is the parser's business anyway.
CANONICAL_TYPE_NAMES: dict[str, str] = {
    'char':      'char',            # identity, and kept so on purpose
    'int':       'int32',
    'float':     'float32',
    'double':    'float64',
    'size_t':    'size',
    'int8_t':    'int8',
    'int16_t':   'int16',
    'int32_t':   'int32',
    'int64_t':   'int64',
    'uint8_t':   'uint8',
    'uint16_t':  'uint16',
    'uint32_t':  'uint32',
    'uint64_t':  'uint64',
    'bool':      'uint8',
    '_Bool':     'uint8',
    'intptr_t':  'int64',           # 64-bit on the targets we support
    'uintptr_t': 'uint64',
    'ptrdiff_t': 'int64',
    'void':      VOID,
}


def canonical(type_name: str) -> str:
    """The registry key a C type name resolves through."""
    return CANONICAL_TYPE_NAMES.get(type_name, type_name)


def type_form(elem: ET.Element) -> Any:
    """Build a thunk for the C type a ``<member>`` / ``<param>`` declares.

    Same logic as the DSL parser: base type name from ``<type>``, pointer
    levels from the ``*``-bearing text fragments, array dimension from
    a nested ``<enum>NAME</enum>`` or a literal ``[N]`` in the
    ``<name>``'s tail. ``void *`` short-circuits to the type-erased
    ``void_p`` slot (``Pointer[None]``); ``char *`` resolves through the
    normal pointer path to ``Pointer[char]`` (its data supplied as
    ``bytes``).
    """
    type_elem = elem.find('type')
    name_elem = elem.find('name')
    if type_elem is None or type_elem.text is None:
        return Force(VOID_P)

    type_name = type_elem.text.strip()
    pre_text = (elem.text or '')
    tail_text = (type_elem.tail or '')
    ptr_count = pre_text.count('*') + tail_text.count('*')

    name_tail = (name_elem.tail or '') if name_elem is not None else ''
    array_len: str | int | None = None
    enum_child = elem.find('enum')
    if enum_child is not None and enum_child.text:
        array_len = enum_child.text.strip()
    elif name_tail:
        bracket = _bracket_array_length(name_tail)
        if bracket is not None:
            try:
                array_len = int(bracket, 0)
            except ValueError:
                array_len = bracket

    if type_name == 'void' and ptr_count >= 1:
        form = Force(VOID_P)
        ptr_count -= 1
    else:
        form = Force(canonical(type_name))

    for _ in range(ptr_count):
        form = Call('ref', [form])

    if array_len is not None:
        if isinstance(array_len, int):
            length: Any = array_len
        else:
            length = Force(array_len)
        form = Call('array', [form, length])

    return form


def field_entry(elem: ET.Element) -> list:
    """Format a single ``<member>`` / ``<param>`` as ``[name, type_form[, bits]]``."""
    name_elem = elem.find('name')
    fname = name_elem.text if name_elem is not None and name_elem.text else ''
    tform = type_form(elem)
    name_tail = (name_elem.tail or '') if name_elem is not None else ''
    m = re.search(r':\s*(\d+)', name_tail)
    if m:
        return [fname, tform, int(m.group(1))]
    return [fname, tform]


# ---------------------------------------------------------------------------
# Enum value computation
# ---------------------------------------------------------------------------

EXT_BASE = 1_000_000_000
EXT_BLOCK = 1_000


#: A C integer literal plus its suffix. The suffix set is **only**
#: ``[uUlL]`` — ``f``/``F`` is a *float* suffix in C and, crucially, a
#: hex digit. Stripping it blindly turns ``0x1F`` into ``0x1`` and
#: ``0x7FFFFFFF`` into ``0x7``: silently wrong values, not errors.
_C_INT_RE = re.compile(
    r'''^(?P<body> [+-]?0[xX][0-9a-fA-F]+     # hex
                 | [+-]?0[bB][01]+            # binary
                 | [+-]?\d+                   # decimal / octal
        )
        (?P<suffix>[uUlL]*)$''', re.VERBOSE)


def parse_int(s: str) -> int:
    """``int(s, 0)`` tolerating C integer suffixes (``U`` / ``UL`` / ``ULL``)."""
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()
    match = _C_INT_RE.match(s)
    if match:
        return int(match.group('body'), 0)
    # Not a well-formed C integer literal. Fall back to the permissive
    # reading so genuinely odd input still gets a shot, and let int()
    # produce the error message if it can't.
    return int(s.rstrip('uUlLfF'), 0)


#: One term of a small C integer expression: an optional leading
#: ``+``/``-``, an optional bitwise ``~``, then a literal with its
#: suffix. Chained with :func:`_fold_int_expr` to cover the ``(~0U-1)``
#: idiom that vk.xml used for sentinel constants up to ~v1.2.131.
_INT_TERM_RE = re.compile(
    r'\s*(?P<op>[-+])?\s*(?P<invert>~)?\s*'
    r'(?P<num>0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)[uUlL]*\s*')

_IDENTIFIER_RE = re.compile(r'^[A-Za-z_]\w*$')


def _fold_int_expr(text: str) -> int | None:
    """Fold a small C integer expression such as ``(~0U-1)``.

    Deliberately hand-rolled rather than handed to :func:`eval`: the
    input is third-party XML, and the grammar we actually need is a
    chain of ``+``/``-`` terms with an optional bitwise ``~``. Anything
    outside that shape returns ``None`` so the caller can fall through
    instead of guessing.

    Note the result is Python-signed — ``~0U`` is ``-1``, not
    ``0xFFFFFFFF``, matching how this parser has always read ``~0U``.
    ctypes narrows it correctly on the way into a ``uint32`` field.
    """
    s = text.strip()
    while s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()
    total: int | None = None
    pos = 0
    for match in _INT_TERM_RE.finditer(s):
        if match.start() != pos:
            return None                      # a gap we don't model
        pos = match.end()
        value = int(match.group('num'), 0)
        if match.group('invert'):
            value = ~value
        op = match.group('op')
        if total is None:
            total = -value if op == '-' else value
        elif op == '-':
            total -= value
        elif op == '+':
            total += value
        else:
            return None                      # two terms, no operator
    if total is None or pos != len(s):
        return None
    return total


def enum_value(elem: ET.Element,
               extnumber: int | None = None) -> tuple[Any, str | None]:
    """Compute the numeric value (or alias) for one ``<enum>`` element."""
    if 'alias' in elem.attrib:
        return (None, elem.get('alias'))
    if 'value' in elem.attrib:
        raw = elem.get('value', '').strip()
        if elem.get('type') == 'float' or re.match(r'-?\d+\.\d', raw.rstrip('fF')):
            return (float(raw.rstrip('fF')), None)
        try:
            return (parse_int(raw), None)
        except ValueError:
            pass
        if raw.startswith('"') and raw.endswith('"'):
            return (raw[1:-1], None)
        folded = _fold_int_expr(raw)
        if folded is not None:
            return (folded, None)
        stripped = raw.strip('() ')
        if _IDENTIFIER_RE.match(stripped):
            # A bare name in `value`. Older vk.xml expressed an alias
            # this way, before the `alias` attribute existed; video.xml
            # points at a top-level define the same way. Both are
            # deferred references, so hand it to the alias machinery
            # rather than trying to resolve it during this phase — the
            # target may not have been parsed yet.
            return (None, stripped)
        raise ValueError(
            f"cannot interpret enum value {raw!r} for "
            f"{elem.get('name', '<unnamed>')!r}")
    if 'bitpos' in elem.attrib:
        return (1 << int(elem.get('bitpos', '0'), 0), None)
    if 'offset' in elem.attrib:
        offset = int(elem.get('offset', '0'), 0)
        explicit_ext = elem.get('extnumber')
        if explicit_ext is not None:
            ext_num = int(explicit_ext, 0)
        else:
            ext_num = extnumber if extnumber is not None else 0
        sign = -1 if elem.get('dir') == '-' else 1
        return (sign * (EXT_BASE + (ext_num - 1) * EXT_BLOCK + offset), None)
    return (None, None)


# ---------------------------------------------------------------------------
# Registry walk
# ---------------------------------------------------------------------------

class XmlReader:
    """Reads vk.xml and accumulates the thunk-graph data dict.

    Run order matches the C-header dependency layout so forward
    references are minimised: ``<enums>`` blocks first, then extensions
    (which inject enumerants), then ``<types>``, then ``<commands>``,
    then free-constant emission.
    """

    def __init__(self, root: ET.Element, api: str = 'vulkan'):
        self.root = root
        self.api = api
        self.enum_values: dict[str, tuple[Any, str | None]] = {}
        self.group_members: dict[str, list[tuple[str, Any, str | None]]] = {}
        self.free_constants: dict[str, Any] = {}
        self.output: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Execute every phase in order; return the assembled output dict."""
        self.read_enums_blocks()
        self.read_extensions()
        self.read_features()
        self.read_types()
        self.read_commands()
        self.emit_free_constants()
        return self.output

    # ------------------------------------------------------------------
    # Phase 1: <enums> blocks
    # ------------------------------------------------------------------

    def read_enums_blocks(self) -> None:
        for block in self.root.findall('enums'):
            group = block.get('name')
            kind = block.get('type')
            if group == 'API Constants' or kind is None:
                for elem in block.findall('enum'):
                    if not self._api_matches(elem):
                        continue
                    self._record_free_constant(elem)
            else:
                members = self.group_members.setdefault(group, [])
                for elem in block.findall('enum'):
                    if not self._api_matches(elem):
                        continue
                    name = elem.get('name')
                    if name is None:
                        continue
                    val, alias = enum_value(elem)
                    self.enum_values[name] = (val, alias)
                    members.append((name, val, alias))

    def _record_free_constant(self, elem: ET.Element) -> None:
        name = elem.get('name')
        if name is None:
            return
        val, alias = enum_value(elem)
        self.enum_values[name] = (val, alias)
        if alias is None and val is not None:
            self.free_constants[name] = val
        elif alias is not None:
            self.free_constants[name] = ('alias', alias)

    # ------------------------------------------------------------------
    # Phase 2: <extensions> — inject extension-added enums into groups
    # ------------------------------------------------------------------

    def read_extensions(self) -> None:
        for ext in self.root.findall('extensions/extension'):
            if not self._api_matches(ext):
                continue
            ext_num = ext.get('number')
            ext_num_int = int(ext_num, 0) if ext_num else None
            supported = (ext.get('supported') or '').split(',')
            if self.api not in supported and supported != ['']:
                continue
            for req in ext.findall('require'):
                for elem in req.findall('enum'):
                    if not self._api_matches(elem):
                        continue
                    self._absorb_extension_enum(elem, ext_num_int)

    # ------------------------------------------------------------------
    # Phase 2b: <feature> — absorb enums declared in core-version promotions
    # ------------------------------------------------------------------

    def read_features(self) -> None:
        for feat in self.root.findall('feature'):
            if not self._api_matches(feat):
                continue
            for req in feat.findall('require'):
                for elem in req.findall('enum'):
                    if not self._api_matches(elem):
                        continue
                    self._absorb_extension_enum(elem, None)

    def _absorb_extension_enum(self, elem: ET.Element,
                               ext_num: int | None) -> None:
        name = elem.get('name')
        if name is None:
            return
        extends = elem.get('extends')
        val, alias = enum_value(elem, extnumber=ext_num)
        if extends:
            members = self.group_members.setdefault(extends, [])
            if any(m[0] == name for m in members):
                return
            self.enum_values[name] = (val, alias)
            members.append((name, val, alias))
        else:
            self.enum_values[name] = (val, alias)
            if alias is not None:
                self.free_constants.setdefault(name, ('alias', alias))
            elif val is not None:
                self.free_constants.setdefault(name, val)

    def _resolve_chain(self, name: str, seen: set[str] | None = None) -> Any:
        """Walk alias→alias chains until we hit a real value (or give up)."""
        seen = seen if seen is not None else set()
        if name in seen:
            return None
        seen.add(name)
        if name not in self.enum_values:
            return None
        val, alias = self.enum_values[name]
        if val is not None:
            return val
        if alias is not None:
            return self._resolve_chain(alias, seen)
        return None

    # ------------------------------------------------------------------
    # Phase 3: <types>
    # ------------------------------------------------------------------

    def read_types(self) -> None:
        for elem in self.root.findall('types/type'):
            if not self._api_matches(elem):
                continue
            category = elem.get('category')
            name = self._type_name(elem)
            if not name:
                continue
            alias = elem.get('alias')
            if alias is not None:
                self.output[name] = Force(canonical(alias))
                continue

            if category == 'basetype':
                self.output[name] = self._emit_basetype(elem, name)
            elif category == 'handle':
                self.output[name] = self._emit_handle(elem, name)
            elif category == 'bitmask':
                self.output[name] = self._emit_bitmask_typedef(elem, name)
            elif category == 'enum':
                self.output[name] = self._emit_enum_group(name)
            elif category == 'bitmask' or name in self.group_members:
                self.output[name] = self._emit_enum_group(name)
            elif category == 'funcpointer':
                emitted = self._emit_funcpointer(elem, name)
                if emitted is not None:
                    self.output[name] = emitted
            elif category in ('struct', 'union'):
                self.output[name] = self._emit_struct_like(elem, name, category)
            elif category == 'define':
                literal = self._extract_define_literal(elem, name)
                if literal is not None:
                    self.output[name] = literal

        for group_name in self.group_members:
            if group_name not in self.output:
                self.output[group_name] = self._emit_enum_group(group_name)

    def _emit_basetype(self, elem: ET.Element, name: str) -> Any:
        type_child = elem.find('type')
        underlying = type_child.text if type_child is not None and type_child.text else None
        return Call('make_basetype', [name, underlying], with_registry=True)

    def _emit_handle(self, elem: ET.Element, name: str) -> Any:
        type_child = elem.find('type')
        macro = type_child.text if type_child is not None else ''
        dispatchable = (macro == 'VK_DEFINE_HANDLE')
        parent = elem.get('parent')
        objtypeenum = elem.get('objtypeenum')
        return Call('make_handle',
                    [name, dispatchable, parent, objtypeenum],
                    with_registry=True)

    def _emit_bitmask_typedef(self, elem: ET.Element, name: str) -> Any:
        type_child = elem.find('type')
        underlying = type_child.text if type_child is not None and type_child.text else 'VkFlags'
        bitwidth = 64 if underlying == 'VkFlags64' else 32
        requires = elem.get('requires') or elem.get('bitvalues')
        return Call('make_bitmask',
                    [name, bitwidth, requires], with_registry=True)

    def _emit_enum_group(self, name: str) -> Any:
        block = self.root.find(f"enums[@name='{name}']")
        kind = 'enum'
        bitwidth = 32
        if block is not None:
            kind = block.get('type') or 'enum'
            bw = block.get('bitwidth')
            if bw:
                bitwidth = int(bw, 0)
            elif kind == 'bitmask':
                bitwidth = 32
        members = self.group_members.get(name, [])

        values: list[list[Any]] = []
        aliases: list[list[Any]] = []
        for mname, mval, malias in members:
            if mval is not None:
                values.append([mname, mval])
            elif malias is not None:
                resolved = self._resolve_chain(malias)
                if resolved is not None:
                    aliases.append([mname, resolved])
                else:
                    # Not another member of this group — so it names
                    # something at registry level (a define, a constant
                    # from another block). Defer it rather than emitting
                    # the bare string, which would reach make_enum as a
                    # value of the wrong type.
                    aliases.append([mname, Force(malias)])

        return Call('make_enum',
                    [name, kind, bitwidth, values, aliases],
                    with_registry=True)

    def _emit_funcpointer(self, elem: ET.Element, name: str) -> Any:
        proto = elem.find('proto')
        if proto is None:
            return None
        rettype = type_form(proto)
        params = [field_entry(p) for p in elem.findall('param')]
        return Call('make_funcpointer',
                    [name, rettype, params], with_registry=True)

    def _extract_define_literal(self, elem: ET.Element,
                                name: str) -> int | float | None:
        """Best-effort literal extraction from a ``<type category="define">``."""
        name_elem = elem.find('name')
        if name_elem is not None and name_elem.tail:
            if re.match(r'\s*\(', name_elem.tail):
                return None

        text = ''.join(elem.itertext())
        text_clean = re.sub(r'//[^\n]*', '', text)
        m = re.search(rf'#define\s+{re.escape(name)}\s+([^\n]+)', text_clean)
        if not m:
            return None
        body = m.group(1).strip()
        try:
            return parse_int(body)
        except ValueError:
            pass
        m_f = re.match(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[fF]?$', body)
        if m_f:
            try:
                return float(body.rstrip('fF'))
            except ValueError:
                pass
        call = re.match(r'(VK_\w+)\s*\(([^)]*)\)\s*$', body)
        if call:
            return self._eval_macro_call(call.group(1), call.group(2))
        return None

    def _eval_macro_call(self, macro: str, args_raw: str) -> int | None:
        """Replay a handful of Vulkan version macros at parse time."""
        try:
            args: list[int] = []
            for piece in args_raw.split(','):
                piece = piece.strip()
                try:
                    args.append(parse_int(piece))
                    continue
                except ValueError:
                    pass
                if piece in self.output and isinstance(self.output[piece], int):
                    args.append(self.output[piece])
                else:
                    return None
        except Exception:
            return None

        if macro == 'VK_MAKE_API_VERSION' and len(args) == 4:
            variant, major, minor, patch = args
            return (variant << 29) | (major << 22) | (minor << 12) | patch
        if macro == 'VK_MAKE_VERSION' and len(args) == 3:
            major, minor, patch = args
            return (major << 22) | (minor << 12) | patch
        return None

    def _emit_struct_like(self, elem: ET.Element, name: str,
                          category: str) -> Any:
        members_xml = elem.findall('member')
        fields = [field_entry(m) for m in members_xml
                  if self._api_matches(m)]
        factory = 'make_struct' if category == 'struct' else 'make_union'
        # Quote the fields list so the kernel doesn't descend before the
        # factory runs. The factory plants the in-progress class in
        # ``registry._cache`` first, then descends the fields itself —
        # the standard self-cycle-safe pattern (VkBaseInStructure.pNext).
        return Call(factory, [name, Quote(fields)], with_registry=True)

    # ------------------------------------------------------------------
    # Phase 4: <commands>
    # ------------------------------------------------------------------

    def read_commands(self) -> None:
        for cmd in self.root.findall('commands/command'):
            if not self._api_matches(cmd):
                continue
            alias = cmd.get('alias')
            if alias is not None:
                cmd_name = cmd.get('name')
                if cmd_name:
                    self.output[cmd_name] = Force(alias)
                continue
            proto = cmd.find('proto')
            if proto is None:
                continue
            name_elem = proto.find('name')
            if name_elem is None or not name_elem.text:
                continue
            cmd_name = name_elem.text.strip()
            rettype = type_form(proto)
            params = [field_entry(p) for p in cmd.findall('param')
                      if self._api_matches(p)]
            attrs = {k: v for k, v in cmd.attrib.items() if k != 'name'}
            self.output[cmd_name] = Call(
                'make_command',
                [cmd_name, rettype, params, attrs],
                with_registry=True)

    # ------------------------------------------------------------------
    # Phase 5: emit free constants (with alias chain resolution)
    # ------------------------------------------------------------------

    def emit_free_constants(self) -> None:
        unresolved: list[str] = []
        for name, payload in self.free_constants.items():
            if name in self.output:
                continue
            if isinstance(payload, tuple) and payload[0] == 'alias':
                resolved = self._resolve_chain(payload[1])
                if resolved is None:
                    unresolved.append(name)
                    continue
                self.output[name] = resolved
            else:
                self.output[name] = payload
        # Enumerants resolve to their group's member rather than to the
        # bare integer, so ``vk.VK_SUCCESS is vk.VkResult.VK_SUCCESS``
        # holds — see :func:`vulkan_stdlib.enum_member`. The computed
        # integer rides along as the fallback, which also keeps the
        # entry independently meaningful if the group can't be built.
        #
        # No cycle is introduced by the group reference. An enumerant
        # entry forces its group, and the only thing a group forces back
        # is the ``Force(malias)`` fallback in :meth:`_emit_enum_group` —
        # reached solely when ``_resolve_chain`` found the alias target
        # is *not* an enumerant anywhere, so it can never point into
        # this loop's output.
        for group, members in self.group_members.items():
            for mname, mval, malias in members:
                if mname in self.output:
                    continue
                if mval is not None:
                    value = mval
                elif malias is not None:
                    value = self._resolve_chain(malias)
                    if value is None:
                        unresolved.append(mname)
                        continue
                else:
                    continue
                self.output[mname] = Call('enum_member',
                                          [Force(group), mname, value])
        if unresolved:
            self.output['_unresolved_aliases'] = unresolved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _type_name(self, elem: ET.Element) -> str:
        name = elem.get('name')
        if name:
            return name
        name_elem = elem.find('name')
        if name_elem is not None and name_elem.text:
            return name_elem.text.strip()
        proto_name = elem.find('proto/name')
        if proto_name is not None and proto_name.text:
            return proto_name.text.strip()
        return ''

    def _api_matches(self, elem: ET.Element) -> bool:
        api = elem.get('api')
        if not api:
            return True
        return self.api in api.split(',')


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def parse_registry(source: Any = None, *,
                   api: str = 'vulkan') -> tuple[dict[str, Any], Provenance | None]:
    """Convert a Vulkan XML registry to thunk-graph data, with provenance.

    ``source`` is a URI or path as understood by
    :func:`volkano.xml_source.resolve_source`, plus an
    :class:`~xml.etree.ElementTree.Element` for in-memory use.
    """
    root, provenance = load_xml(source)
    data = XmlReader(root, api=api).run()
    if provenance is None:
        return data, None
    # Recorded, not checked. There was once a cross-check here against
    # the version a semver-shaped tag name implied, because a *name*
    # could lie about the bytes filed under it. A URI-keyed cache has no
    # such indirection to guard: the cache entry is derived from the URI
    # it was fetched from, so there is no second claim to disagree with.
    return data, provenance._replace(
        header_version=data.get('VK_HEADER_VERSION'))


def parse_xml(source: Any = None, *, api: str = 'vulkan') -> dict[str, Any]:
    """Convert a Vulkan XML registry to thunk-graph data.

    Returns a dict whose values are :class:`_Thunk` / :class:`_Raw`
    objects (or plain literals for constants). Hand it to
    :func:`volkano.vulkan_stdlib.build_registry` to wrap it in a
    :class:`Lazy` and merge in the stdlib factories.
    """
    return parse_registry(source, api=api)[0]
