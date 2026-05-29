"""vk.xml → thunk-graph converter for :mod:`volkano.lazy`.

Same XML walker as :mod:`lazyregistry.xml_parser` but emits
:class:`_Thunk` / :class:`_Raw` objects directly instead of the DSL's
form-shaped lists. The two parsers are line-for-line near-identical
apart from the form builders — :func:`Force`, :func:`Call`, :func:`Quote`
are re-exported from :mod:`volkano.lazy` and produce thunk
objects rather than marker-prefixed lists.

The dict this module returns is *not* JSON-serialisable: its values are
Python objects with embedded callables and sentinels. That's the
trade-off for losing the DSL's bookkeeping overhead — the thunk graph
goes straight from XML parse into the runtime registry with no
intermediate serialisation step.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from .lazy import Force, Call, Quote


# Parser-side logger. Network fetches surface at INFO (so a single
# ``logging.basicConfig(level=logging.INFO)`` shows them); cached-file
# usage drops to DEBUG so the steady-state case stays quiet.
logger = logging.getLogger('volkano.parser')


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------

#: Canonical location of the Vulkan-Docs registry. Pinned to the ``main``
#: branch so :func:`parse_xml` always picks up the latest published API
#: surface; pass ``source=`` explicitly to pin to a tag or commit.
DEFAULT_VK_XML_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/"
    "refs/heads/main/xml/vk.xml"
)


def _user_cache_dir() -> pathlib.Path:
    """Return a writable per-user cache directory for downloaded XML.

    Honours XDG_CACHE_HOME on POSIX and LOCALAPPDATA on Windows. Shares
    the cache filename with :mod:`lazyregistry.xml_parser` since both
    packages consume the same XML — no point downloading twice.
    """
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~/AppData/Local')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    path = pathlib.Path(base) / 'lazyregistry'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path_for_url(url: str) -> pathlib.Path:
    if url == DEFAULT_VK_XML_URL:
        return _user_cache_dir() / 'vk.xml'
    digest = hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]
    return _user_cache_dir() / f'vk-{digest}.xml'


def fetch_xml(url: str = DEFAULT_VK_XML_URL, *, refresh: bool = False,
              cache_path: pathlib.Path | str | None = None) -> pathlib.Path:
    """Download ``url`` to the local cache (unless already present).

    Returns the path to the cached file. Set ``refresh=True`` to force a
    redownload; pass ``cache_path`` to override the default location.
    """
    target = pathlib.Path(cache_path) if cache_path is not None else _cache_path_for_url(url)
    if target.exists() and not refresh:
        logger.debug('using cached %s', target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url, headers={'User-Agent': 'volkano/0.1 (+https://github.com/KhronosGroup/Vulkan-Docs)'})
    logger.info('fetching %s -> %s', url, target)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    target.write_bytes(data)
    logger.info('cached %d bytes', len(data))
    return target


# ---------------------------------------------------------------------------
# Type-fragment → thunk graph
# ---------------------------------------------------------------------------

def _bracket_array_length(name_tail: str) -> str | None:
    """Extract ``[N]`` from the text trailing a ``<name>`` element."""
    m = re.search(r'\[([^\]]+)\]', name_tail or '')
    return m.group(1).strip() if m else None


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
        return Force('void_p')

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
        form = Force('void_p')
        ptr_count -= 1
    else:
        form = Force(type_name)

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


def parse_int(s: str) -> int:
    """``int(s, 0)`` tolerating C suffixes ``U`` / ``ULL`` / ``L`` / ``F``."""
    s = s.strip().rstrip('uUlLfF')
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    return int(s, 0)


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
            stripped = raw.strip('() ')
            if stripped.startswith('~'):
                return (~parse_int(stripped[1:]), None)
            if raw.startswith('"') and raw.endswith('"'):
                return (raw[1:-1], None)
            raise
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
                self.output[name] = Force(alias)
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
                    aliases.append([mname, malias])

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
        for members in self.group_members.values():
            for mname, mval, malias in members:
                if mname in self.output:
                    continue
                if mval is not None:
                    self.output[mname] = mval
                elif malias is not None:
                    resolved = self._resolve_chain(malias)
                    if resolved is not None:
                        self.output[mname] = resolved
                    else:
                        unresolved.append(mname)
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

def parse_xml(source: Any = None, *,
              api: str = 'vulkan',
              refresh: bool = False,
              cache_path: pathlib.Path | str | None = None) -> dict[str, Any]:
    """Convert a Vulkan XML registry to thunk-graph data.

    ``source`` accepts the same shapes as :func:`lazyregistry.xml_parser.parse_xml`:
    ``None`` (default — fetch from GitHub), a URL string, a filesystem
    path, or an :class:`xml.etree.ElementTree.Element`.

    Returns a dict whose values are :class:`_Thunk` / :class:`_Raw`
    objects (or plain literals for constants). Hand it to
    :func:`volkano.vulkan_stdlib.build_registry` to wrap it in a
    :class:`Lazy` and merge in the stdlib factories.
    """
    if isinstance(source, ET.Element):
        root = source
    else:
        path: pathlib.Path
        if source is None:
            path = fetch_xml(DEFAULT_VK_XML_URL, refresh=refresh,
                             cache_path=cache_path)
        elif isinstance(source, str) and (source.startswith('http://')
                                          or source.startswith('https://')):
            path = fetch_xml(source, refresh=refresh, cache_path=cache_path)
        else:
            path = pathlib.Path(source)
        root = ET.parse(str(path)).getroot()
    reader = XmlReader(root, api=api)
    return reader.run()
