"""Generate a ``.pyi`` stub file for the :mod:`volkano` package.

The runtime package resolves all 8 000+ Vulkan symbols dynamically via
PEP 562 :func:`__getattr__`. That works at runtime but is invisible to
static analysers — IDEs can't autocomplete ``volkano.vk…`` because they
never see those names declared. This module walks the live registry
(forcing every resolvable entry) and emits a stub file that declares
each name with a permissive type, so type checkers and language
servers gain full coverage of the Vulkan API surface.

Use as a CLI::

    python -m volkano.stub                  # writes volkano/__init__.pyi
    python -m volkano.stub -o other.pyi     # custom path
    python -m volkano.stub --refresh        # redownload main.xml first

Or programmatically::

    from volkano.stub import write_stub
    write_stub()

The stub carries a provenance stamp on its first line naming the
sha256 of the XML it was generated from. :func:`sync_stub` compares
that against the registry it is handed and rewrites only on a
mismatch, so the file tracks whichever ``vk.xml`` the package facade
is actually built from. Stamping content rather than keying off "did
we just download something" is what makes it self-healing: a deleted
stub, a pre-populated cache and an interrupted write all repair
themselves on the next import.

Only the package facade (:func:`volkano.get_registry`) syncs the stub.
A registry built through :func:`volkano.registry` never touches it.

The stub is purely a hint for tooling — the package itself works
without it.
"""

from __future__ import annotations

import argparse
import ctypes
import enum
import itertools
import logging
import os
import re
import sys
import tempfile
from typing import Any, IO

from . import cbase


# Stub-side logger. Regeneration is an INFO lifecycle event; failures
# are WARNINGs rather than exceptions, since a stale stub costs
# autocomplete and never correctness.
logger = logging.getLogger('volkano.stub')


_HEADER = '''\
# Auto-generated stub for the volkano package — do not edit by hand.
# Regenerate with: python -m volkano.stub
#
# Every name here is resolved dynamically at runtime via PEP 562
# __getattr__ on the package; this file exists purely so static
# analysers can offer autocomplete and parameter hints.

from typing import Any
import enum

# Every type the registry emits is a cbase type (no raw ctypes): char*
# is Pointer[char], void* is Pointer[None], scalars are the width-explicit
# wrappers. The stub imports them all from cbase, never from ctypes.
from volkano.cbase import (
    Struct, Union, Handle, Pointer, Array, char,
    byte, ubyte, int8, uint8, int16, uint16, int32, uint32,
    int64, uint64, float32, float64, size, ssize,
)


# Real functions in volkano/__init__.py.
def registry(source: Any = ..., *, library: Any = ...) -> Any: ...
def get_registry() -> Any: ...

'''


def _classify(value: Any) -> str:
    """Tag a forced value so the stub-writer knows how to render it.

    The runtime registry mixes IntEnum / IntFlag classes (from
    ``make_enum``), cbase :class:`~cbase.Struct` / :class:`~cbase.Union`
    subclasses (from ``make_struct`` / ``make_union``),
    :class:`~cbase.Handle` subclasses (from ``make_handle``),
    function-pointer typedef classes (from ``make_funcpointer``),
    :class:`CommandSignature` callables (from ``make_command``), scalar
    wrapper classes, and bare numeric / string literals. Each category
    needs a different stub shape.

    Dispatch is on the cbase class hierarchy, not ctypes-generic checks
    or ``hasattr`` heuristics — the type's place in the substrate is the
    single source of truth.
    """
    if isinstance(value, type):
        if issubclass(value, enum.Enum):
            return 'enum'
        if issubclass(value, cbase.Struct):
            return 'struct'
        if issubclass(value, cbase.Union):
            return 'union'
        if issubclass(value, cbase.Handle):
            return 'handle'
        if issubclass(value, cbase.CFuncPtrBase):
            return 'funcpointer'
        return 'scalar'
    # CommandSignature is the only callable instance in the registry.
    from .vulkan_stdlib import CommandSignature
    if isinstance(value, CommandSignature):
        return 'command'
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float'
    if isinstance(value, str):
        return 'str'
    if value is None:
        return 'none'
    return 'any'


def _render_ctype(ct: Any, *, depth: int = 0) -> str:
    """Map a cbase type to its stub annotation — which is its own name.

    Every type the registry emits is a cbase type that already names
    itself with exactly the annotation we want:

    - scalars name themselves ``uint32`` / ``float32`` / …
    - handles / structs / unions / enums use their class name (each is
      emitted as a class in this stub)
    - a typed :class:`~cbase.Pointer` subclass is *literally* named
      ``Pointer[VkX]`` by :attr:`~cbase.Mixin.pointer_type` (and
      ``Pointer[c_char_p]`` for the blessed string pointer)
    - the blessed :data:`~cbase.c_char_p` names itself ``c_char_p``

    So the annotation simply *is* ``ct.__name__`` — one source of truth,
    no format-code table, no per-category dispatch.

    The lone exception is a fixed-size array. Its name embeds the length
    (``Array[uint32, 4]``), which isn't a valid static annotation — a C
    array's compile-time length has no place in a Python type. Arrays
    therefore render as the ctypes/typeshed shape ``Array[element]``
    (length dropped), or ``bytes`` for a ``char`` / byte element (which
    ctypes reads back as ``bytes``).

    ``depth`` guards against a pathological element-type chain.
    """
    if ct is None:
        return 'None'
    if not isinstance(ct, type):
        return 'Any'
    if depth > 4:
        return 'Any'

    if issubclass(ct, ctypes.Array):
        elem = getattr(ct, '_type_', None)
        if getattr(elem, '_type_', None) in ('c', 'B'):
            return 'bytes'
        return f'Array[{_render_ctype(elem, depth=depth + 1)}]'

    return ct.__name__


def _emit_constant(out: IO[str], name: str, value: Any, kind: str) -> None:
    """Constants are rendered with both their type *and* their literal value.

    Including the value lets IDEs surface it on hover, which is the
    single highest-value piece of information for Vulkan constants —
    you almost always want to know "what number is ``VK_STRUCTURE_-
    TYPE_APPLICATION_INFO`` again?" without scrolling to the source.
    """
    if kind == 'int':
        out.write(f'{name}: int = {value}\n')
    elif kind == 'float':
        out.write(f'{name}: float = {value!r}\n')
    elif kind == 'str':
        out.write(f'{name}: str = {value!r}\n')
    elif kind == 'bool':
        out.write(f'{name}: bool = {value}\n')
    elif kind == 'none':
        out.write(f'{name}: Any = ...\n')


def _emit_enum(out: IO[str], name: str, cls: type) -> None:
    """Enum classes get a real class declaration with member assignments.

    IntFlag groups inherit from :class:`enum.IntFlag` (so bitwise ops
    type-check), IntEnum from :class:`enum.IntEnum`. Member values are
    emitted as literals so hover-in-IDE shows the number.
    """
    base = 'IntFlag' if issubclass(cls, enum.IntFlag) else 'IntEnum'
    out.write(f'class {name}(enum.{base}):\n')
    members = list(cls.__members__.items())
    if not members:
        out.write('    pass\n')
        return
    for mname, mvalue in members:
        # No type annotation: a ``.pyi`` enum member must be a bare
        # ``MEMBER = value`` assignment. Annotating it (``MEMBER: int =
        # value``) makes type checkers reject it ("annotations not
        # allowed for enum members") and, worse, treat the whole enum as
        # a plain variable — cascading into "Variable not allowed in
        # type expression" at every site that uses the enum as a type.
        out.write(f'    {mname} = {int(mvalue)}\n')


def _emit_struct_like(out: IO[str], name: str, cls: type, kind: str) -> None:
    """Structures/unions get a class with typed field annotations.

    Field types are mapped via :func:`_render_ctype` — pointers become
    ``<target> | None``, enums become their class name, fixed char arrays
    become ``bytes``, integers become ``int``, etc. The constructor
    accepts the same types as defaults, matching ctypes' actual
    ``Structure.__init__`` behaviour (keyword args matching ``_fields_``).

    Base is :class:`volkano.cbase.Struct` / :class:`...Union` so
    the :class:`...Mixin` surface (``.ref`` / ``.ptr`` / ``.size`` /
    ``Pointer[X]`` / arithmetic) is reachable to static analysers via
    normal MRO traversal — no need to hand-stub the convenience
    properties.
    """
    base = 'Struct' if kind == 'struct' else 'Union'
    out.write(f'class {name}({base}):\n')
    fields = getattr(cls, '_fields_', ())
    if not fields:
        out.write('    pass\n')
        return
    typed_fields: list[tuple[str, str]] = []
    for field in fields:
        fname = field[0]
        ftype = field[1]
        rendered = _render_ctype(ftype)
        typed_fields.append((fname, rendered))
        out.write(f'    {fname}: {rendered}\n')
    # __init__: every field keyword-only with its rendered type and a
    # ``...`` default, so callers can write ``VkX(sType=…, pNext=None)``
    # and the type checker validates each kwarg.
    params = ', '.join(f'{fn}: {ft} = ...' for fn, ft in typed_fields)
    out.write(f'    def __init__(self, *, {params}) -> None: ...\n')


def _emit_handle(out: IO[str], name: str, cls: type) -> None:
    """Handles inherit from :class:`volkano.cbase.Handle`.

    The :class:`...Mixin` surface (``.ref`` / ``.ptr`` / ``.size`` /
    arithmetic via int-coercion) reaches the stub through the base
    class declaration, so static analysers see the full instance
    surface without per-handle property stubs.
    """
    out.write(f'class {name}(Handle): ...\n')


def _emit_command(out: IO[str], name: str, sig) -> None:
    """Commands are emitted as functions with typed parameters and return.

    Parameter and return types come from :func:`_render_ctype` against
    the resolved ctypes signature on :attr:`CommandSignature.params` /
    ``.rettype``. The annotations are precise enough for IDEs to show
    "this param is a ``VkInstanceCreateInfo``-shaped pointer" or "this
    returns a ``VkResult``", but loose enough (``| None`` on pointers,
    ``int`` on scalars) that the user can still pass ``ci.ref`` / plain
    Python ints without the type checker complaining.

    Defaults remain ``...`` even on required params — Vulkan call sites
    nearly always pass all arguments positionally, and the alternative
    (no defaults) would force every IDE call template to fill in every
    slot, which is more annoying than helpful for exploratory work.
    """
    params = ', '.join(
        f'{pname}: {_render_ctype(ptype)} = ...' for pname, ptype in sig.params)
    rettype = _render_ctype(sig.rettype)
    out.write(f'def {name}({params}) -> {rettype}: ...\n')


def _emit_scalar_or_funcpointer(out: IO[str], name: str) -> None:
    """ctypes scalars and function-pointer typedefs get a permissive alias."""
    out.write(f'{name}: Any\n')


def default_stub_path() -> str:
    """Path of the stub that ships beside this module (``volkano/__init__.pyi``)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '__init__.pyi')


#: Bump whenever a change to this module or to the XML parser alters
#: the *content* of a generated stub. The XML digest alone can't
#: detect that: identical input plus improved code yields a different
#: (and previously wrong) stub, which would otherwise never be
#: rewritten — leaving the stub contradicting the runtime on specific
#: values. Bumped to 2 when parse_int stopped eating hex digits.
_GENERATOR = 2

# The provenance stamp is written as its own first line rather than
# folded into _HEADER, so the template stays free of format braces and
# a reader only ever has to parse line 1.
_STAMP_RE = re.compile(r'^#\s*volkano-stub:\s*sha256=([0-9a-f]{64})\s+'
                       r'gen=(\d+)\s+'
                       r'header-version=(\S+)\s+source=(.*)$')


def _stamp_line(provenance: Any) -> str:
    return (f'# volkano-stub: sha256={provenance.sha256} '
            f'gen={_GENERATOR} '
            f'header-version={provenance.header_version} '
            f'source={provenance.label}\n')


def _expected_stamp(provenance: Any) -> tuple[str, int, str]:
    return (provenance.sha256, _GENERATOR, provenance.label)


def read_stub_provenance(path: str) -> tuple[str, int, str] | None:
    """Return ``(sha256, generator, source_label)`` from a stub's stamp.

    ``None`` if the file is absent, unreadable, or predates the current
    stamp format — all of which mean "regenerate", so the caller needs
    no special case for any of them.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # The stamp is line 1; scan a few more purely so a stray
            # leading blank line or editor artefact doesn't defeat it.
            for line in itertools.islice(f, 8):
                match = _STAMP_RE.match(line.strip())
                if match:
                    return match.group(1), int(match.group(2)), match.group(4)
    except OSError:
        return None
    return None


def sync_stub(registry: Any, *, path: str | None = None) -> bool:
    """Regenerate ``path`` if its stamp disagrees with ``registry``.

    Returns ``True`` if the stub was written. Never raises: a stale stub
    costs autocomplete, not correctness, and an installed package can
    sit on a read-only prefix — neither should sink an otherwise-good
    build.

    A registry with no provenance (one built from an
    :class:`~xml.etree.ElementTree.Element`, which has no identity to
    record) is never written. That absence is what keeps test fixtures
    from clobbering the real stub, without needing a flag to say so.
    """
    provenance = getattr(registry, '_provenance', None)
    if provenance is None:
        return False
    path = path or default_stub_path()
    if read_stub_provenance(path) == _expected_stamp(provenance):
        return False
    # Log before, not after: this forces every key in the registry and
    # is the multi-second pause a user notices on a cold install.
    logger.info('regenerating %s from %s', path, provenance.label)
    try:
        count = write_stub(path, registry=registry, provenance=provenance)
    except Exception as exc:
        # Deliberately broad. os.replace onto a .pyi held open by a
        # language server raises PermissionError on Windows, and an
        # emitter bug shouldn't be fatal either.
        logger.warning('could not regenerate %s: %s', path, exc)
        return False
    logger.info('regenerated %s: %d declarations', path, count)
    return True


def write_stub(path: str | None = None, *, registry: Any = None,
               provenance: Any = None) -> int:
    """Walk a registry and write a ``.pyi`` stub at ``path``.

    Returns the number of declarations emitted. Forces every resolvable
    entry as a side effect — that's the warm-up cost the runtime would
    pay incrementally anyway, paid up front here so the stub captures
    the full surface.

    ``path`` defaults to :func:`default_stub_path`. ``registry``
    defaults to the package singleton. ``provenance``, when given, is
    stamped onto the first line for :func:`sync_stub` to compare
    against later.

    The write is atomic — a temp file in the target directory followed
    by :func:`os.replace` — so a crash mid-emit can't leave a truncated
    1.5 MB stub for an IDE to choke on.
    """
    if path is None:
        path = default_stub_path()
    if registry is None:
        from . import get_registry
        registry = get_registry()

    # Bucket entries by kind so the stub groups related declarations
    # together — much easier to skim by hand than alphabetical order.
    buckets: dict[str, list[tuple[str, Any]]] = {
        kind: [] for kind in (
            'handle', 'enum', 'struct', 'union', 'funcpointer',
            'command', 'int', 'float', 'str', 'bool', 'scalar', 'any', 'none')
    }
    skipped: list[str] = []
    for key in sorted(registry):
        if key.startswith('_'):
            continue
        try:
            value = registry(key)
        except Exception:
            skipped.append(key)
            continue
        kind = _classify(value)
        buckets.setdefault(kind, []).append((key, value))

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.__init__.pyi.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            if provenance is not None:
                f.write(_stamp_line(provenance))
            f.write(_HEADER)

            # Order matters here for human readability — types first,
            # then constants that often reference them, then commands.
            if buckets['handle']:
                f.write('\n# --- Handles ---\n')
                for name, value in buckets['handle']:
                    _emit_handle(f, name, value)

            if buckets['enum']:
                f.write('\n# --- Enums ---\n')
                for name, value in buckets['enum']:
                    _emit_enum(f, name, value)
                    f.write('\n')

            if buckets['struct'] or buckets['union']:
                f.write('\n# --- Structs / unions ---\n')
                for name, value in buckets['struct']:
                    _emit_struct_like(f, name, value, 'struct')
                    f.write('\n')
                for name, value in buckets['union']:
                    _emit_struct_like(f, name, value, 'union')
                    f.write('\n')

            if buckets['scalar'] or buckets['funcpointer']:
                f.write('\n# --- Type aliases (primitives, bitmask typedefs, function pointers) ---\n')
                for name, _ in buckets['scalar']:
                    _emit_scalar_or_funcpointer(f, name)
                for name, _ in buckets['funcpointer']:
                    _emit_scalar_or_funcpointer(f, name)

            for const_kind in ('int', 'float', 'str', 'bool', 'none'):
                if buckets[const_kind]:
                    f.write(f'\n# --- {const_kind.capitalize()} constants ---\n')
                    for name, value in buckets[const_kind]:
                        _emit_constant(f, name, value, const_kind)

            if buckets['command']:
                f.write('\n# --- Commands ---\n')
                for name, value in buckets['command']:
                    _emit_command(f, name, value)

            if skipped:
                f.write('\n# --- Unresolved (registry forced raised; emitted as Any) ---\n')
                for name in skipped:
                    f.write(f'{name}: Any\n')
        os.replace(tmp, path)
    except BaseException:
        # Leave the previous stub intact; a half-written one is worse
        # than a stale one.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    total = sum(len(v) for v in buckets.values()) + len(skipped)
    return total


def main(argv: list[str] | None = None) -> int:
    """Maintenance CLI: refresh the cached XML and/or rewrite the stub.

    This is the only supported way to move a cached ``main.xml``
    forward. Nothing revalidates during a normal import, so without an
    explicit ``--refresh`` the cache is authoritative indefinitely.
    """
    from . import registry as build_registry
    from .xml_source import ENV_VAR, refresh_source, resolve_source

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-o', '--output', default=None,
                        help='where to write the .pyi (default: volkano/__init__.pyi)')
    parser.add_argument('--source', default=None,
                        help="'main', 'tag:<name>', 'branch:<name>', or a "
                             f"path to a local vk.xml (default: ${ENV_VAR}, "
                             f"else main)")
    parser.add_argument('--refresh', action='store_true',
                        help='redownload the source before generating '
                             '(no-op for a tag - those never move)')
    args = parser.parse_args(argv)

    source = args.source if args.source is not None else os.environ.get(ENV_VAR)
    if args.refresh:
        src = resolve_source(source)
        print(f'{src.label}: {refresh_source(src)}', file=sys.stderr)

    # Built through the public mode-2 entry point: no singleton, and
    # library=None so generating a stub never needs a Vulkan driver.
    reg = build_registry(source, library=None)
    output = args.output or default_stub_path()
    count = write_stub(output, registry=reg, provenance=reg._provenance)
    print(f'wrote {count} declarations to {output}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
