"""Generate a ``.pyi`` stub file for the :mod:`volkano` package.

The runtime package resolves all 8 000+ Vulkan symbols dynamically via
PEP 562 :func:`__getattr__`. That works at runtime but is invisible to
static analysers — IDEs can't autocomplete ``volkano.vk…`` because they
never see those names declared. This module walks the live registry
(forcing every resolvable entry) and emits a stub file that declares
each name with a permissive type, so type checkers and language
servers gain full coverage of the Vulkan API surface.

Use as a CLI::

    python -m volkano update                # refetch, then rewrite the stub
    python -m volkano update -o other.pyi   # write it somewhere else

Or programmatically::

    from volkano.stub import update
    update()

The stub carries a provenance stamp on its first line naming the
sha256 of the XML it was generated from. :func:`sync_stub` compares
that against the registry it is handed and rewrites only on a
mismatch. Stamping content rather than keying off "did we just
download something" is what makes it self-healing: a deleted stub, a
pre-populated cache and an interrupted write all repair themselves on
the next build, with no bookkeeping to get out of step.

That comparison is cheap enough — one line read, nothing forced — that
:func:`volkano.get_registry` runs it on every build, which is why a
fresh install gets a stub without being asked to. The expensive half
runs only on a mismatch, and :func:`sync_stub` keeps it that way: it
declines up front when the target isn't writable (an installed package
on a read-only prefix would otherwise force all 8 000+ entries every
import only to fail the write), and swallows whatever the write raises.

The stub is purely a hint for tooling; the package itself works
without it, which is what keeps every one of those failures harmless.
"""

from __future__ import annotations

import argparse
import ast
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
# Regenerate with: python -m volkano update
#
# Every name here is resolved dynamically at runtime via PEP 562
# __getattr__ on the package; this file exists purely so static
# analysers can offer autocomplete and parameter hints.

from typing import Any, TypeAlias
import enum

# Every type the registry emits is a cbase type (no raw ctypes): char*
# is Pointer[char], void* is Pointer[None], scalars are the width-explicit
# wrappers. The stub imports them all from cbase, never from ctypes.
from volkano.cbase import (
    Struct, Union, Handle, Pointer, Array, char,
    byte, ubyte, int8, uint8, int16, uint16, int32, uint32,
    int64, uint64, float32, float64, size, ssize,
)


# Real names in volkano/__init__.py.
__version__: str
def registry(source: Any = ..., *, library: Any = ...) -> Any: ...
def get_registry() -> Any: ...

'''


def _header_bound_names() -> frozenset[str]:
    """Names :data:`_HEADER` already binds, plus the builtins we annotate with.

    A registry entry whose key is one of these must not be declared
    again. The header imports ``float32`` from cbase *as a type* and the
    emitters write ``x: float32`` thousands of times over; a later
    ``float32: Any`` rebinds the name to a variable, and every one of
    those annotations becomes "Variable not allowed in type expression".
    The same trap catches the C spellings that shadow builtins — vk.xml
    registers ``int``, ``float`` and ``bool``, against the ``: int`` /
    ``: float`` / ``: bool`` the constant emitter writes.

    Skipping them loses nothing: the import *is* the declaration, and it
    is a better one — ``char`` reaches a checker as the cbase class the
    registry actually returns rather than as ``Any``.

    Read out of the header text rather than listed by hand. The header
    and the emitters are edited independently, and a hand-kept copy of
    one inside the other is exactly the kind of thing that drifts
    silently until a stub stops type-checking.
    """
    names = {'int', 'float', 'str', 'bool', 'bytes'}
    for node in ast.walk(ast.parse(_HEADER)):
        if isinstance(node, ast.alias):
            names.add((node.asname or node.name).split('.')[0])
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(names)


_HEADER_NAMES = _header_bound_names()


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
    # Before the int check, and deliberately: an enum member *is* an
    # int, and since enumerants resolve to their group's member
    # (:func:`vulkan_stdlib.enum_member`) the plain-int branch would
    # claim nearly every constant in the registry and annotate it
    # ``int``. A composite IntFlag value has no name to spell and falls
    # through to that branch, which is the honest rendering for it.
    if isinstance(value, enum.Enum):
        # The name has to round-trip as a single attribute of its class:
        # a composite IntFlag value is nameless on 3.10 and named
        # ``'A|B'`` from 3.11 on, and neither can be spelled as one
        # attribute access. Those render as plain ints.
        mname = value.name
        if mname and getattr(type(value), mname, None) is value:
            return 'enum_member'
        return 'int'
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
    if kind == 'enum_member':
        # ``{value}`` is not usable here: IntEnum's ``__str__``
        # prints the member on 3.10 and the bare number from 3.11
        # on. Spelling the group and the member out keeps the stub
        # identical across interpreters - and annotating with the
        # group rather than ``int`` is what tells an IDE that
        # ``vk.VK_SUCCESS`` and ``VkResult.VK_SUCCESS`` are the
        # same object.
        group = type(value).__name__
        out.write(f'{name}: {group} = {group}.{value.name}\n')
    elif kind == 'int':
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
    """ctypes scalars and function-pointer typedefs get a permissive alias.

    ``TypeAlias``, not a bare ``: Any``, because these names are used
    in *annotation* position elsewhere in the same file - a struct
    field typed ``PFN_vkAllocationFunction``, for one. A plain
    annotated assignment declares a variable, and a checker reading
    ``x: PFN_...`` afterwards reports "Variable not allowed in type
    expression"; the ``TypeAlias`` form says the name is a type and is
    exactly as permissive.
    """
    out.write(f'{name}: TypeAlias = Any\n')


def _is_writable(path: str) -> bool:
    """Whether ``path`` could plausibly be written (best effort).

    The stub is replaced by writing a sibling temp file and renaming
    over it, so the directory's permissions are what matter, not the
    file's. Best effort by nature — Windows ACLs and a language server
    holding the ``.pyi`` open are both invisible here — which is why
    :func:`sync_stub` still wraps the write itself. This only exists to
    keep the common, *statically* unwritable case from re-forcing the
    whole registry on every import.
    """
    return os.access(os.path.dirname(os.path.abspath(path)), os.W_OK)


def default_stub_path() -> str:
    """Path of the stub that ships beside this module (``volkano/__init__.pyi``)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '__init__.pyi')


#: Bump whenever a change to this module or to the XML parser alters
#: the *content* of a generated stub. The XML digest alone can't
#: detect that: identical input plus improved code yields a different
#: (and previously wrong) stub, which would otherwise never be
#: rewritten — leaving the stub contradicting the runtime on specific
#: values. Bumped to 2 when parse_int stopped eating hex digits, and
#: to 3 when enumerants became enum members rather than bare ints, and
#: to 4 when the names the header imports stopped being re-declared
#: and scalar / funcpointer aliases became TypeAlias.
_GENERATOR = 4

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
            f'source={provenance.uri}\n')


def _expected_stamp(provenance: Any) -> tuple[str, int, str]:
    return (provenance.sha256, _GENERATOR, provenance.uri)


def read_stub_provenance(path: str) -> tuple[str, int, str] | None:
    """Return ``(sha256, generator, source_uri)`` from a stub's stamp.

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
    ``update``.

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
    if not _is_writable(path):
        # Checked before the walk, not after: the walk is the expensive
        # part, and on a read-only prefix it would be paid on every
        # import for a write that cannot land.
        logger.info('%s is out of date but not writable; run '
                    '`python -m volkano update` somewhere it can be '
                    'written', path)
        return False
    logger.info('regenerating %s from %s', path, provenance.uri)
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
            'command', 'enum_member', 'int', 'float', 'str', 'bool',
            'scalar', 'any', 'none')
    }
    skipped: list[str] = []
    for key in sorted(registry):
        if key.startswith('_'):
            continue
        if key in _HEADER_NAMES:
            # Declared already by the header's imports, and better than
            # we could here — see :func:`_header_bound_names`.
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

            # After the enum classes above, which these reference by
            # name: a reader takes a .pyi in file order, so the group
            # has to be declared before a constant annotated with it.
            if buckets['enum_member']:
                f.write('\n# --- Enumerants (each resolves to its group member) ---\n')
                for name, value in buckets['enum_member']:
                    _emit_constant(f, name, value, 'enum_member')

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


def update(source: Any = None, *, output: str | None = None,
           fetch: bool = True) -> tuple[str, int]:
    """Refetch ``source``, rebuild from it, and rewrite the stub.

    The whole of ``python -m volkano update``, minus argument parsing.
    Returns ``(fetch_status, declarations_written)`` — the status being
    one of :func:`~volkano.xml_source.refresh_source`'s words, or
    ``'cached'`` when ``fetch`` is off.

    This is the only supported way to move a cached URI forward.
    Nothing revalidates during a normal import, so between two
    ``update`` runs the cache is authoritative indefinitely.
    """
    from . import registry as build_registry
    from .xml_source import refresh_source, resolve_source

    src = resolve_source(source)
    status = refresh_source(src) if fetch else 'cached'
    # Built through the public mode-2 entry point: no singleton, and
    # library=None so generating a stub never needs a Vulkan driver.
    reg = build_registry(source, library=None)
    count = write_stub(output or default_stub_path(), registry=reg,
                       provenance=reg._provenance)
    return status, count


def main(argv: list[str] | None = None) -> int:
    """Argument parsing for ``python -m volkano update``.

    Lives here rather than in ``__main__`` so that the command is
    importable and testable without a subprocess.
    """
    from .xml_source import ENV_VAR, resolve_source

    parser = argparse.ArgumentParser(
        prog='python -m volkano', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command')
    up = sub.add_parser('update',
                        help='refetch the XML and rewrite the .pyi stub')
    up.add_argument('source', nargs='?', default=None,
                    help=f'a URI or path to a vk.xml (default: ${ENV_VAR}, '
                         f'else the Khronos main branch)')
    up.add_argument('-o', '--output', default=None,
                    help='where to write the .pyi (default: volkano/__init__.pyi)')
    up.add_argument('--no-fetch', action='store_true',
                    help='rebuild the stub from the cached XML without '
                         'redownloading')
    args = parser.parse_args(argv)
    if args.command != 'update':
        parser.print_help(sys.stderr)
        return 2

    source = args.source if args.source is not None else os.environ.get(ENV_VAR)
    src = resolve_source(source)
    status, count = update(source, output=args.output,
                           fetch=not args.no_fetch)
    print(f'{src.uri}: {status}', file=sys.stderr)
    print(f'wrote {count} declarations to '
          f'{args.output or default_stub_path()}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
