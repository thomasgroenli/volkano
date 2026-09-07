"""Where a Vulkan XML registry comes from: URI, fetch, cache.

This module answers "which bytes, and from where", and hands back a
document plus the identity of what it read. It knows how to fetch a URI
and how to cache one, and nothing whatsoever about Vulkan's schema.
:mod:`volkano.xml_parser` is the other half: given a document, produce
the thunk graph. Keeping them apart means the parser is testable
against an in-memory Element with no notion of caching, and this module
is testable with no notion of what a ``<type>`` means.

A source is a **URI** — or a filesystem path, which is one spelled
without ceremony:

- ``None`` — :data:`DEFAULT_URI`, Khronos's main branch.
- ``https://…`` / ``http://…`` — fetched once and cached forever.
- ``file://…`` — read in place, never cached.
- anything else — a filesystem path, resolved and read in place.

There is no vocabulary to learn beyond that, and volkano knows nothing
about Khronos's repository past the one default URI: no tag naming
scheme, no branch namespace, no directory layout. Pinning a release
means naming the URL of that release's ``vk.xml`` — the identifier
GitHub already gives you, which stays correct across the layout move
from ``src/spec/`` to ``xml/`` and across all three of the tag-naming
families Khronos has used.

Nothing is ever revalidated. A URI is fetched when it is missing from
the cache and at no other time, so a normal import does no network I/O
whatever the source is. ``python -m volkano update`` is the one thing
that refetches.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import re
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, NamedTuple


# Acquisition logger. Network fetches surface at INFO (so a single
# ``logging.basicConfig(level=logging.INFO)`` shows them); cached-file
# usage drops to DEBUG so the steady-state case stays quiet.
logger = logging.getLogger('volkano.source')


#: Name of the environment variable that selects the source for the
#: package facade. Read by :func:`volkano.get_registry` and nowhere
#: else — :func:`resolve_source` is deliberately free of ambient state,
#: so a registry built through :func:`volkano.registry` depends only on
#: its arguments.
ENV_VAR = 'VOLKANO_XML'

#: The default source: Khronos's main branch. The newest published API
#: surface — but note it is the *branch*, not the newest release: it
#: runs ahead of the tags and can carry provisional extensions. Name a
#: tag's URL instead if you need reproducibility.
#:
#: This constant is the only thing in volkano that knows anything about
#: Khronos's repository, and it knows exactly one URL rather than a
#: scheme for building them.
DEFAULT_URI = ('https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'
               '/refs/heads/main/xml/vk.xml')

#: Schemes we fetch. Everything else is rejected by name rather than
#: attempted and failed, so ``ftp://`` gets a sentence instead of a
#: :class:`~urllib.error.URLError` from three frames down.
_FETCHABLE_SCHEMES = ('http', 'https')

#: A scheme needs **two or more** characters before ``://``. One would
#: match the ``C`` of ``C://weird/but/legal``, and a Windows drive
#: letter must never be read as a scheme. Ordinary ``C:\\…`` has no
#: ``//`` and so never reaches this at all.
_SCHEME_RE = re.compile(r'^([A-Za-z][A-Za-z0-9+.\-]+)://')

_UNSAFE_IN_FILENAME_RE = re.compile(r'[^A-Za-z0-9._+-]')
_USER_AGENT = 'volkano/0.1 (+https://github.com/KhronosGroup/Vulkan-Docs)'

#: How much of a URI to keep in its cache filename. Long enough that
#: the tail of a raw.githubusercontent URL still names the ref, short
#: enough to stay well inside every filesystem's limit once the digest
#: is appended.
_CACHE_NAME_CHARS = 56


class XmlSource(NamedTuple):
    """A resolved source: what to read, and where it lives locally."""

    uri: str                     # 'https://…' or 'file:///…'
    path: pathlib.Path           # cache entry, or the file itself

    @property
    def local(self) -> bool:
        """Whether the bytes already sit on disk under the user's control.

        A local file is read in place and never copied into the cache:
        caching a path would only add a second, staler copy of a file
        the user can already edit.
        """
        return self.uri.startswith('file:')


class Provenance(NamedTuple):
    """Identity of the XML a registry was actually built from.

    ``sha256`` is what :func:`volkano.stub.sync_stub` compares against
    the stamp in an existing stub. ``uri`` is compared alongside it so
    that switching between two sources still regenerates even in the
    pathological case where the bytes match — and unlike a bare label, a
    URI is absolute, so it still means the same thing when the stamp is
    read back from a different working directory tomorrow.

    ``header_version`` is the odd one out: it is filled in by
    :func:`volkano.xml_parser.parse_registry` once the document has
    actually been read, because knowing what ``VK_HEADER_VERSION`` means
    is the parser's job, not this module's. It is ``None`` on the
    instance :func:`load_xml` returns.
    """

    sha256: str
    uri: str
    header_version: int | None


def _cache_filename(uri: str) -> str:
    """A legible, collision-free cache filename for one URI.

    The digest is what makes it correct — two URIs can differ only in
    characters a filename cannot hold — and the sanitised tail is what
    makes an ``ls`` of the cache directory readable. The tail is never
    load-bearing, so truncating it costs nothing.
    """
    tail = _UNSAFE_IN_FILENAME_RE.sub('-', uri.split('://', 1)[-1])
    tail = tail[-_CACHE_NAME_CHARS:].strip('-.')
    if tail.endswith('.xml'):
        tail = tail[:-4].strip('-.')
    digest = hashlib.sha256(uri.encode('utf-8')).hexdigest()[:12]
    return f'{tail}-{digest}.xml' if tail else f'{digest}.xml'


def _local_source(path: pathlib.Path) -> XmlSource:
    """A source read in place: absolute, symlink-free, never cached.

    Resolving matters for more than tidiness. The URI goes into the
    stub's provenance stamp, and a relative path would record an
    identity that means something different depending on where the
    interpreter was started — regenerating the stub on nothing more
    than a ``cd``.
    """
    path = path.expanduser().resolve()
    return XmlSource(path.as_uri(), path)


def resolve_source(source: str | None = None) -> XmlSource:
    """Resolve a source to a URI and the local path holding its bytes.

    - ``None`` (or blank) — :data:`DEFAULT_URI`.
    - ``https://…`` / ``http://…`` — cached under :func:`cache_dir`.
    - ``file://…`` — that file, read in place.
    - anything else — a filesystem path, read in place.

    Pure: it reads no environment and touches no filesystem, so
    resolving a source has no side effect and two calls with the same
    argument are indistinguishable.
    """
    # Empty and whitespace-only count as "unspecified", not as the
    # current directory. `VOLKANO_XML=` (a declared-but-blank CI
    # variable, a blank line in a .env) is the common way to arrive
    # here, and Path('') resolving to '.' fails much later with a
    # baffling PermissionError.
    source = str(source).strip() if source is not None else ''
    if not source:
        source = DEFAULT_URI

    match = _SCHEME_RE.match(source)
    scheme = match.group(1).lower() if match else None

    if scheme in _FETCHABLE_SCHEMES:
        return XmlSource(source, cache_dir() / _cache_filename(source))
    if scheme == 'file':
        parts = urllib.parse.urlsplit(source)
        if parts.netloc and parts.netloc.lower() != 'localhost':
            raise ValueError(
                f'{source!r} names the host {parts.netloc!r}; volkano reads '
                f'file:// URIs from the local filesystem only.')
        # url2pathname is what turns '/C:/x' back into 'C:\\x' on
        # Windows and undoes percent-encoding everywhere.
        return _local_source(
            pathlib.Path(urllib.request.url2pathname(parts.path)))
    if scheme is not None:
        raise ValueError(
            f'{source!r} uses the {scheme!r} scheme, which volkano does not '
            f'fetch. Use an http(s) URL, a file:// URI, or a path to a local '
            f'vk.xml (download it yourself if it lives somewhere else).')

    return _local_source(pathlib.Path(source))


def cache_dir() -> pathlib.Path:
    """Per-user cache directory for downloaded XML.

    Honours XDG_CACHE_HOME on POSIX and LOCALAPPDATA on Windows. Pure —
    it does not create the directory, so resolving a source has no
    filesystem side effect; :func:`_write_atomic` creates it on the way
    to writing.
    """
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~/AppData/Local')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return pathlib.Path(base) / 'volkano' / 'xml'


def _write_atomic(path: pathlib.Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file and :func:`os.replace`.

    An interrupted plain write would leave a truncated XML in the cache
    that then fails to parse on every subsequent run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f'.{path.name}.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _download(uri: str) -> bytes:
    req = urllib.request.Request(uri, headers={'User-Agent': _USER_AGENT})
    logger.info('fetching %s', uri)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    logger.info('fetched %d bytes', len(data))
    return data


def ensure_cached(src: XmlSource, *, refresh: bool = False) -> pathlib.Path:
    """Return a local path for ``src``, downloading only if needed.

    There is no revalidation here — a cached entry is used as-is, and no
    network traffic happens during a normal import. A *missing* entry is
    still fetched; "no revalidation" is not "never fetch".
    """
    if src.local:
        if not src.path.exists():
            raise FileNotFoundError(
                f'no such vk.xml: {src.path} (a source is a filesystem path '
                f'or an http(s)/file URI; to pin a Khronos release, name the '
                f'raw URL of that release\'s vk.xml)')
        return src.path
    if src.path.exists() and not refresh:
        logger.debug('using cached %s', src.path)
        return src.path
    _write_atomic(src.path, _download(src.uri))
    return src.path


def refresh_source(src: XmlSource) -> str:
    """Redownload ``src``, reporting what changed as a short status word.

    Unconditional, because a URI carries no evidence about whether what
    it names can move: a tag's URL and a branch's URL are the same
    shape. Rather than guess, volkano never revalidates on its own and
    always refetches when explicitly told to — the cost of being wrong
    is one request the caller asked for.
    """
    if src.local:
        return 'local'
    before = src.path.read_bytes() if src.path.exists() else None
    data = _download(src.uri)
    _write_atomic(src.path, data)
    if before is None:
        return 'fetched'
    return 'unchanged' if data == before else 'updated'


def load_xml(source: Any = None) -> tuple[ET.Element, Provenance | None]:
    """Read ``source`` into an XML root plus the identity of those bytes.

    An :class:`~xml.etree.ElementTree.Element` is returned as-is with
    **no** provenance — it has no stable identity to record. That
    absence is load-bearing: it is what stops in-memory test fixtures
    from ever rewriting the package stub.
    """
    if isinstance(source, ET.Element):
        return source, None
    src = resolve_source(source)
    path = ensure_cached(src)
    data = path.read_bytes()
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        if src.local:
            raise
        # A cache entry we own failed to parse — a download interrupted
        # before atomic writes existed, or a captive portal that served
        # a login page with a 200. Discard and refetch once; a second
        # failure is real and propagates.
        logger.warning('cached %s did not parse; refetching', path)
        path = ensure_cached(src, refresh=True)
        data = path.read_bytes()
        root = ET.fromstring(data)
    return root, Provenance(hashlib.sha256(data).hexdigest(), src.uri, None)
