"""Where a Vulkan XML registry comes from: vocabulary, fetch, cache.

This module answers "which bytes, and from where", and hands back a
document plus the identity of what it read. It knows about Khronos's
repository — branch and tag refs, the layout move from ``src/spec/`` to
``xml/`` — and nothing whatsoever about Vulkan's schema.
:mod:`volkano.xml_parser` is the other half: given a document, produce
the thunk graph. Keeping them apart means the parser is testable
against an in-memory Element with no notion of caching, and this module
is testable with no notion of what a ``<type>`` means.

The source vocabulary is closed and prefix-tagged, so nothing is ever
inferred from the shape of a string:

- ``None`` / ``'main'`` — the Khronos main branch (mutable)
- ``'branch:<name>'`` — that branch head (mutable)
- ``'tag:<name>'`` — that tag, verbatim (immutable)
- anything else — a local filesystem path (never cached)

Two rules earn their keep repeatedly. Tag names pass through
**verbatim** rather than being rebuilt from a version number, because
rebuilding is inference about Khronos's naming history and is wrong for
89 of the 382 published tags. And repo layouts are *tried*, not
predicted: a 404 is an observation, a tag name is not evidence.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import re
import tempfile
import urllib.error
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

_RAW_ROOT = 'https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'

#: Where vk.xml lives inside the repo, newest layout first. Khronos
#: moved it from ``src/spec/`` to ``xml/`` between v1.1.70 and v1.2.131,
#: so old tags 404 on the modern path. Candidates are *tried*, not
#: inferred from the ref name — a 404 is an observation, whereas
#: guessing the layout from a tag would be the same mistake as
#: reconstructing the tag from a version.
_REPO_XML_PATHS = ('xml/vk.xml', 'src/spec/vk.xml')

_REF_URL = _RAW_ROOT + '/refs/{kind}s/{ref}/{path}'

#: Default branch. The newest published API surface — but note it is the
#: *branch*, not the newest release: it runs ahead of the tags and can
#: carry provisional extensions. Pin a tag if you need reproducibility.
DEFAULT_BRANCH = 'main'
MAIN_URL = _REF_URL.format(kind='head', ref=DEFAULT_BRANCH,
                           path=_REPO_XML_PATHS[0])

#: Tag names are passed through **verbatim**. Reconstructing them from a
#: parsed version looks like construction but is really inference about
#: Khronos's naming history, and it is wrong for 89 of the 382 published
#: tags: plain ``vX.Y.Z`` only starts at v1.1.70, and everything older
#: uses ``v1.0.33-core`` or ``v1.0-core+wsi-20160216``. ``v1.0.33`` has
#: never existed. The tag *is* the identifier; don't rebuild it.
_PREFIXES = {'tag:': 'tag', 'branch:': 'head'}

#: Only a tag shaped exactly like this can be cross-checked against
#: VK_HEADER_VERSION. The dated and ``-core`` families carry no
#: comparable patch component, so the check opts in rather than assuming.
_SEMVER_TAG_RE = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')

_UNSAFE_IN_FILENAME_RE = re.compile(r'[^A-Za-z0-9._+-]')
_USER_AGENT = 'volkano/0.1 (+https://github.com/KhronosGroup/Vulkan-Docs)'


class XmlSource(NamedTuple):
    """A resolved source: what to parse, and where it is cached."""

    kind: str                    # 'branch' | 'tag' | 'path'
    label: str                   # 'main' | 'v1.0.33-core' | abs path
    urls: tuple[str, ...]        # candidate URLs; empty for a local path
    path: pathlib.Path           # cache entry, or the file itself
    ref: str | None              # branch or tag name; None for a path

    @property
    def url(self) -> str | None:
        """The preferred URL — the modern repo layout."""
        return self.urls[0] if self.urls else None

    @property
    def mutable(self) -> bool:
        """Whether re-fetching this source could ever change anything.

        Structural, not inferred: a branch head moves, a tag does not.
        """
        return self.kind == 'branch'


class Provenance(NamedTuple):
    """Identity of the XML a registry was actually built from.

    ``sha256`` is what :func:`volkano.stub.sync_stub` compares against
    the stamp in an existing stub. ``label`` is compared alongside it so
    that switching between ``main`` and a local file still regenerates
    even in the pathological case where the bytes match.
    """

    sha256: str
    label: str
    header_version: int | None
    url: str | None


def _ref_urls(url_kind: str, ref: str) -> tuple[str, ...]:
    """Candidate URLs for one ref, newest repo layout first."""
    return tuple(_REF_URL.format(kind=url_kind, ref=ref, path=path)
                 for path in _REPO_XML_PATHS)


def _cache_filename(kind: str, ref: str) -> str:
    """A legible, collision-free cache filename for one ref.

    Refs may contain characters a filename can't (``feature/foo``) or
    that merely look odd (``v1.0-core+wsi-20160216``). Sanitising alone
    would let ``a/b`` and ``a-b`` collide and silently serve the wrong
    XML, so a short digest of the original is appended whenever
    sanitising actually changed something. Well-behaved refs — which is
    nearly all of them — keep a clean name.
    """
    safe = _UNSAFE_IN_FILENAME_RE.sub('-', ref)
    if safe != ref:
        safe = f'{safe}-{hashlib.sha1(ref.encode("utf-8")).hexdigest()[:8]}'
    return f'{kind}-{safe}.xml'


def resolve_source(source: str | None = None) -> XmlSource:
    """Resolve a source string to a URL and a cache location.

    The vocabulary is closed and prefix-tagged, so nothing has to be
    inferred from the shape of a string:

    - ``None`` or ``'main'`` — the Khronos main branch (mutable).
    - ``'branch:<name>'`` — that branch head (mutable).
    - ``'tag:<name>'`` — that tag, verbatim (immutable; never refetched
      once cached). ``tag:v1.4.359``, ``tag:v1.0.33-core``,
      ``tag:v1.0-core+wsi-20160216`` all work.
    - anything else — a local filesystem path, never cached.

    Arbitrary URLs are rejected. A cached URL is just a local file with
    extra steps, and the one case where a URL genuinely beats a path — a
    source that moves — is exactly the case caching breaks. Download it
    yourself and pass the path.
    """
    # Empty and whitespace-only count as "unspecified", not as the
    # current directory. `VOLKANO_XML=` (a declared-but-blank CI
    # variable, a blank line in a .env) is the common way to arrive
    # here, and Path('') resolving to '.' fails much later with a
    # baffling PermissionError.
    source = str(source).strip() if source is not None else ''
    if not source:
        source = DEFAULT_BRANCH
    if '://' in source:
        raise ValueError(
            f"{source!r} looks like a URL, which is not an accepted source. "
            f"Use 'main', 'tag:<name>', 'branch:<name>', or a path to a "
            f"local vk.xml (download the URL yourself if you need a "
            f"bespoke one).")

    prefix = next((p for p in _PREFIXES if source.startswith(p)), None)
    if prefix is not None:
        ref = source[len(prefix):].strip().strip('/')
        if not ref:
            raise ValueError(f"{source!r} names no ref after {prefix!r}.")
        url_kind = _PREFIXES[prefix]
        kind = 'tag' if url_kind == 'tag' else 'branch'
        return XmlSource(kind, ref, _ref_urls(url_kind, ref),
                         cache_dir() / _cache_filename(kind, ref), ref)

    if source == DEFAULT_BRANCH:
        # The one bare word, because it is also the default. Everything
        # else must say which namespace it means.
        return XmlSource('branch', DEFAULT_BRANCH, _ref_urls('head', DEFAULT_BRANCH),
                         cache_dir() / _cache_filename('branch', DEFAULT_BRANCH),
                         DEFAULT_BRANCH)

    path = pathlib.Path(source).expanduser()
    return XmlSource('path', str(path), (), path, None)


def expected_header_version(src: XmlSource) -> int | None:
    """The ``VK_HEADER_VERSION`` a tag's name implies, if it implies one.

    A semver-shaped tag's patch component *is* the header version
    (``v1.4.359`` -> 359), which makes a cheap cross-check that the
    cache entry is what its name claims. Returns ``None`` for branches,
    paths, and the older ``-core`` / dated tag families, which carry no
    comparable component — the check opts in rather than assuming.

    Tag-naming knowledge lives here rather than in the parser: the
    parser knows what ``VK_HEADER_VERSION`` means, this module knows
    what a tag name means.
    """
    if src.kind != 'tag' or src.ref is None:
        return None
    match = _SEMVER_TAG_RE.match(src.ref)
    return int(match.group(3)) if match else None


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


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
    logger.info('fetching %s', url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    logger.info('fetched %d bytes', len(data))
    return data


def _download_any(urls: tuple[str, ...]) -> bytes:
    """Try each candidate URL in turn; return the first that exists.

    Only a 404 advances to the next candidate — that means "this repo
    layout, not that one". Any other error (offline, 500, timeout) is
    about the request rather than the path, so it propagates
    immediately instead of being retried against a URL that is even
    less likely to work.
    """
    for index, url in enumerate(urls):
        try:
            return _download(url)
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or index == len(urls) - 1:
                raise
            logger.debug('%s is not there (404); trying the older layout', url)
    raise ValueError('no candidate URLs to try')


def ensure_cached(src: XmlSource, *, refresh: bool = False) -> pathlib.Path:
    """Return a local path for ``src``, downloading only if needed.

    There is no revalidation here — a cached entry is used as-is, and no
    network traffic happens during a normal import. A *missing* entry is
    still fetched; "no revalidation" is not "never fetch".
    """
    if src.kind == 'path':
        if not src.path.exists():
            hint = ''
            if _SEMVER_TAG_RE.match(src.label.strip()) or re.match(
                    r'^\d+\.\d+\.\d+$', src.label.strip()):
                # Almost certainly someone reaching for a release.
                ref = src.label.strip()
                hint = (f" (to pin a release, write 'tag:"
                        f"{ref if ref.startswith('v') else 'v' + ref}')")
            raise FileNotFoundError(f'no such vk.xml: {src.path}{hint}')
        return src.path
    if src.path.exists() and not refresh:
        logger.debug('using cached %s', src.path)
        return src.path
    _write_atomic(src.path, _download_any(src.urls))
    return src.path


def refresh_source(src: XmlSource) -> str:
    """Redownload ``src`` if that can possibly change anything.

    Returns a short status word for the CLI to print. A tag is immutable
    by construction, so refreshing one is a no-op — but only once it
    exists; a cold cache still fetches.
    """
    if src.kind == 'path':
        return 'local'
    if not src.mutable and src.path.exists():
        return 'pinned'
    before = src.path.read_bytes() if src.path.exists() else None
    data = _download_any(src.urls)
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
        if src.kind == 'path':
            raise
        # A cache entry we own failed to parse — most likely a download
        # interrupted before atomic writes existed. Discard and refetch
        # once; a second failure is real and propagates.
        logger.warning('cached %s did not parse; refetching', path)
        path = ensure_cached(src, refresh=True)
        data = path.read_bytes()
        root = ET.fromstring(data)
    return root, Provenance(hashlib.sha256(data).hexdigest(), src.label,
                            None, src.url)
