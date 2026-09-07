"""Lazy, XML-driven Vulkan bindings for Python.

There are two ways to use this package, and the first is defined in
terms of the second.

**Explicit** — :func:`registry` builds an independent registry. It has
no ambient inputs: no environment, no process-wide singleton, and it
never writes to your source tree::

    import volkano
    vk = volkano.registry(library=True)
    vk.vkCreateInstance(...)

**Defaults** — the package itself is a module-style facade over one
lazily-built registry, so a bare ``import volkano`` gives you
``volkano.vkCreateInstance(...)`` / ``volkano.VkInstance`` /
``volkano.VK_SUCCESS`` with no ceremony::

    import volkano as vk
    instance = vk.VkInstance()
    vk.vkCreateInstance(...)

That facade *is* a :func:`registry` call, applied to two environment
variables and cached in a module global — see :func:`get_registry`.
There is only one implementation.

Configuring the facade
----------------------

``VOLKANO_XML`` selects the source: unset for the Khronos main branch,
otherwise a URI or a path to a ``vk.xml``. ``VOLKANO_LIBRARY`` gives
the Vulkan loader's path; unset means autodetect, and the literal
``none`` means don't attach one at all (useful for codegen and CI,
where there is no driver and nothing to call).

Both are read on first use, not at import, and only by
:func:`get_registry`. Everything else takes arguments.

:func:`use` is the in-process equivalent, for a script that would
rather say so in code than in its environment. It has to run before the
first attribute access, because that is what builds the registry — so
it is a convenience, not a replacement: an environment variable is
still the only thing that can configure a volkano imported
transitively, or one reached through ``python -c``.

The stub
--------

A companion ``__init__.pyi`` declares every name with light type
annotations, so IDEs surface autocomplete and parameter hints even
though all names resolve dynamically. It is written on first use, by
the same call that builds the registry: find the XML in the cache or
fetch it, build, and bring the stub level with what was built.

That is one rule rather than two, and it holds whatever state a fresh
install starts from — no cached XML, no stub, or a stub left over from
a source the user has since changed. What it costs is paid once: the
stamp on line 1 of the stub records the XML it came from, so every
later import compares one line and writes nothing.

The case this does not serve well is two projects sharing one
virtualenv and naming different sources: each one's first import
rewrites the other's stub, a full regeneration every time they
alternate. Accepted for v1 — it costs autocomplete and a second of
import, never correctness, and avoiding it means keeping the second
rule this replaces.

``python -m volkano update`` is the other half — the only thing that
refetches a URI already in the cache, since nothing revalidates on its
own.
"""

from __future__ import annotations

import logging as _logging
import os as _os
import threading as _threading
from typing import Any


#: The one place this project's version is written. ``pyproject.toml``
#: reads it from here (``dynamic = ["version"]``), so the distribution
#: metadata, ``volkano.__version__`` and the User-Agent the fetcher
#: sends cannot drift apart the way a second copy would.
__version__ = '0.0.1'


_logger = _logging.getLogger('volkano')

# Lazy singleton for the module facade. ``None`` means "not built yet";
# the first non-underscore attribute access builds it.
_registry: Any = None

# Guards the build, so racing threads share one registry rather than
# each parsing the XML and handing out its own ctypes classes — two
# VkInstance classes for one Vulkan type is an
# ``expected VkInstance instance, got VkInstance`` waiting to happen.
# Re-entrant because anything the build touches may itself reach for a
# ``volkano.<name>``. The kernel has its own lock for forcing; this one
# covers the step above it, which had none.
_lock = _threading.RLock()

# Distinguishes "use() was never called for this" from "use() was called
# with None" — the latter is a real choice meaning "take the default",
# and has to override an environment variable that says otherwise.
_UNSET: Any = object()
_source_override: Any = _UNSET
_library_override: Any = _UNSET

# Names of this package's real submodules (``cbase``, ``stub``,
# ``vulkan_stdlib``, …). They must shadow the dynamic Vulkan namespace in
# :func:`__getattr__`: a bare ``getattr(volkano, 'cbase')`` — which both
# ``from volkano import cbase`` and ``mock.patch`` target resolution
# perform — has to return the submodule, never force a registry build for
# a key that happens to be named after one. Computed once at import from
# the package directory so a newly-added submodule is picked up for free.
_SUBMODULES = frozenset(
    name[:-3] for name in _os.listdir(_os.path.dirname(__file__))
    if name.endswith('.py') and name != '__init__.py')


def registry(source: Any = None, *, library: Any = None):
    """Build an independent :class:`~volkano.vulkan_stdlib.VkRegistry`.

    ``source`` is ``None`` (Khronos's main branch), an http(s) URL or
    ``file://`` URI, a path to a local ``vk.xml``, or an
    :class:`~xml.etree.ElementTree.Element`. To pin a release, name the
    raw URL of that release's ``vk.xml``::

        MAIN = ('https://raw.githubusercontent.com/KhronosGroup/'
                'Vulkan-Docs/refs/tags/v1.4.359/xml/vk.xml')
        vk = volkano.registry(MAIN, library=True)

    ``library`` is ``None`` (attach nothing), ``True`` (autodetect), a
    path, or an open CDLL; either way the loader is not opened until
    the first command is actually called.

    A pure function of its arguments — it reads no environment and
    touches no global state, so several registries can coexist and none
    of them will rewrite the package's ``.pyi``.
    """
    from .vulkan_stdlib import build_registry
    return build_registry(source, library=library)


def use(source: Any = _UNSET, *, library: Any = _UNSET) -> None:
    """Configure the facade from code instead of from the environment.

    ``volkano.use('https://…/vk.xml')`` at the top of a script is the
    in-process equivalent of setting ``VOLKANO_XML``, and takes
    precedence over it. Arguments left unspecified are left alone, so
    setting one does not silently reset the other to its default.

    Must be called before the first ``volkano.<name>`` access, since
    that is what builds the registry. Calling it afterwards raises
    rather than being quietly ignored: a call that looks like it
    configured the registry but didn't is the worst of the three
    available behaviours, and there is no safe way to rebuild in place
    once ctypes classes from the old registry are loose in the program.
    """
    global _source_override, _library_override
    # Same lock as the build, so a call that races one either lands
    # wholly before it or raises — never half-applies to a registry
    # that is already being constructed from the old settings.
    with _lock:
        if _registry is not None:
            raise RuntimeError(
                'volkano.use() was called after the registry had already '
                'been built by an earlier attribute access; move it above '
                'the first use of volkano, or build a separate one with '
                'volkano.registry().')
        if source is not _UNSET:
            _source_override = source
        if library is not _UNSET:
            _library_override = library


def get_registry():
    """Return the facade's registry, building it on first call.

    This is the *only* place the environment is read: mode 1 is mode 2
    applied to ``VOLKANO_XML`` / ``VOLKANO_LIBRARY`` (or to whatever
    :func:`use` overrode them with), plus this module global and the
    stub check.
    """
    global _registry
    # Fast path, deliberately lock-free: once published, the singleton
    # is never reassigned, and reading a module global is atomic.
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is not None:
            return _registry
        # An unset variable and a blank one mean the same thing: take
        # the default. Blank is easy to produce by accident (`export
        # VOLKANO_XML=`, a declared-but-empty CI variable) and must not
        # be mistaken for a value.
        if _source_override is not _UNSET:
            source = _source_override
        else:
            source = _os.environ.get('VOLKANO_XML', '').strip() or None
        if _library_override is not _UNSET:
            library = _library_override
        else:
            # 'none' is spelled as a string because environment
            # variables are strings; the API itself keeps
            # None|True|path|CDLL. Translating here keeps that magic
            # word at the boundary.
            spec = _os.environ.get('VOLKANO_LIBRARY', '').strip() or True
            library = None if spec == 'none' else spec
        built = registry(source=source, library=library)
        # Publish inside the lock; sync outside it. Publishing first is
        # what lets a re-entrant ``volkano.<attr>`` — from the sync or
        # from anything it touches — find the singleton instead of
        # starting a second build. Releasing first is what keeps every
        # other thread from waiting on a stub regeneration that has
        # nothing to do with the name it asked for.
        _registry = built
    _sync_stub(built)
    return built


def _sync_stub(built: Any) -> None:
    """Bring ``__init__.pyi`` level with what was just built.

    A no-op in the steady state — :func:`~volkano.stub.sync_stub` reads
    the stamp on line 1 and returns without forcing anything when it
    already agrees. It writes on the three occasions where it doesn't:
    a fresh install with no stub, a source the user has changed, and a
    stub this generator has outgrown.

    Swallows everything, and :func:`~volkano.stub.sync_stub` swallows
    its own write failures on top of that. A stub costs autocomplete
    and never correctness, so there is no state of it — missing,
    unreadable, unwritable — worth failing an ``import`` over.
    """
    try:
        from .stub import sync_stub
        sync_stub(built)
    except Exception:               # pragma: no cover - never load-bearing
        _logger.debug('stub sync failed', exc_info=True)


def __getattr__(name: str) -> Any:
    """PEP 562: invoked when ``volkano.<name>`` misses the package namespace.

    Delegates to the singleton registry's attribute access (which in
    turn force-resolves the entry through the cache). Names starting
    with underscore go through the normal Python attribute path and
    never trigger registry resolution, so introspection tools probing
    for ``__loader__`` etc. don't accidentally force the build.
    """
    if name.startswith('_'):
        raise AttributeError(name)
    if name in _SUBMODULES:
        import importlib
        return importlib.import_module(f'{__name__}.{name}')
    return getattr(get_registry(), name)


def __dir__() -> list[str]:
    """Surface every registered name to ``dir(volkano)`` and tab completion."""
    base = {'registry', 'get_registry', 'use', '__version__'}
    base.update(name for name in dir(get_registry())
                if not name.startswith('_'))
    return sorted(base)
