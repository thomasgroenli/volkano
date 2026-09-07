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
though all names resolve dynamically. **Importing volkano never writes
it** — generating it forces every entry in the registry and writes into
the installed package, which is far too much to do behind an ``import``
statement. ``python -m volkano update`` writes it; import only notices
when it has gone stale, and says so at INFO.
"""

from __future__ import annotations

import logging as _logging
import os as _os
from typing import Any


_logger = _logging.getLogger('volkano')

# Lazy singleton for the module facade. ``None`` means "not built yet";
# the first non-underscore attribute access builds it.
_registry: Any = None

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
    if _registry is not None:
        raise RuntimeError(
            'volkano.use() was called after the registry had already been '
            'built by an earlier attribute access; move it above the first '
            'use of volkano, or build a separate one with volkano.registry().')
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
    if _registry is None:
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
        # Publish before the stub check, so that a re-entrant
        # volkano.<attr> from anything it touches finds the singleton
        # rather than starting a second build.
        _registry = built
        _note_if_stub_is_stale(built)
    return _registry


def _note_if_stub_is_stale(built: Any) -> None:
    """Log a pointer at ``python -m volkano update``; never write.

    Deliberately only a log line. Regenerating here would force all
    8 000-odd entries and write a megabyte into site-packages as a side
    effect of ``import`` — slow on a cold install, silently impossible
    on a read-only prefix, and a tug-of-war between two projects that
    share one virtualenv and name different sources.

    Swallows everything: a missing or unreadable stub is the normal
    state of a fresh install, and nothing about this check is worth
    failing an import over.
    """
    try:
        from .stub import stub_is_stale
        if stub_is_stale(built):
            _logger.info(
                'volkano/__init__.pyi is out of date with %s; run '
                '`python -m volkano update` to refresh editor completions',
                getattr(getattr(built, '_provenance', None), 'uri',
                        'the current source'))
    except Exception:               # pragma: no cover - never load-bearing
        _logger.debug('stub staleness check failed', exc_info=True)


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
    base = {'registry', 'get_registry', 'use'}
    base.update(name for name in dir(get_registry())
                if not name.startswith('_'))
    return sorted(base)
