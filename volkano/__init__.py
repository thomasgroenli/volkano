"""Lazy, XML-driven Vulkan bindings for Python.

There are two ways to use this package, and the first is defined in
terms of the second.

**Explicit** — :func:`registry` builds an independent registry. It has
no ambient inputs: no environment, no process-wide singleton, and it
never writes to your source tree::

    import volkano
    vk = volkano.registry('1.4.359', library=True)
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

``VOLKANO_XML`` selects the source: unset or ``main`` for the Khronos
main branch, a version like ``1.4.359`` to pin a tag, or a path to a
local ``vk.xml``. ``VOLKANO_LIBRARY`` gives the Vulkan loader's path;
unset means autodetect, and the literal ``none`` means don't attach one
at all (useful for codegen and CI, where there is no driver and nothing
to call).

Both are read on first use, not at import, and only by
:func:`get_registry`. Everything else takes arguments.

The stub
--------

A companion ``__init__.pyi`` declares every name with light type
annotations, so IDEs surface autocomplete and parameter hints even
though all names resolve dynamically. It is regenerated whenever it
disagrees with the XML the facade is built from, and only ever by the
facade — a registry you construct with :func:`registry` will not touch
it.
"""

from __future__ import annotations

import os as _os
from typing import Any


# Lazy singleton for the module facade. ``None`` means "not built yet";
# the first non-underscore attribute access builds it.
_registry: Any = None

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

    ``source`` is ``None`` / ``'main'`` (Khronos main branch), a version
    like ``'1.4.359'``, a path to a local ``vk.xml``, or an
    :class:`~xml.etree.ElementTree.Element`. ``library`` is ``None``
    (attach nothing), ``True`` (autodetect), a path, or an open CDLL;
    either way the loader is not opened until the first command is
    resolved.

    A pure function of its arguments — it reads no environment and
    touches no global state, so several registries can coexist and none
    of them will rewrite the package's ``.pyi``.
    """
    from .vulkan_stdlib import build_registry
    return build_registry(source, library=library)


def get_registry():
    """Return the facade's registry, building it on first call.

    This is the *only* place the environment is read: mode 1 is mode 2
    applied to ``VOLKANO_XML`` / ``VOLKANO_LIBRARY``, plus this module
    global and the stub sync.
    """
    global _registry
    if _registry is None:
        # An unset variable and a blank one mean the same thing: take
        # the default. Blank is easy to produce by accident (`export
        # VOLKANO_XML=`, a declared-but-empty CI variable) and must not
        # be mistaken for a value.
        source = _os.environ.get('VOLKANO_XML', '').strip() or None
        # 'none' is spelled as a string because environment variables
        # are strings; the API itself keeps None|True|path|CDLL. Doing
        # the translation here keeps that magic word at the boundary.
        spec = _os.environ.get('VOLKANO_LIBRARY', '').strip() or True
        built = registry(source=source,
                         library=None if spec == 'none' else spec)
        # Publish before syncing: sync_stub forces every key, which is
        # slow, and any re-entrant volkano.<attr> during it must find
        # the singleton rather than start a second build.
        _registry = built
        from .stub import sync_stub
        sync_stub(built)
    return _registry


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
    base = {'registry', 'get_registry'}
    base.update(name for name in dir(get_registry())
                if not name.startswith('_'))
    return sorted(base)
