"""Lazy, XML-driven Vulkan bindings for Python.

The public entrypoint is the :mod:`volkano.vk` submodule::

    from volkano import vk
    instance = vk.VkInstance()
    vk.vkCreateInstance(...)

See :mod:`volkano.vk` for configuration options and
:mod:`volkano.stub` for generating a ``.pyi`` for IDE autocomplete.
"""
