"""Host-side per-operation costs of driving Vulkan through volkano.

Each ``section_*`` function returns a :class:`~bench._harness.Section`.
``collect()`` runs them all. Import and call individual sections to drill
into one group.
"""
from __future__ import annotations

import ctypes

import volkano as vk
from volkano import cbase
from volkano.vulkan_stdlib import CommandSignature

from ._harness import Section, bench
from . import _native

# Pulled out of the loops so construction cost isn't double-counted.
_STYPE_WDS = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
_STYPE_SUBMIT = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO
_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7


def section_ffi() -> Section:
    """FFI floor and volkano's wrapper tax over a *cheap* native call."""
    s = Section("FFI floor & CommandSignature wrapper tax")

    s.add("python no-op (loop floor)", bench(lambda: None))

    dll, sym, restype, raw0 = _native.zero_arg()
    s.add(f"raw ctypes call ({sym}, 0 args)", bench(raw0))

    sig0 = CommandSignature(sym, restype, params=(), attrs={})
    sig0.bind(dll)
    wrapped0 = bench(lambda: sig0())
    s.add(f"  via CommandSignature wrapper", wrapped0)

    psym, rawp, parg = _native.ptr_arg()
    s.add(f"raw ctypes call ({psym}, 1 ptr arg)", bench(lambda: rawp(parg)))
    sigp = CommandSignature(psym, cbase.int32, params=(), attrs={})
    sigp._fn = rawp  # bypass friendly() on the param; we only time the call path
    s.add(f"  via CommandSignature wrapper", bench(lambda: sigp(parg)))

    s.footnote(
        "wrapper tax is the gap between each '  via ...' row and the raw row "
        "above it (~0.1-0.2 us); this is the per-vkCmd* overhead."
    )
    return s


def section_objects() -> Section:
    """Cost of constructing the cbase/struct objects a dispatch builds,
    and the lazy-registry lookup tax of reaching the type via ``vk.<Name>``
    instead of a hoisted local alias."""
    s = Section("Object construction & the vk.<Name> lookup tax")

    s.add("cbase.uint32() + .ref", bench(_scalar_ref))
    s.add("VkWriteDescriptorSet()  [empty, hoisted type]", bench(_WDS))
    s.add("VkWriteDescriptorSet + 5 sets + .ref  [hoisted]", bench(_build_wds_hoisted))
    s.add("VkWriteDescriptorSet + 5 sets + .ref  [via vk.X]", bench(_build_wds_registry))
    s.footnote(
        "the ~1.4 us gap between [hoisted] and [via vk.X] is volkano's lazy "
        "registry __getattr__, NOT struct construction. A driver must alias "
        "Vk types at import; touching vk.X in the record loop is the tax."
    )
    return s


def section_composite() -> Section:
    """Composite operations: what one kaldera Stream.run / submit does."""
    s = Section("Composite host operations (per dispatch)")

    s.add(
        "6-binding descriptor writes (apply)  [via vk.X]",
        bench(_build_6_binding_writes_registry, iters=25_000),
        note="<- naive: pays lookup x12",
    )
    s.add(
        "6-binding descriptor writes (apply)  [hoisted]",
        bench(_build_6_binding_writes_hoisted, iters=25_000),
        note="<- types aliased at import",
    )
    s.add("build VkSubmitInfo + sets + .ref  [hoisted]", bench(_build_submit, iters=50_000))
    s.footnote(
        "hoisting the types is most of the win here. What remains (descriptor "
        "writes + ~4 vkCmd* calls) a descriptor-set cache removes; "
        "command-buffer replay removes the recording entirely on repeats."
    )
    return s


def section_reference() -> Section:
    """One-time native calls — NOT per dispatch. Here for scale only."""
    s = Section("Reference: expensive loader calls (once at device build)")
    ver = cbase.uint32(0)
    r = ver.ref
    # Low iter count: this call is hundreds of us; it rescans ICDs natively.
    s.add(
        "vkEnumerateInstanceVersion (native loader work)",
        bench(lambda: vk.vkEnumerateInstanceVersion(r), iters=2_000, warmup=200),
        note="not volkano overhead; happens once",
    )
    v = int(ver)
    s.footnote(f"loader reports Vulkan {v >> 22}.{(v >> 12) & 0x3ff}.{v & 0xfff}")
    return s


# --- bench bodies (module-level so closures don't add lookup cost) ---------
# Types hoisted out of the lazy registry once, the way a real driver would
# alias them at import. The `_registry` variants below deliberately reach
# through `vk.<Name>` each call to measure the lookup tax.
_DBI = vk.VkDescriptorBufferInfo
_WDS = vk.VkWriteDescriptorSet
_uint32 = cbase.uint32


def _scalar_ref():
    return _uint32(0).ref


def _build_wds_hoisted():
    w = _WDS()
    w.sType = _STYPE_WDS
    w.dstBinding = 0
    w.dstArrayElement = 0
    w.descriptorCount = 1
    w.descriptorType = _DESCRIPTOR_TYPE_STORAGE_BUFFER
    return w.ref


def _build_wds_registry():
    w = vk.VkWriteDescriptorSet()
    w.sType = _STYPE_WDS
    w.dstBinding = 0
    w.dstArrayElement = 0
    w.descriptorCount = 1
    w.descriptorType = _DESCRIPTOR_TYPE_STORAGE_BUFFER
    return w.ref


def _build_6_binding_writes_hoisted():
    writes = []
    for b in range(6):
        d = _DBI()
        d.buffer = 0
        d.offset = 0
        d.range = 4096
        w = _WDS()
        w.sType = _STYPE_WDS
        w.dstBinding = b
        w.descriptorCount = 1
        w.descriptorType = _DESCRIPTOR_TYPE_STORAGE_BUFFER
        w.pBufferInfo = d.ptr
        writes.append(w)
    return writes


def _build_6_binding_writes_registry():
    writes = []
    for b in range(6):
        d = vk.VkDescriptorBufferInfo()
        d.buffer = 0
        d.offset = 0
        d.range = 4096
        w = vk.VkWriteDescriptorSet()
        w.sType = _STYPE_WDS
        w.dstBinding = b
        w.descriptorCount = 1
        w.descriptorType = _DESCRIPTOR_TYPE_STORAGE_BUFFER
        w.pBufferInfo = d.ptr
        writes.append(w)
    return writes


def _build_submit():
    si = vk.VkSubmitInfo()
    si.sType = _STYPE_SUBMIT
    si.commandBufferCount = 1
    si.waitSemaphoreCount = 0
    si.signalSemaphoreCount = 0
    return si.ref


def collect() -> list[Section]:
    return [
        section_ffi(),
        section_objects(),
        section_composite(),
        section_reference(),
    ]
