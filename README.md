# volkano

Pure-Python Vulkan driver interface for **compute** — ctypes only, lazy
registry-driven, zero compiled code in the wheel.

```sh
pip install volkano
```

```python
import volkano as vk

app = vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           pApplicationName=b'demo',
                           apiVersion=vk.VK_API_VERSION_1_0)
info = vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               pApplicationInfo=app.ptr)

instance = vk.VkInstance()
result = vk.vkCreateInstance(info.ref, None, instance.ref)
assert result is vk.VK_SUCCESS

count = vk.uint32()
vk.vkEnumeratePhysicalDevices(instance, count.ref, None)
print(f'{count.value} physical device(s)')

vk.vkDestroyInstance(instance, None)
```

Four things in there are the whole calling convention:

- **`.ptr` for a struct field, `.ref` for an argument.** `x.ptr` is a real
  typed pointer, which is what a `pApplicationInfo` slot holds; `x.ref` is
  `byref(x)`, which is what a command parameter wants. Using `.ref` in a
  field raises `TypeError: expected Pointer[VkApplicationInfo] instance` —
  the one mistake everybody makes first.
- **Commands return the enum member**, so `result is vk.VK_SUCCESS` holds
  and a failure prints as `VkResult.VK_ERROR_INITIALIZATION_FAILED` rather
  than as `-3`.
- **Enumerants are their group's member**: `vk.VK_SUCCESS` *is*
  `vk.VkResult.VK_SUCCESS`, not a separate `int` that compares equal.
- **Scalars are named as cbase names them** — `vk.uint32`, `vk.float32` —
  and carry `.ref`, `.ptr`, `.value` and `.size`. vk.xml's C spellings
  (`uint32_t`) are translated to these, so there is exactly one name per
  type.

## Scope of the first release

Compute is the target, and it works end to end: instances, physical
device enumeration and properties, logical devices and queues, buffers
and memory requirements, descriptor sets, pipelines, command pools and
submission all resolve and dispatch.

Two areas do not, and both for the same reason — `vk.xml` *declares*
their types without defining them, leaving `HINSTANCE`, `Display`,
`wl_display`, `HANDLE` and the `StdVideo*` family to platform and video
headers volkano does not read:

- **Window-system integration.** `VkWin32SurfaceCreateInfoKHR`,
  `VkXlibSurfaceCreateInfoKHR`, `VkWaylandSurfaceCreateInfoKHR` and
  `vkCreateWin32SurfaceKHR` are unreachable, so there is no way to make a
  surface to present to. The swapchain types themselves are fine — it is
  only the surface underneath them that is missing.
- **Win32 external memory, semaphore and fence interop**, which needs
  `HANDLE`.
- **Video encode and decode**, which needs the separate `vulkan_video`
  registry.

That is 109 of 9 074 entries. Touching one raises `AttributeError` naming
the type that is missing rather than failing obscurely, and `hasattr`
reports them absent — so a capability probe works, and nothing else in
the registry is affected.

## Two ways to use it

**The module.** `import volkano` gives you the whole Vulkan surface as
module attributes, backed by one registry built on first access:

```python
import volkano as vk
vk.vkCreateInstance(...)
```

**An object.** `volkano.registry()` builds an independent registry. It
reads no environment, keeps no global state, and never writes to your
source tree — so several can coexist:

```python
import volkano

TAG = ('https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs'
       '/refs/tags/v1.4.359/xml/vk.xml')

vk = volkano.registry(library=True)         # Khronos main branch
vk = volkano.registry(TAG, library=True)    # or any vk.xml, by URL
```

The first is the second applied to two environment variables. There is
only one implementation.

## Configuring the module

| Variable | Meaning |
|---|---|
| `VOLKANO_XML` | unset — the Khronos main branch (default) |
| | an `http(s)://` URL — that `vk.xml`, fetched once and cached |
| | a `file://` URI or a path — that file, read in place |
| `VOLKANO_LIBRARY` | unset — autodetect the Vulkan loader |
| | a path — use that loader |
| | `none` — attach no loader at all (codegen, CI) |

A source is just a **URI**. There is no vocabulary to learn and no
version syntax to get right: pinning a release means naming the URL of
that release's `vk.xml`, which is the identifier GitHub already gives
you.

```sh
VOLKANO_XML=https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/refs/tags/v1.4.359/xml/vk.xml
```

That generalises where a version number wouldn't. Khronos has used
three tag-naming families — `v1.4.359`, `v1.0.33-core`,
`v1.0-core+wsi-20160216` — and plain `vX.Y.Z` only starts at v1.1.70,
so any scheme that rebuilds a tag from a version is wrong for 89 of the
382 published tags. It also moved `vk.xml` from `src/spec/` to `xml/`
between v1.1.70 and v1.2.131. A URL sidesteps both, and reaches a fork,
a mirror or an internal artifact server for free — volkano knows one
Khronos URL (the default) and nothing else about the repository.

`volkano.use('https://…/vk.xml')` does the same thing from code, and
takes precedence over the environment. It has to run before the first
`volkano.<name>` access — that access is what builds the registry — and
raises if it doesn't, rather than being silently ignored. An
environment variable remains the only way to configure a volkano that
something else imported first.

Both variables are read on first use, not at import, and only by the
module facade — `registry()` takes arguments and nothing else.

Remote XML is fetched into `%LOCALAPPDATA%\volkano\xml` (or
`~/.cache/volkano/xml`) the first time it is needed and **never
revalidated** after that, so no import past the first does network
I/O. A URI carries no evidence of whether
what it names can move — a tag URL and a branch URL are the same shape
— so rather than guess, volkano refetches only when told to:

```sh
python -m volkano update
```

That refetches the configured source and rewrites the `.pyi` stub from
it. Everything else is first-use: the XML is taken from the cache if it
is there and fetched into it if it isn't, and the stub is brought level
with whatever was built.

## Notes

- **The loader is opened lazily** — on the first *command you call*, not at
  build and not at lookup. The entire registry, commands included, resolves
  and introspects on a machine with no Vulkan driver installed; only
  dispatch needs one.
- **`__init__.pyi` is written on first use**, and stamped with the hash and
  URI of the XML it came from. Later imports compare that one line and stop
  there — the walk that regenerating costs (~1 s, all 8 000-odd entries) is
  paid only when the stamp disagrees: a fresh install, a changed source, or
  an improved generator. An unwritable target is skipped rather than
  retried, and every failure is swallowed; a missing or stale stub costs
  autocomplete and nothing else.
- **`volkano.registry()` never writes it.** Only the module facade syncs the
  stub, so a registry you construct yourself — one per source, several at
  once — leaves your source tree alone.
- **Forcing is thread-safe** — racing threads share one value per key, so
  ctypes class identity stays consistent.

## Layout

- `volkano` — the package itself is a lazy registry of Vulkan types, structs, enums, and command thunks. Built once on first attribute access.
- `volkano.cbase` — C-ABI substrate (`Pointer`, `Array`, `Struct`, `Handle`, `Mixin`).
- `volkano.lazy` — the memoising thunk kernel the registry is built on.
- `volkano.xml_source` — *where* the XML comes from: URI resolution, fetch, cache, provenance. Knows how to fetch bytes, nothing about Vulkan's schema.
- `volkano.xml_parser` — *what* the XML means: the vk.xml → thunk-graph walker. Knows the schema, nothing about caching.
- `volkano.vulkan_stdlib` — the factories the thunk graph calls, plus `VkRegistry`.
- `volkano.stub` — `.pyi` generation and the `python -m volkano update` maintenance CLI.

Designed as a stable substrate for higher-level libraries.
