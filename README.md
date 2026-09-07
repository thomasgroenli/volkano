# volkano

Pure-Python Vulkan driver interface — ctypes only, lazy registry-driven, zero compiled code in the wheel.

```sh
pip install volkano
```

```python
import volkano as vk
inst_info = vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
# ...
```

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
vk = volkano.registry(library=True)                  # Khronos main
vk = volkano.registry(V1_4_359_URL, library=True)    # or any vk.xml
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

Remote XML is cached under `%LOCALAPPDATA%\volkano\xml` (or
`~/.cache/volkano/xml`) and **never revalidated**, so imports do no
network I/O whatever the source. A URI carries no evidence of whether
what it names can move — a tag URL and a branch URL are the same shape
— so rather than guess, volkano refetches only when told to:

```sh
python -m volkano update
```

That refetches the configured source and rewrites the `.pyi` stub from
it.

## Notes

- **The loader is opened lazily** — on the first *command you call*, not at
  build and not at lookup. The entire registry, commands included, resolves
  and introspects on a machine with no Vulkan driver installed; only
  dispatch needs one.
- **`__init__.pyi` is generated** by `python -m volkano update`, and stamped
  with the hash and URI of the XML it came from. Importing volkano never
  writes it — that would force all 8 000-odd entries and write a megabyte
  into the installed package behind an `import` statement. Import only
  compares the stamp and logs at INFO when it has gone stale. A missing or
  stale stub costs autocomplete and nothing else.
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

Designed as a stable substrate for higher-level libraries. See [kaldera](../kaldera) for the ergonomic compute layer on top.
