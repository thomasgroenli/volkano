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
vk = volkano.registry('1.4.359', library=True)
```

The first is the second applied to two environment variables. There is
only one implementation.

## Configuring the module

| Variable | Meaning |
|---|---|
| `VOLKANO_XML` | unset / `main` — the Khronos main branch (default) |
| | `tag:v1.4.359` — that tag, pinned and cached forever |
| | `branch:sc_main` — that branch head |
| | any other value — a path to a local `vk.xml` |
| `VOLKANO_LIBRARY` | unset — autodetect the Vulkan loader |
| | a path — use that loader |
| | `none` — attach no loader at all (codegen, CI) |

Tag names are passed through **verbatim**, so every published tag is
reachable — including the older families, `tag:v1.0.33-core` and
`tag:v1.0-core+wsi-20160216`. Deriving the tag from a version number
would be inference about Khronos's naming history rather than
construction, and it is wrong for 89 of the 382 tags: plain `vX.Y.Z`
only starts at v1.1.70, and `v1.0.33` has never existed.

Arbitrary URLs are not accepted: once cached, a URL is a local file with
extra steps. Download it yourself and pass the path.

Both variables are read on first use, not at import, and only by the
module facade — `registry()` takes arguments and nothing else.

XML is cached under `%LOCALAPPDATA%\volkano\xml` (or `~/.cache/volkano/xml`)
and never revalidated, so imports do no network I/O. To move `main`
forward:

```sh
python -m volkano.stub --refresh
```

## Notes

- **The loader is opened lazily** — on the first *command* you resolve, not
  at build. Constants, enums and structs work on a machine with no Vulkan
  driver installed.
- **`__init__.pyi` is generated**, stamped with the hash of the XML it came
  from, and rewritten whenever the two disagree. Only the module facade
  does this; `registry()` never touches it.
- **Forcing is thread-safe** — racing threads share one value per key, so
  ctypes class identity stays consistent.

## Layout

- `volkano` — the package itself is a lazy registry of Vulkan types, structs, enums, and command thunks. Built once on first attribute access.
- `volkano.cbase` — C-ABI substrate (`Pointer`, `Array`, `Struct`, `Handle`, `Mixin`).
- `volkano.lazy` — the memoising thunk kernel the registry is built on.
- `volkano.xml_source` — *where* the XML comes from: source vocabulary, fetch, cache, provenance. Knows Khronos's repository, nothing about Vulkan's schema.
- `volkano.xml_parser` — *what* the XML means: the vk.xml → thunk-graph walker. Knows the schema, nothing about caching.
- `volkano.vulkan_stdlib` — the factories the thunk graph calls, plus `VkRegistry`.
- `volkano.stub` — `.pyi` generation and the `python -m volkano.stub` maintenance CLI.

Designed as a stable substrate for higher-level libraries. See [kaldera](../kaldera) for the ergonomic compute layer on top.
