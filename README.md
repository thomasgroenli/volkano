# volkano

Pure-Python Vulkan driver interface — ctypes only, lazy registry-driven, zero compiled code in the wheel.

```sh
pip install volkano
```

```python
from volkano import cbase, vk
inst_info = vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
# ...
```

## Layout

- `volkano.cbase` — C-ABI substrate (`Pointer`, `Array`, `Struct`, `Handle`, `Mixin`).
- `volkano.vk` — lazy registry of Vulkan types, structs, enums, and command thunks. Built once on first attribute access.

Designed as a stable substrate for higher-level libraries. See [kaldera](../kaldera) for the ergonomic compute layer on top.
