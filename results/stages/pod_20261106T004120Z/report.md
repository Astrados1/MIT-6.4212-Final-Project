# Stage report: pod

Written 20261106T004121Z. Result: **OK**.

| check | status | detail |
|---|---|---|
| driver:gpu0 | PASS | NVIDIA L40S, driver 550.144.03 |
| rt-cores:gpu0 | PASS | NVIDIA L40S, compute capability 8.9, 46068 MiB |
| cuda | PASS | created a CUDA context and allocated 1 MiB on 1 GPU(s) |
| vulkan | PASS | Vulkan sees the NVIDIA GPU |
| graphics-libs | PASS | libEGL, libGL, libXt and libGLU present |
| os | PASS | ubuntu 22.04 |
| disk | PASS | 107 GB free at /workspace |
| env:OMNI_KIT_ACCEPT_EULA | WARN | not set: Isaac Sim stops at its license prompt without it (set it yourself once you accept) |
| env:NVIDIA_DRIVER_CAPABILITIES | PASS | compute,display,graphics,utility,video |
