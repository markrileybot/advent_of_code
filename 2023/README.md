Setup
=====

1. Install llvm `7`.  Yes you have to have 7.  I built from source:
    ```shell
    git clone https://github.com/llvm/llvm-project.git
    git checkout -b release/llvmorg-7.1.0 llvmorg-7.1.0
    cmake -S llvm -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/llvm/7.1.0 -DCMAKE_BUILD_TYPE=Release
    ninja -C build
    sudo ninja -C build install
    ```
2. Install cuda
3. Set `CUDA_ROOT` and `LLVM_CONFIG` in `.cargo/config.toml`:
    ```toml
    [env]
    CUDA_ROOT = "/usr/lib/cuda"
    LLVM_CONFIG = "/opt/llvm/7.1.0/bin/llvm-config"
    ```