# C companion

This starts with a C17 executable, a configuration header, and an implementation file. The `tiny` and `gpt2` presets match Python's `ModelConfig` dimensions and Java's `ModelConfig` record. Printing them allocates no weights and performs no inference or training.

From the repository root on Windows:

```powershell
.\scripts\run-gpt2.ps1 -Language c
.\scripts\run-gpt2.ps1 -Language c -Model gpt2
```

Or from this folder, with CMake, Ninja, and a C compiler on `PATH`:

```bash
cmake --preset default
cmake --build --preset default
./build/gpt2_c tiny
./build/gpt2_c gpt2
```

On Windows the executable is `build/gpt2_c.exe`. CMake can also be configured with another generator using a separate build directory.

Read [main.c](src/main.c), [model_config.h](include/model_config.h), and [model_config.c](src/model_config.c). C does not read experiment JSON yet. We will select numerical/tokenizer libraries after checking a small working example; the target is eventually real pretrained GPT-2 inference, with small primitive examples where useful for understanding it.
