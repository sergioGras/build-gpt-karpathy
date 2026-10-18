# C companion

The C17 project includes a small executable, a configuration header, and implementation files. The `tiny` and `gpt2` presets match Python's `ModelConfig` dimensions and Java's `ModelConfig` record. Printing them allocates no weights and performs no inference or training.

From the repository root on Windows:

```powershell
.\scripts\run-gpt2.ps1 -Language c
.\scripts\run-gpt2.ps1 -Language c -Model gpt2
.\scripts\run-gpt2.ps1 -Language c -Action lesson
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Part 2
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 2 -Part 2
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 3 -Part 2
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 4 -Part 2
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 5 -Part 2
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 6
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 7
```

Or from this folder, with CMake, Ninja, and a C compiler on `PATH`:

```bash
cmake --preset default
cmake --build --preset default
./build/gpt2_c tiny
./build/gpt2_c gpt2
./build/gpt2_architecture_lesson all
./build/gpt2_architecture_lesson 2
./build/gpt2_attention_lesson all
```

On Windows the executable is `build/gpt2_c.exe`. CMake can also be configured with another generator using a separate build directory.

Lesson 1 is [architecture_lesson.c](src/architecture_lesson.c). Its steps compare the first tutorial's tiny model with GPT-2, explain head shapes, and count GPT-2 parameters. [model_config.c](src/model_config.c) has the shared GPT-2 parameter formula. Select `all`, `1`, `2`, `3`, `1-2`, `2-3`, or `1-3` with `-Part`.

Lesson 2 is [tokenizer_lesson.c](src/tokenizer_lesson.c). It uses `tiktoken-c` 0.9.1 to compare GPT-2 token IDs, UTF-8 examples, and the end-of-text document marker. The lesson runner downloads the Windows x86_64 release when needed and checks its SHA-256; CMake skips this optional target until the library is present. The example keeps the tokenizer opaque and frees returned buffers with `tiktoken_free`, which demonstrates the C API's ownership boundary.

Lesson 3 is [batching_lesson.c](src/batching_lesson.c). It keeps train and validation ID streams separate, shifts a `B*T+1` window into input/target matrices, and shows the cursor wrapping at the end. The short integer fixture matches the Java and Python batching example; it avoids adding tokenizer setup to this lesson.

Lesson 4 is [embeddings_lesson.c](src/embeddings_lesson.c). A tiny shared table makes the token lookup, position lookup, and addition visible in plain C arrays. It corresponds to `wte(idx) + wpe(pos)` in the Python reference; the loops show how the position vectors are reused across the batch.

Lesson 5 is [attention_lesson.c](src/attention_lesson.c). It uses the first residual-stream example from Lesson 4 to show Q/K/V projections, scaled dot products, the causal mask, stable softmax, and value mixing. C arrays and loops keep each operation visible; the final check changes the last input and verifies that position 0 cannot see it.

Lesson 6 is [block_lesson.c](src/block_lesson.c). It adds LayerNorm before causal attention, the first residual update, a per-token MLP that expands from C to 4C with tanh-approximate GELU, and the second residual update. The same small input and fixed MLP weights are used in Python and Java. Select `all`, `1`, `2`, or `3` with `-Part` to focus on a stage.

Lesson 7 is [gpt_forward_lesson.c](src/gpt_forward_lesson.c). It follows token IDs through token and position lookup, one decoder block, final LayerNorm, and a weight-tied projection to four vocabulary logits. The leading singleton batch dimension is omitted from the C arrays to keep the loops direct; printed shapes include it. The final-position argmax is only a toy choice, not generated language. Select `all`, `1`, `2`, or `3` with `-Part`.

Read [main.c](src/main.c), [model_config.h](include/model_config.h), and [model_config.c](src/model_config.c) for the original scaffold presets. C does not read experiment JSON yet. The tokenizer package is an early prototype; numerical and inference libraries will still be checked with small working examples before they become part of the course.
