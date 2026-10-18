# GPT tutorial 1 in C

This is the C companion project for the guided [Python lessons](../python/README.md). It uses standard C17 and a small CMake project with conventional `src/` and `include/` folders. The [lesson map](../README.md#run-lessons-by-step) connects each numbered step to its Python and Java counterpart.

## Build and run

From this directory, configure and build with CMake, then run the dataset lesson:

```powershell
cmake --preset default
cmake --build --preset default --target gpt1_c
build/gpt1_c.exe
```

The preset selects Ninja and puts generated files in the ignored `build/` directory.

## Run a lesson step

From the repository root, use the [shared runner](../README.md#run-lessons-by-step):

```powershell
.\scripts\run-lesson.ps1 -Lesson 2 -Language c -Part 1-2
.\scripts\run-lesson.ps1 -Lesson 4 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 5 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 6 -Language c -Part 4
```

Or use **Terminal → Run Task → Run tutorial lesson** in VS Code. Choose lesson `1` for corpus reading, `2` for character tokenization, `3` for next-token batches, `4` for the untrained bigram model, `5` for bigram training, or `6` for the attention math. Lessons 1–5 have three steps; Lesson 6 has four and accepts `4`, `1-4`, `2-4`, and `3-4` as well. A selected step calculates any prerequisites while hiding their output.

Lesson 1 starts at `src/main.c` and delegates file reading to `dataset_read` in `src/dataset.c`. Lesson 2 starts at `src/tokenization_main.c`; `src/vocabulary.c` builds a byte-to-ID lookup table, encodes text into integer IDs, and decodes IDs back into text. Lesson 3 starts at `src/batches_main.c` and stores batches in flat arrays. Lesson 4 starts at `src/bigram_main.c`; `src/bigram_model.c` implements table lookup, stable cross-entropy, and sampling. Lesson 5 starts at `src/bigram_training_main.c` and calls `bigram_model_train_batch` to calculate the softmax-cross-entropy gradient and perform an AdamW update. Lesson 6 starts at `src/attention_math_main.c`; its two-dimensional arrays expose causal averaging, matrix multiplication, masked softmax, and query-key weighted values. See the [Python Lesson 6 notebook](../python/lesson_06_self_attention_math.ipynb) and [Java project](../java/README.md) for the parallel implementations.
