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
.\scripts\run-lesson.ps1 -Lesson 2 -Language c -Part 3
```

Or use **Terminal → Run Task → Run tutorial lesson** in VS Code. Choose lesson `1` for corpus reading or lesson `2` for character tokenization. Each lesson accepts `all`, `1`, `2`, `3`, `1-2`, and `1-3`; one selected step runs its prerequisites without showing their output.

Lesson 1 starts at `src/main.c` and delegates file reading to `dataset_read` in `src/dataset.c`. Lesson 2 starts at `src/tokenization_main.c`; `src/vocabulary.c` builds a byte-to-ID lookup table, encodes text into integer IDs, and decodes IDs back into text. The table is deliberately simple to expose the mapping. This C version maps bytes; the current shared corpus is ASCII, so each byte is one character and the IDs match Python and Java. See the [Python lesson](../python/lesson_02_character_tokenization.ipynb) and [Java project](../java/README.md) for the parallel implementations.
