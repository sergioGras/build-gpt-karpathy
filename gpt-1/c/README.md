# GPT tutorial 1 in C

This is the C companion project for the [guided Python lesson](../python/lesson_01_dataset.ipynb). The starter reads the same corpus at `../data/input.txt`; follow the notebook's dataset section to see how `main.c`, `dataset.h`, and `dataset.c` divide the work.

## Build and run

From this directory, configure and build with CMake, then run the resulting executable:

```powershell
cmake --preset default
cmake --build --preset default
build/gpt1_c.exe
```

The preset selects Ninja and puts generated files in the ignored `build/` directory. This project uses standard C17 and keeps its first entry point in `src/main.c`.
