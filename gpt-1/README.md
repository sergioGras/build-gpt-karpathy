# GPT tutorial 1

Start with the guided [Python lessons](python/README.md), then follow their links to the corresponding C and Java code. The original tutorial notebook remains available as [`gpt_dev.ipynb`](gpt_dev.ipynb).

- Shared Tiny Shakespeare data: [`data/input.txt`](data/input.txt)
- Python notebooks: [`python/`](python/README.md)
- C project: [`c/`](c/README.md)
- Java Maven project: [`java/`](java/README.md)

## Run lessons by step

The Python notebooks use concept-sized cells. C and Java use conventional source files and entry points, so a lesson's numbered steps group naturally into functions and classes instead of pretending each source file is a notebook cell. In all three versions, the step numbers describe the same learning progression.

### Lesson 1: Read and inspect the training text

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Find the shared corpus | `dataset_path` in [`lesson_01_dataset.ipynb`](python/lesson_01_dataset.ipynb) | `main.c` checks the file | `TinyShakespeareDataset.findDefaultPath` searches common launch folders |
| 2 | Read the corpus | `Path.read_text` | `dataset_read` | `TinyShakespeareDataset.load` |
| 3 | Count and preview the text | `len` and slicing | `byte_count` and `fwrite` | `characterCount` and `preview` |

### Lesson 2: Character-level tokenization

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Build a stable character vocabulary | `sorted(set(text))` in [`lesson_02_character_tokenization.ipynb`](python/lesson_02_character_tokenization.ipynb) | `vocabulary_build` in `vocabulary.c` | `CharacterVocabulary.fromText` |
| 2 | Encode and decode a toy string | `stoi`, `itos`, `encode`, `decode` | `vocabulary_encode` and `vocabulary_decode` | `CharacterVocabulary.encode` and `.decode` |
| 3 | Encode and round-trip the full corpus | `encoded_text` list | `tokenization_main.c` | `CharacterTokenizationLesson` |

In VS Code, open this repository or the `gpt-1` folder and choose **Terminal → Run Task → Run tutorial lesson**. Select lesson `1` or `2`, then `c` or `java`, followed by the steps to show. `all` shows all three steps, `1-2` shows the first two, and `3` shows only the last step. The program performs earlier work needed for the selected step but hides its output.

The same workflow is available from a PowerShell terminal at the repository root:

```powershell
.\scripts\run-lesson.ps1 -Lesson 2 -Language c -Part 1-2
.\scripts\run-lesson.ps1 -Lesson 2 -Language java -Part 3
```

The script uses CMake and the Maven Wrapper. It finds this machine's MSYS2 CMake and JDK 25 installations, or uses tools already on `PATH`/`JAVA_HOME`. If Java 25 is installed elsewhere, pass `-JavaHome "C:\path\to\jdk-25"`.

Python iterates Unicode code points, Java's `char` represents UTF-16 code units, and this C lesson maps bytes. The shared corpus currently contains ASCII only, so all three versions assign the same vocabulary order and token IDs. Unicode-aware C tokenization is a later extension if the lessons need it.
