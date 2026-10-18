# Tutorial 2: Reproduce GPT-2 (124M)

This track follows Andrej Karpathy's [*Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU). It continues from the small, character-level GPT in `gpt-1` to GPT-2's byte-pair tokenizer, larger decoder, web-text training data, and training workflow.

## Keep the source and the lessons distinct

[`build-nanogpt/`](build-nanogpt/README.md) is the upstream source checkout that accompanies the video. It is here as a reference for the original code and its progression. The guided explanations and language exercises for this project belong alongside that checkout, rather than being edits to its files.

The reference remains an independent checkout and is ignored by the outer repository. Its revision and fresh-clone instructions are recorded in [docs/reference-source.md](docs/reference-source.md).

We will keep the same learning approach as Tutorial 1: understand the Python behavior first, make tensor shapes and data flow visible, then build readable C and Java counterparts where they clarify the concepts. Libraries are welcome when they remove routine boilerplate; comments and small examples can expose the underlying operations. We will organize each language in a conventional project structure as the lessons take shape.

## Planned progression

We will begin by surveying the reference code and tracing the data and model configuration before writing a training run. Then we can build lessons around GPT-2 tokenization and data preparation, the GPT-2 decoder and its parameter shapes, training and validation, checkpointing, and text generation. Lesson boundaries will follow the concepts rather than copying the video or source file line for line.

The first checks and reduced experiments should run locally where practical. Reproducing large-scale GPT-2 training is a separate compute experiment: before renting a GPU, we will define the exact configuration, runtime and spend limit, outputs, and cleanup plan. The project-level [remote-compute plan](../docs/remote-compute-plan.md) describes that workflow.

## Start here

The project has runnable scaffolds, an orientation notebook, and three prepared lessons. [Lesson 1](python/notebooks/lesson_01_gpt1_to_gpt2.ipynb) compares the first tutorial's tiny model with GPT-2. [Lesson 2](python/notebooks/lesson_02_gpt2_tokenization.ipynb) explores byte-level BPE, token IDs, and document boundaries. [Lesson 3](python/notebooks/lesson_03_token_streams_and_batches.ipynb) turns separate train and validation token streams into shifted next-token batches.

From the repository root on Windows:

```powershell
.\scripts\setup-gpt2-python.ps1
.\scripts\run-gpt2.ps1 -Language python
.\scripts\run-gpt2.ps1 -Language c
.\scripts\run-gpt2.ps1 -Language java
.\scripts\run-gpt2.ps1 -Language c -Action lesson
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Part 2
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 2 -Part 1
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 2
.\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 3
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 3 -Part 2
```

Open `gpt-2/` as a VS Code folder for its own interpreter and CMake settings. **Terminal → Run Task → GPT-2: Run Lesson 1/2/3** runs the selected C or Java companion. Open [the orientation notebook](python/notebooks/00_project_tour.ipynb), then Lessons [1](python/notebooks/lesson_01_gpt1_to_gpt2.ipynb), [2](python/notebooks/lesson_02_gpt2_tokenization.ipynb), and [3](python/notebooks/lesson_03_token_streams_and_batches.ipynb), and select **Karpathy GPT-2 (project .venv)** as the kernel. Python setup installs `tiktoken`; the first C Lesson 2 run downloads a pinned Windows library release and verifies its SHA-256. Python and C may also fetch and cache GPT-2's vocabulary data on first use. Java resolves JTokkit through Maven. These lessons need no model weights or GPU. If you keep the whole repository open, select the GPT-2 kernel explicitly; the repository's default interpreter still belongs to GPT-1.

## Project map

| Location | Purpose |
| --- | --- |
| [python/](python/README.md) | Learning notebooks and the importable `gpt2_lab` package |
| [c/](c/README.md) | C17/CMake companion |
| [java/](java/README.md) | Java 25/Maven companion |
| [experiments/](experiments/README.md) | Configurations, custom datasets, and run preparation |
| [data/](data/README.md) | Small public samples and ignored local/cache locations |
| [docs/learning-plan.md](docs/learning-plan.md) | Milestones, language goals, and completion criteria |
| [docs/summary transcript.txt](docs/summary%20transcript.txt) | Supplied transcript summary |
| `runs/` | Generated settings/provenance and, later, training outputs; ignored by Git |
| `../scripts/` | Setup/run helpers, with remote execution to be added when needed |

| Lesson 1 step | Python | C | Java |
| --- | --- | --- | --- |
| 1. Compare dimensions | Notebook, section 1 | [architecture_lesson.c](c/src/architecture_lesson.c) | [ArchitectureLesson.java](java/src/main/java/learning/karpathy/gpt2/ArchitectureLesson.java) |
| 2. Calculate head width and attention shapes | Notebook, section 2 | Same file, step 2 | Same file, step 2 |
| 3. Estimate GPT-2 parameter count | Notebook, section 3 | [parameter formula](c/src/model_config.c) | [parameter formula](java/src/main/java/learning/karpathy/gpt2/ModelConfig.java) |

| Lesson 2 step | Python | C | Java |
| --- | --- | --- | --- |
| 1. Encode text and inspect IDs | Notebook, sections 1–2 | [tokenizer_lesson.c](c/src/tokenizer_lesson.c), step 1 | [TokenizerLesson.java](java/src/main/java/learning/karpathy/gpt2/TokenizerLesson.java), step 1 |
| 2. Compare spaces and UTF-8 text | Notebook, section 2 | Same file, step 2 | Same file, step 2 |
| 3. Add an end-of-text boundary | Notebook, section 3 | Same file, step 3 | Same file, step 3 |

| Lesson 3 step | Python | C | Java |
| --- | --- | --- | --- |
| 1. Keep train and validation streams separate | Notebook, section 1 | [batching_lesson.c](c/src/batching_lesson.c), step 1 | [BatchingLesson.java](java/src/main/java/learning/karpathy/gpt2/BatchingLesson.java), step 1 |
| 2. Shift token windows into inputs and targets | Notebook, section 2 | Same file, step 2 | Same file, step 2 |
| 3. Advance and wrap the loader cursor | Notebook, section 3 | Same file, step 3 | Same file, step 3 |

For a playful dataset or architecture idea, begin with the [experiment guide](experiments/README.md). The same settings and run records will support both tutorial baselines and your own variations. C and Java now use GPT-2-compatible tokenizer libraries for matching ordinary-text token IDs; the Python CLI owns experiment JSON until later lessons add language-specific integrations.

## References

- [Karpathy's GPT-2 reproduction video](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Upstream `build-nanogpt` source and README](build-nanogpt/README.md)
- [GPT-2 model card](https://huggingface.co/openai-community/gpt2)
- [OpenAI tiktoken](https://github.com/openai/tiktoken)
- [JTokkit](https://github.com/knuddelsgmbh/jtokkit)
- [tiktoken-c](https://github.com/kojix2/tiktoken-c)
- [Project remote-compute plan](../docs/remote-compute-plan.md)
