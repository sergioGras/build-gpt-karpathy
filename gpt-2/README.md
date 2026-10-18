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

The project now has runnable scaffolds, an orientation notebook, and experiment configuration/provenance support. The GPT-2 model, tokenization, training, and inference-library integrations will be implemented through the lessons.

From the repository root on Windows:

```powershell
.\scripts\setup-gpt2-python.ps1
.\scripts\run-gpt2.ps1 -Language python
.\scripts\run-gpt2.ps1 -Language c
.\scripts\run-gpt2.ps1 -Language java
```

Open `gpt-2/` as a VS Code folder for its own interpreter and CMake settings. Use **Terminal → Run Task → GPT-2: Inspect scaffold** to select a language. Open [the orientation notebook](python/notebooks/00_project_tour.ipynb) and select **Karpathy GPT-2 (project .venv)** as the kernel. If you keep the whole repository open, select this new kernel explicitly; the repository's default interpreter still belongs to GPT-1.

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

For a playful dataset or architecture idea, begin with the [experiment guide](experiments/README.md). The same settings and run records will support both tutorial baselines and your own variations. C and Java currently inspect matching dimension presets; the Python CLI owns experiment JSON until later lessons add language-specific integrations.

## References

- [Karpathy's GPT-2 reproduction video](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Upstream `build-nanogpt` source and README](build-nanogpt/README.md)
- [GPT-2 model card](https://huggingface.co/openai-community/gpt2)
- [Project remote-compute plan](../docs/remote-compute-plan.md)
