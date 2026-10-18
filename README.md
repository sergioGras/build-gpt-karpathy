# build-gpt-karpathy

A guided learning project following Andrej Karpathy's tutorials to understand language models and Transformers through equivalent implementations in Python, C, and Java.

The goal is readable, executable code with explanations that other learners can follow. Performance optimization and parallel execution are not priorities.

## Learning principles

- Follow the tutorial's progression, starting with the Python notebook and translating each meaningful step into C and Java.
- Explain what each step does, why it matters, and how data and tensor shapes change.
- Use libraries where they reduce boilerplate. Explain the underlying primitive operations with short comments, pseudocode, or small examples when useful; rebuilding every library is not required.
- Prefer clear names, small functions, and conventional project structures over clever abstractions or premature optimization.
- Compare equivalent behavior across languages using shared data and small examples. Floating-point results and random sequences may differ between libraries.
- Include exercises, observations, and references alongside the code. Explain reasoning and abstractions rather than commenting every obvious statement.
- Develop the tutorial incrementally with the learner, rather than generating the complete implementation ahead of the lessons.

## Languages and development environment

- **Python:** retain the notebook format, using the tutorial implementation as the reference.
- **Java:** use Maven and a `pom.xml`, with conventional source and test directories.
- **C:** use a small, conventional source/header structure and a documented build process.
- **Editor:** aim to inspect and work on all three implementations in Visual Studio Code.

Try straightforward local setup first. If compiler, JDK, Python environment, or dependency setup becomes a distraction, consider Docker or a remote development environment. Tooling should support the learning experience.

Specific C and Java numerical libraries, versions, and build tools beyond Maven will be selected as their requirements become clear.

## Lesson format

Each lesson should connect the relevant notebook cells to:

1. The concept and its role in the model.
2. The Python behavior, including inputs, outputs, and shapes.
3. Readable C and Java equivalents, including the abstractions their libraries provide.
4. Instructions to run the examples and check their behavior.
5. A small experiment or exercise and relevant conceptual or library documentation.

Group related cells when that makes the concept easier to understand. Keep navigation between equivalent implementations explicit as the project grows.

## Training lab: intended direction

The eventual tutorial should offer a reproducible path to longer training experiments on a rented GPU, with SSH access and scripted execution where supported by the provider.

The intended experience is to configure provider credentials outside the repository, select an explicit resource and spending limit, provision and prepare the environment, run training, inspect loss curves and progress, and retrieve logs and checkpoints. Resource shutdown and cleanup should be part of that workflow, with provider-specific billing behavior documented.

Providing a token alone does not define a spending limit or authorize an arbitrary GPU rental. The provider, resource, and budget must be established before paid provisioning.

GPU execution in each language depends on the chosen numerical library and its GPU support; CPU examples remain useful for understanding and checking individual operations.

This lab is a future capability, not an implemented feature. Provider selection, dashboards, checkpoint conventions, and automation will be developed when the lessons need them.

The [remote-compute plan](docs/remote-compute-plan.md) adapts a RunPod workflow playbook to this tutorial and maps the next architecture lessons to a bounded first GPU experiment.

## Tutorial 1 — complete

The first tutorial follows Karpathy's *Let's build GPT: from scratch, in code, spelled out*, progressing from Tiny Shakespeare and character encoding to a bigram model, causal self-attention, and a decoder-only Transformer.

The reference notebook is [gpt-1/gpt_dev.ipynb](gpt-1/gpt_dev.ipynb).

To run the C or Java version a step at a time in VS Code, use **Terminal → Run Task → Run tutorial lesson**. The [lesson guide](gpt-1/README.md#run-lessons-by-step) maps each step to the notebook and both projects.

The guided path has fourteen lessons. Python covers training the complete small decoder. C and Java cover the decoder forward pass and generation, with explicit gradient and optimizer calculations in the bigram lesson; full Transformer backpropagation is outside this tutorial's C and Java implementations.

## Tutorial 2 — GPT-2 reproduction

The next track follows Karpathy's [*Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU). Its upstream code checkout is in [gpt-2/build-nanogpt](gpt-2/build-nanogpt/README.md). The guided track starts with the [project tour and Lesson 1](gpt-2/README.md), then follows with [GPT-2 tokenization](gpt-2/python/notebooks/lesson_02_gpt2_tokenization.ipynb), [token streams and batches](gpt-2/python/notebooks/lesson_03_token_streams_and_batches.ipynb), and [token and position embeddings](gpt-2/python/notebooks/lesson_04_token_and_position_embeddings.ipynb) before continuing into attention, training, evaluation, and experiments.
