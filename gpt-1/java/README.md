# GPT tutorial 1 in Java

This Maven project is the Java companion to the guided [Python lessons](../python/README.md). It reads the same corpus and uses a conventional `src/main/java` layout. `Main` and `TinyShakespeareDataset` implement the first lesson; the [lesson map](../README.md#run-lessons-by-step) links each step to Python and C.

## Build and run

From this directory, with `JAVA_HOME` pointing to JDK 25, use the included Maven Wrapper:

```powershell
.\mvnw.cmd compile
java -cp target\classes learning.karpathy.gpt1.Main
```

The project targets Java 25 LTS. If your terminal's `java -version` reports an older release, use the shared runner below; it selects JDK 25 for Maven and the program. The Maven Wrapper downloads Maven on first use.

## Run a lesson step

From the repository root:

```powershell
.\scripts\run-lesson.ps1 -Lesson 2 -Language java -Part 1-2
.\scripts\run-lesson.ps1 -Lesson 2 -Language java -Part 3
.\scripts\run-lesson.ps1 -Lesson 5 -Language java -Part all
```

Or choose **Terminal → Run Task → Run tutorial lesson** in VS Code. Select lesson `1` for corpus reading, `2` for character tokenization, `3` for next-token batches, `4` for the untrained bigram model, or `5` for bigram training. Each lesson accepts `all`, `1`, `2`, `3`, `1-2`, and `1-3`; one selected step runs its prerequisites without showing their output.

Lesson 2's entry point is `CharacterTokenizationLesson`. Lesson 3's `NextTokenBatchLesson` uses `int[][]` arrays for batches. Lesson 4 starts at `BigramLanguageModelLesson` and uses `BigramLanguageModel` to look up next-token logits, calculate stable cross-entropy, and sample an untrained continuation. Lesson 5 starts at `BigramTrainingLesson`; its `trainBatch` method accumulates the softmax-cross-entropy gradient for the selected lookup-table rows and performs an AdamW update. This makes the operations hidden behind PyTorch autograd and the optimizer visible in ordinary Java arrays. Compare it with the [Python notebook](../python/lesson_05_train_bigram.ipynb) and [C implementation](../c/README.md).
