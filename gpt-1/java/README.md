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
.\scripts\run-lesson.ps1 -Lesson 6 -Language java -Part 4
.\scripts\run-lesson.ps1 -Lesson 7 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 8 -Language java -Part all
```

Or choose **Terminal → Run Task → Run tutorial lesson** in VS Code. Select lesson `1` for corpus reading, `2` for character tokenization, `3` for next-token batches, `4` for the untrained bigram model, `5` for bigram training, `6` for the attention math, `7` for attention-score scaling, or `8` for LayerNorm. Lessons 1–5 and 7–8 have three steps; Lesson 6 has four and accepts `4`, `1-4`, `2-4`, and `3-4` as well. A selected step calculates any prerequisites while hiding their output.

Lesson 2's entry point is `CharacterTokenizationLesson`. Lesson 3's `NextTokenBatchLesson` uses `int[][]` arrays for batches. Lesson 4 starts at `BigramLanguageModelLesson` and uses `BigramLanguageModel` to look up next-token logits, calculate stable cross-entropy, and sample an untrained continuation. Lesson 5 starts at `BigramTrainingLesson`; its `trainBatch` method accumulates the softmax-cross-entropy gradient for the selected lookup-table rows and performs an AdamW update. Lesson 6 starts at `SelfAttentionMathLesson` and uses `double[][]` arrays to show the causal mask, softmax weights, and query-key weighted aggregation. Lesson 7 starts at `AttentionScalingLesson`; its exact sign-vector enumeration makes the score-variance increase and the effect of scaling visible. Lesson 8 starts at `LayerNormLesson` and uses `double[][][]` arrays to calculate per-token statistics and apply the channel-wise scale/shift. Compare it with the [Python notebook](../python/lesson_08_layer_norm.ipynb) and [C implementation](../c/README.md).
