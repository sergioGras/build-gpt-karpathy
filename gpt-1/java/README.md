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
.\scripts\run-lesson.ps1 -Lesson 9 -Language java -Part 3-4
.\scripts\run-lesson.ps1 -Lesson 10 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 11 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 12 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 13 -Language java -Part all
```

Or choose **Terminal → Run Task → Run tutorial lesson** in VS Code. Select lesson 1 for corpus reading, 2 for character tokenization, 3 for next-token batches, 4 for the untrained bigram model, 5 for bigram training, 6 for attention math, 7 for attention-score scaling, 8 for LayerNorm, 9 for a causal attention head, 10 for multi-head attention, 11 for a Transformer block, 12 for token and position embeddings, or 13 for a tiny decoder language model. Lessons 1–5 and 7–8 have three steps; Lessons 6 and 9–13 have four and accept the extended ranges 1-4, 2-4, and 3-4. A selected step calculates prerequisites while hiding earlier output.

Lesson 2's entry point is `CharacterTokenizationLesson`. Lesson 3's `NextTokenBatchLesson` uses `int[][]` arrays for batches. Lesson 4 starts at `BigramLanguageModelLesson` and uses `BigramLanguageModel` to look up next-token logits, calculate stable cross-entropy, and sample an untrained continuation. Lesson 5 starts at `BigramTrainingLesson`; its `trainBatch` method accumulates the softmax-cross-entropy gradient for the selected lookup-table rows and performs an AdamW update. Lesson 6 starts at `SelfAttentionMathLesson` and uses `double[][]` arrays to show the causal mask, softmax weights, and query-key weighted aggregation. Lesson 7 starts at `AttentionScalingLesson`; its exact sign-vector enumeration makes the score-variance increase and the effect of scaling visible. Lesson 8 starts at `LayerNormLesson` and uses `double[][][]` arrays to calculate per-token statistics and apply the channel-wise scale/shift. Compare it with the [Python notebook](../python/lesson_08_layer_norm.ipynb) and [C implementation](../c/README.md).


Lesson 9 starts at `CausalAttentionHeadLesson`. It uses `double[][][]` arrays and fixed query/key/value weights to calculate scaled scores, apply the causal mask and softmax, then mix value vectors. Its perturbation check shows that changing a future token leaves earlier outputs unchanged. The weights are constants here, so this Java companion demonstrates the forward pass without training.

Lesson 10 starts at `MultiHeadAttentionLesson`. It runs two independent causal heads, concatenates their two-channel outputs into a batch/time/feature array, and applies a fixed projection back to model width. The Java path models evaluation mode, where dropout is inactive, and checks that the combined output remains causal.


Lesson 11 starts at TransformerBlockLesson. Its FeedForward helper class applies two linear transforms with a ReLU between them, independently at each batch/time position. The entry point shows pre-LayerNorm, attention and feed-forward residual additions, and checks that changing one token does not change another token's feed-forward result. Its attention branch uses the fixed Lesson 10 output for the shared input.


Lesson 12 starts at `TokenAndPositionEmbeddingLesson`. It uses two-dimensional arrays as embedding tables, adds token and position vectors at each batch/time location, and checks that repeated tokens at different positions receive different model inputs.

Lesson 13 starts at `DecoderLanguageModelLesson`. Its `forward` method connects the embedding tables to two pre-LayerNorm blocks and the vocabulary projection. The entry point calculates stable cross-entropy and performs deterministic autoregressive generation with a context window. Fixed weights make the Java output comparable to Python and C; this lesson does not train the model.

Lesson 14 trains the full decoder in the Python notebook with PyTorch autograd. For the Java-side training mechanics, revisit Lesson 5: `BigramTrainingLesson` and `BigramLanguageModel.trainBatch` derive the softmax-cross-entropy gradient and apply AdamW explicitly. Lesson 13 remains the Java forward-pass counterpart for the full decoder; this project does not yet implement its complete backward pass in Java.
