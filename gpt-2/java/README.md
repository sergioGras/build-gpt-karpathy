# Java companion

This is a separate Maven project targeting Java 25, matching the first tutorial's installed toolchain. It includes its own Maven Wrapper. `Main` inspects a configuration record; it does not allocate model weights or run inference/training.

From the repository root on Windows:

```powershell
.\scripts\run-gpt2.ps1 -Language java
.\scripts\run-gpt2.ps1 -Language java -Model gpt2
.\scripts\run-gpt2.ps1 -Language java -Action lesson
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Part 2
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 2 -Part 2
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 3 -Part 2
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 4 -Part 2
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 5 -Part 2
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 6
.\scripts\run-gpt2.ps1 -Language java -Action lesson -Lesson 7
```

The helper checks `JAVA_HOME`, `PATH`, and the existing local JDK installation folders. Use `-JavaHome 'C:\path\to\jdk'` for an explicit JDK 25 or later.

Or from this folder, with JDK 25 or later configured:

```bash
sh ./mvnw compile
java -cp target/classes learning.karpathy.gpt2.Main tiny
java -cp target/classes learning.karpathy.gpt2.Main gpt2
java -cp target/classes learning.karpathy.gpt2.ArchitectureLesson all
java -cp target/classes learning.karpathy.gpt2.ArchitectureLesson 2
```

On Windows use `.\mvnw.cmd compile`. VS Code's **Run** above `Main.main` also works; this entry point has no working-directory-dependent data access.

Lesson 1 is [ArchitectureLesson.java](src/main/java/learning/karpathy/gpt2/ArchitectureLesson.java). Its steps compare the first tutorial's tiny model with GPT-2, explain head shapes, and count GPT-2 parameters. Select `all`, `1`, `2`, `3`, `1-2`, `2-3`, or `1-3` with `-Part`.

Lesson 2 is [TokenizerLesson.java](src/main/java/learning/karpathy/gpt2/TokenizerLesson.java). It uses JTokkit 1.1.0's `r50k_base` encoding, which shares GPT-2's ordinary token vocabulary and IDs. The sample adds the end-of-text ID explicitly to show that document boundaries belong to preprocessing. Maven resolves the dependency; the repository-root helper supplies it on the Java runtime classpath.

Lesson 3 is [BatchingLesson.java](src/main/java/learning/karpathy/gpt2/BatchingLesson.java). It uses the same small integer fixture as Python and C to show the train/validation boundary, the one-token input/target shift, and cursor wrap without introducing extra libraries.

Lesson 4 is [EmbeddingsLesson.java](src/main/java/learning/karpathy/gpt2/EmbeddingsLesson.java). It looks up the same toy token and position vectors as Python and C, adds them into an initial residual stream, and prints each dimension so the table indexing and batch reuse stay visible.

Lesson 5 is [AttentionLesson.java](src/main/java/learning/karpathy/gpt2/AttentionLesson.java). It computes the same Q/K/V projections, scaled scores, causal mask, softmax weights, and context vectors as Python and C. Small methods separate each matrix operation, and a final check confirms that a future token cannot affect an earlier output.

Lesson 6 is [BlockLesson.java](src/main/java/learning/karpathy/gpt2/BlockLesson.java). It combines LayerNorm, causal attention and its residual, then a 4C position-wise MLP with tanh-approximate GELU and its residual. Small methods name the matrix operations while keeping the arithmetic visible; the fixed toy weights match Python and C. Select `all`, `1`, `2`, or `3` with `-Part`.

Lesson 7 is [GptForwardLesson.java](src/main/java/learning/karpathy/gpt2/GptForwardLesson.java). It runs token and position lookup, one decoder block, final LayerNorm, and the tied output projection that yields four logits per position. Java arrays omit the leading singleton batch dimension for readability. The last-position argmax is a toy choice from untrained weights. Select `all`, `1`, `2`, or `3` with `-Part`.

Read [Main.java](src/main/java/learning/karpathy/gpt2/Main.java) and [ModelConfig.java](src/main/java/learning/karpathy/gpt2/ModelConfig.java) for the original scaffold presets. Java does not read experiment JSON yet. We will investigate libraries for actual pretrained GPT-2 inference before adopting a larger dependency.
