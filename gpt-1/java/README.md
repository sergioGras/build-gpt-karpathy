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
```

Or choose **Terminal → Run Task → Run tutorial lesson** in VS Code. Select lesson `1` for corpus reading or lesson `2` for character tokenization. Each lesson accepts `all`, `1`, `2`, `3`, `1-2`, and `1-3`; one selected step runs its prerequisites without showing their output.

Lesson 2's entry point is `CharacterTokenizationLesson`. It uses `CharacterVocabulary` to assign IDs in sorted character order, encode strings into `int[]`, and decode them with a `StringBuilder`. The vocabulary class keeps the mapping separate from the lesson's print-and-experiment flow, following normal Java project structure while keeping the operations visible. Compare it with the [Python notebook](../python/lesson_02_character_tokenization.ipynb) and [C implementation](../c/README.md).
