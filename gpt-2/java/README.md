# Java companion

This is a separate Maven project targeting Java 25, matching the first tutorial's installed toolchain. It includes its own Maven Wrapper. `Main` inspects a configuration record; it does not allocate model weights or run inference/training.

From the repository root on Windows:

```powershell
.\scripts\run-gpt2.ps1 -Language java
.\scripts\run-gpt2.ps1 -Language java -Model gpt2
```

The helper checks `JAVA_HOME`, `PATH`, and the existing local JDK installation folders. Use `-JavaHome 'C:\path\to\jdk'` for an explicit JDK 25 or later.

Or from this folder, with JDK 25 or later configured:

```bash
sh ./mvnw compile
java -cp target/classes learning.karpathy.gpt2.Main tiny
java -cp target/classes learning.karpathy.gpt2.Main gpt2
```

On Windows use `.\mvnw.cmd compile`. VS Code's **Run** above `Main.main` also works; this entry point has no working-directory-dependent data access.

Read [Main.java](src/main/java/learning/karpathy/gpt2/Main.java) and [ModelConfig.java](src/main/java/learning/karpathy/gpt2/ModelConfig.java). The `tiny` and `gpt2` presets match the Python/C dimension checks. Java does not read experiment JSON yet. We will investigate libraries for actual pretrained GPT-2 inference before adopting a larger dependency.
