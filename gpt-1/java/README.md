# GPT tutorial 1 in Java

This Maven project is the Java companion to the [guided Python lesson](../python/lesson_01_dataset.ipynb). It reads the same corpus at `../data/input.txt` and uses a conventional `src/main/java` layout. The notebook explains why `Main` delegates file loading to `TinyShakespeareDataset`.

## Build and run

From this directory, use the included Maven Wrapper so the project selects its Maven version consistently:

```powershell
.\mvnw.cmd compile
java -cp target\classes learning.karpathy.gpt1.Main
```

The project targets Java 25 LTS. The JDK is installed under the user's tools directory; new terminals should use it through `JAVA_HOME`. The Maven Wrapper downloads Maven 3.10.0 on first use.
