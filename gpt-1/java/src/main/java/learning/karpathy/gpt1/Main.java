package learning.karpathy.gpt1;

import java.io.IOException;
import java.nio.file.Path;

public final class Main {
    private Main() {
    }

    public static void main(String[] args) throws IOException {
        Path datasetPath = Path.of("..", "data", "input.txt");
        TinyShakespeareDataset dataset = TinyShakespeareDataset.load(datasetPath);

        System.out.printf("Dataset file: %s%n", dataset.source());
        System.out.printf("Dataset length in Java characters: %d%n", dataset.characterCount());
        System.out.println("First 1000 characters:");
        System.out.println(dataset.preview(1000));
    }
}
