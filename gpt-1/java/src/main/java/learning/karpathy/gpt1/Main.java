package learning.karpathy.gpt1;

import java.io.IOException;

public final class Main {
    private Main() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length > 1) {
            throw new IllegalArgumentException("Usage: Main [all|1|2|3|1-2|1-3]");
        }

        String part = args.length == 0 ? "all" : args[0];
        int firstStep;
        int lastStep;
        switch (part) {
            case "1" -> { firstStep = 1; lastStep = 1; }
            case "2" -> { firstStep = 2; lastStep = 2; }
            case "3" -> { firstStep = 3; lastStep = 3; }
            case "1-2" -> { firstStep = 1; lastStep = 2; }
            case "all", "1-3" -> { firstStep = 1; lastStep = 3; }
            default -> throw new IllegalArgumentException("Usage: Main [all|1|2|3|1-2|1-3]");
        }

        var datasetPath = TinyShakespeareDataset.findDefaultPath();

        if (firstStep <= 1) {
            System.out.println("Step 1 - Find the shared corpus");
            System.out.printf("Dataset file: %s%n", datasetPath);
        }

        if (lastStep == 1) {
            return;
        }

        // Later steps still load the text when their earlier output is hidden.
        TinyShakespeareDataset dataset = TinyShakespeareDataset.load(datasetPath);

        if (firstStep <= 2) {
            System.out.println("Step 2 - Read the corpus");
            System.out.println("Corpus loaded into memory.");
        }

        if (lastStep >= 3) {
            System.out.println("Step 3 - Inspect the examples");
            System.out.printf("Dataset length in Java characters: %d%n", dataset.characterCount());
            System.out.println("First 1000 characters:");
            System.out.println(dataset.preview(1000));
        }
    }

}
