package learning.karpathy.gpt1;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/** Runnable Java companion to lesson 4: build and sample an untrained bigram model. */
public final class BigramLanguageModelLesson {
    private static final int BLOCK_SIZE = 8;
    private static final int BATCH_SIZE = 4;

    private BigramLanguageModelLesson() {
    }

    public static void main(String[] args) throws IOException {
        int[] steps = parseSteps(args);
        TinyShakespeareDataset dataset = TinyShakespeareDataset.load(
                TinyShakespeareDataset.findDefaultPath());
        String text = dataset.text();
        CharacterVocabulary vocabulary = CharacterVocabulary.fromText(text);
        int[] tokenIds = vocabulary.encode(text);
        int trainLength = tokenIds.length * 9 / 10;
        int[] trainData = Arrays.copyOfRange(tokenIds, 0, trainLength);
        int[][] inputs = new int[BATCH_SIZE][BLOCK_SIZE];
        int[][] targets = new int[BATCH_SIZE][BLOCK_SIZE];
        sampleBatch(trainData, new Random(1337), inputs, targets);

        BigramLanguageModel model = new BigramLanguageModel(vocabulary.size(), 1337);
        float[][][] logits = model.forward(inputs);

        if (steps[0] <= 1) {
            int firstInputId = inputs[0][0];
            System.out.println("Step 1 - Look up next-character scores");
            System.out.printf("Vocabulary size: %d; parameter table: [%d, %d] (%d scores)%n",
                    vocabulary.size(), vocabulary.size(), vocabulary.size(),
                    vocabulary.size() * vocabulary.size());
            System.out.printf("Input batch: [%d, %d]; logits: [%d, %d, %d]%n",
                    BATCH_SIZE, BLOCK_SIZE, BATCH_SIZE, BLOCK_SIZE, vocabulary.size());
            System.out.printf("First input token ID %d selects table row; first 8 logits: %s%n",
                    firstInputId, preview(logits[0][0]));
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Compare logits with next-character targets");
            System.out.printf("Targets: %d positions (%d batch rows x %d time steps)%n",
                    BATCH_SIZE * BLOCK_SIZE, BATCH_SIZE, BLOCK_SIZE);
            System.out.printf("Mean cross-entropy loss before training: %.4f%n",
                    model.crossEntropy(logits, targets));
            System.out.println("For each position: logsumexp(logits) - target_logit, then average.");
        }

        if (steps[1] >= 3) {
            int[] generatedIds = model.generate(0, 100);
            System.out.println("Step 3 - Sample characters autoregressively (untrained model)");
            System.out.printf("Start token ID: 0; generated text:%n%s%n",
                    vocabulary.decode(generatedIds));
            System.out.println("This model uses only the immediately previous character as context.");
        }
    }

    private static String preview(float[] values) {
        int previewCount = Math.min(values.length, 8);
        StringBuilder result = new StringBuilder("[");
        for (int index = 0; index < previewCount; index++) {
            if (index > 0) result.append(", ");
            result.append(String.format("%.3f", values[index]));
        }
        if (previewCount < values.length) result.append(", ...");
        return result.append(']').toString();
    }

    private static void sampleBatch(int[] source, Random random, int[][] inputs, int[][] targets) {
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            int start = random.nextInt(source.length - BLOCK_SIZE);
            System.arraycopy(source, start, inputs[batch], 0, BLOCK_SIZE);
            System.arraycopy(source, start + 1, targets[batch], 0, BLOCK_SIZE);
        }
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException(
                    "Usage: BigramLanguageModelLesson [all|1|2|3|1-2|1-3]");
        }
        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "1-2" -> new int[] { 1, 2 };
            case "all", "1-3" -> new int[] { 1, 3 };
            default -> throw new IllegalArgumentException(
                    "Usage: BigramLanguageModelLesson [all|1|2|3|1-2|1-3]");
        };
    }
}
