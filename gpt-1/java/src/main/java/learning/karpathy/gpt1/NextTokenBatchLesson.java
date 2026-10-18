package learning.karpathy.gpt1;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/** Runnable companion to lesson 3: prepare shifted next-token batches. */
public final class NextTokenBatchLesson {
    private static final int BLOCK_SIZE = 8;
    private static final int BATCH_SIZE = 4;

    private NextTokenBatchLesson() {
    }

    public static void main(String[] args) throws IOException {
        int[] steps = parseSteps(args);
        TinyShakespeareDataset dataset = TinyShakespeareDataset.load(TinyShakespeareDataset.findDefaultPath());
        String text = dataset.text();
        CharacterVocabulary vocabulary = CharacterVocabulary.fromText(text);
        int[] tokenIds = vocabulary.encode(text);

        int splitIndex = (int) (0.9 * tokenIds.length);
        int[] trainData = Arrays.copyOfRange(tokenIds, 0, splitIndex);
        int[] validationData = Arrays.copyOfRange(tokenIds, splitIndex, tokenIds.length);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Split tokens into training and validation sets");
            System.out.printf("All tokens: %d%n", tokenIds.length);
            System.out.printf("Training tokens (first 90%%): %d%n", trainData.length);
            System.out.printf("Validation tokens (last 10%%): %d%n", validationData.length);
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            requireEnoughTokens(trainData.length);
            int[] window = Arrays.copyOfRange(trainData, 0, BLOCK_SIZE + 1);
            System.out.println("Step 2 - Make context and next-token examples");
            System.out.printf("Block size: %d; the displayed window has %d tokens.%n",
                    BLOCK_SIZE, BLOCK_SIZE + 1);
            System.out.printf("Window token IDs: %s%n", Arrays.toString(window));

            for (int time = 0; time < BLOCK_SIZE; time++) {
                int[] contextIds = Arrays.copyOfRange(window, 0, time + 1);
                int targetId = window[time + 1];
                System.out.printf("Context IDs %s (%s) -> target ID %d (%s)%n",
                        Arrays.toString(contextIds), vocabulary.decode(contextIds),
                        targetId, vocabulary.decode(new int[] { targetId }));
            }
        }

        if (steps[1] >= 3) {
            requireEnoughTokens(trainData.length);
            Random random = new Random(1337);
            int[][] trainInputs = new int[BATCH_SIZE][BLOCK_SIZE];
            int[][] trainTargets = new int[BATCH_SIZE][BLOCK_SIZE];
            int[][] validationInputs = new int[BATCH_SIZE][BLOCK_SIZE];
            int[][] validationTargets = new int[BATCH_SIZE][BLOCK_SIZE];
            sampleBatch(trainData, random, trainInputs, trainTargets);
            sampleBatch(validationData, random, validationInputs, validationTargets);

            System.out.println("Step 3 - Sample a mini-batch");
            System.out.printf("x inputs, shape [%d, %d]: %s%n",
                    BATCH_SIZE, BLOCK_SIZE, Arrays.deepToString(trainInputs));
            System.out.printf("y next-token targets, shape [%d, %d]: %s%n",
                    BATCH_SIZE, BLOCK_SIZE, Arrays.deepToString(trainTargets));
            System.out.printf("Validation x, shape [%d, %d]: %s%n",
                    BATCH_SIZE, BLOCK_SIZE, Arrays.deepToString(validationInputs));
            System.out.printf("Validation y, shape [%d, %d]: %s%n",
                    BATCH_SIZE, BLOCK_SIZE, Arrays.deepToString(validationTargets));
        }
    }

    private static void sampleBatch(int[] source, Random random, int[][] inputs, int[][] targets) {
        requireEnoughTokens(source.length);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            int start = random.nextInt(source.length - BLOCK_SIZE);
            System.arraycopy(source, start, inputs[batch], 0, BLOCK_SIZE);
            System.arraycopy(source, start + 1, targets[batch], 0, BLOCK_SIZE);
        }
    }

    private static void requireEnoughTokens(int tokenCount) {
        if (tokenCount <= BLOCK_SIZE) {
            throw new IllegalArgumentException(
                    "Training split must contain more than " + BLOCK_SIZE + " tokens.");
        }
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException("Usage: NextTokenBatchLesson [all|1|2|3|1-2|1-3]");
        }

        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "1-2" -> new int[] { 1, 2 };
            case "all", "1-3" -> new int[] { 1, 3 };
            default -> throw new IllegalArgumentException(
                    "Usage: NextTokenBatchLesson [all|1|2|3|1-2|1-3]");
        };
    }
}
