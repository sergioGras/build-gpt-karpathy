package learning.karpathy.gpt1;

/** Shows how token identity and sequence position become model-width vectors. */
public final class TokenAndPositionEmbeddingLesson {
    private static final int BATCH_SIZE = 2;
    private static final int TIME_STEPS = 4;
    private static final int CHANNELS = 4;
    private static final int VOCAB_SIZE = 5;

    private static final String[] VOCABULARY = { "a", "b", "c", "d", "e" };
    private static final int[][] TOKEN_IDS = {
            { 0, 1, 0, 2 },
            { 1, 2, 3, 0 }
    };

    private TokenAndPositionEmbeddingLesson() {
    }

    public static void main(String[] args) {
        int[] steps = parseSteps(args);
        double[][] tokenTable = new double[VOCAB_SIZE][CHANNELS];
        double[][] positionTable = new double[TIME_STEPS][CHANNELS];
        initializeEmbeddingTables(tokenTable, positionTable);
        EmbeddingResult embeddings = lookupEmbeddings(tokenTable, positionTable);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Start with token IDs");
            printIds();
            System.out.println("Each integer is an index into a vocabulary; the number itself has no ordered meaning.");
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Look up a learned vector for each token");
            printTokenVectors("Token embeddings", embeddings.tokenVectors());
            System.out.println("The same token ID always selects the same row from the token embedding table.");
        }

        if (steps[0] <= 3 && steps[1] >= 3) {
            System.out.println("Step 3 - Look up a learned vector for each position");
            System.out.printf("Position embedding table, shape [time, channels] = [%d, %d]:%n",
                    TIME_STEPS, CHANNELS);
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.printf("  position %d [", time);
                for (int channel = 0; channel < CHANNELS; channel++) {
                    System.out.printf("%s%.2f", channel == 0 ? "" : ", ",
                            positionTable[time][channel]);
                }
                System.out.println("]");
            }
            System.out.println("The position table is added to every sequence in the batch.");
        }

        if (steps[1] >= 4) {
            double sameTokenDifference = maximumDifference(
                    embeddings.tokenVectors()[0][0], embeddings.tokenVectors()[0][2]);
            double positionedDifference = maximumDifference(
                    embeddings.modelInput()[0][0], embeddings.modelInput()[0][2]);

            System.out.println("Step 4 - Add token and position information");
            printTokenVectors(
                    "Transformer input = token embedding + position embedding",
                    embeddings.modelInput());
            System.out.printf(
                    "Same token at positions 0 and 2: token-vector difference = %.6f; combined-vector difference = %.6f%n",
                    sameTokenDifference, positionedDifference);
            if (sameTokenDifference > 1e-12 || positionedDifference < 1e-12) {
                throw new IllegalStateException("Embedding position check failed.");
            }
            System.out.println(
                    "Check passed: the token vector is reused, while its position changes the combined representation.");
        }
    }

    private static void initializeEmbeddingTables(
            double[][] tokenTable, double[][] positionTable) {
        for (int token = 0; token < VOCAB_SIZE; token++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                int pattern = (token * 5 + channel * 3 + 1) % 11 - 5;
                tokenTable[token][channel] = pattern * 0.1;
            }
        }
        for (int position = 0; position < TIME_STEPS; position++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                int pattern = (position * 7 + channel * 2 + 2) % 9 - 4;
                positionTable[position][channel] = pattern * 0.1;
            }
        }
    }

    private static EmbeddingResult lookupEmbeddings(
            double[][] tokenTable, double[][] positionTable) {
        double[][][] tokenVectors = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
        double[][][] modelInput = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                int tokenId = TOKEN_IDS[batch][time];
                for (int channel = 0; channel < CHANNELS; channel++) {
                    tokenVectors[batch][time][channel] = tokenTable[tokenId][channel];
                    modelInput[batch][time][channel] =
                            tokenVectors[batch][time][channel] + positionTable[time][channel];
                }
            }
        }
        return new EmbeddingResult(tokenVectors, modelInput);
    }

    private static void printIds() {
        System.out.println("Vocabulary: a=0, b=1, c=2, d=3, e=4");
        System.out.printf("Token IDs, shape [batch, time] = [%d, %d]:%n",
                BATCH_SIZE, TIME_STEPS);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            System.out.printf("  batch %d [", batch);
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.printf("%s%d", time == 0 ? "" : ", ", TOKEN_IDS[batch][time]);
            }
            System.out.println("]");
        }
    }

    private static void printTokenVectors(String label, double[][][] values) {
        System.out.printf("%s, shape [batch, time, channels] = [%d, %d, %d]:%n",
                label, BATCH_SIZE, TIME_STEPS, CHANNELS);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            System.out.printf("  batch %d%n", batch);
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.printf("    token %s at position %d [",
                        VOCABULARY[TOKEN_IDS[batch][time]], time);
                for (int channel = 0; channel < CHANNELS; channel++) {
                    System.out.printf("%s%.2f", channel == 0 ? "" : ", ",
                            values[batch][time][channel]);
                }
                System.out.println("]");
            }
        }
    }

    private static double maximumDifference(double[] left, double[] right) {
        double maximum = 0.0;
        for (int channel = 0; channel < CHANNELS; channel++) {
            maximum = Math.max(maximum, Math.abs(left[channel] - right[channel]));
        }
        return maximum;
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException(usage());
        }
        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "4" -> new int[] { 4, 4 };
            case "1-2" -> new int[] { 1, 2 };
            case "1-3" -> new int[] { 1, 3 };
            case "2-4" -> new int[] { 2, 4 };
            case "3-4" -> new int[] { 3, 4 };
            case "all", "1-4" -> new int[] { 1, 4 };
            default -> throw new IllegalArgumentException(usage());
        };
    }

    private static String usage() {
        return "Usage: TokenAndPositionEmbeddingLesson [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]";
    }

    private record EmbeddingResult(
            double[][][] tokenVectors,
            double[][][] modelInput) {
    }
}
