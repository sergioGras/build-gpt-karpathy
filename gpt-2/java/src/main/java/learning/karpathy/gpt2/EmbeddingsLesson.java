package learning.karpathy.gpt2;

/** A tiny numeric example of token lookup plus learned positional embeddings. */
public final class EmbeddingsLesson {
    private static final int BATCH_SIZE = 2;
    private static final int SEQUENCE_LENGTH = 3;
    private static final int EMBEDDING_SIZE = 2;
    private static final int[][] TOKEN_IDS = {{1, 2, 3}, {3, 2, 1}};
    private static final double[][] TOKEN_EMBEDDING_TABLE = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}
    };
    private static final double[][] POSITION_EMBEDDING_TABLE = {
        {0.1, 0.0}, {0.0, 0.2}, {0.3, 0.3}
    };

    private EmbeddingsLesson() {}

    public static void main(String[] args) {
        String part = args.length == 1 ? args[0] : "all";
        if (args.length > 1 || !part.matches("all|1|2|3")) {
            System.err.println("Usage: EmbeddingsLesson [all|1|2|3]");
            System.exit(1);
        }

        double[][][] tokenVectors = tokenVectors();
        if (part.equals("1") || part.equals("all")) {
            System.out.println("Step 1: token IDs and positions are separate inputs");
            System.out.println("Token IDs, shape (B, T) = (2, 3):");
            for (int[] row : TOKEN_IDS) {
                System.out.printf("  [%d, %d, %d]%n", row[0], row[1], row[2]);
            }
            System.out.println("Position IDs, shape (T): [0, 1, 2]");
            System.out.println("GPT-2 maps each token ID to a 768-number vector and supports up to 1,024 positions.\n");
        }
        if (part.equals("2") || part.equals("all")) {
            System.out.println("Step 2: look up one learned vector for each token ID");
            printVectors("Token vectors", tokenVectors);
            System.out.println("The table lookup changes integer IDs into continuous representations.\n");
        }
        if (part.equals("3") || part.equals("all")) {
            System.out.println("Step 3: add the position vector at each sequence location");
            double[][][] inputVectors = new double[BATCH_SIZE][SEQUENCE_LENGTH][EMBEDDING_SIZE];
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int position = 0; position < SEQUENCE_LENGTH; position++) {
                    for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                        inputVectors[batch][position][channel] = tokenVectors[batch][position][channel]
                                + POSITION_EMBEDDING_TABLE[position][channel];
                    }
                }
            }
            printVectors("Token + position vectors (initial residual stream)", inputVectors);
            System.out.println("The position table is reused across rows of the batch; broadcasting is explicit in these loops.\n");
        }
    }

    private static double[][][] tokenVectors() {
        double[][][] vectors = new double[BATCH_SIZE][SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int position = 0; position < SEQUENCE_LENGTH; position++) {
                int tokenId = TOKEN_IDS[batch][position];
                for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                    vectors[batch][position][channel] = TOKEN_EMBEDDING_TABLE[tokenId][channel];
                }
            }
        }
        return vectors;
    }

    private static void printVectors(String label, double[][][] values) {
        System.out.println(label + ", shape (B, T, C) = (2, 3, 2):");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            System.out.print("  batch " + batch + ": ");
            for (int position = 0; position < SEQUENCE_LENGTH; position++) {
                System.out.printf("[%.1f, %.1f]", values[batch][position][0],
                        values[batch][position][1]);
                System.out.print(position + 1 == SEQUENCE_LENGTH ? System.lineSeparator() : " ");
            }
        }
    }
}
