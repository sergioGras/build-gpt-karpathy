package learning.karpathy.gpt2;

import java.util.Arrays;

/** A small, library-free example of data-loader batching and next-token targets. */
public final class BatchingLesson {
    private static final int BATCH_SIZE = 2;
    private static final int CONTEXT_LENGTH = 4;
    private static final int WINDOW_SIZE = BATCH_SIZE * CONTEXT_LENGTH;
    private static final int[] TRAIN_TOKENS = {101, 102, 103, 104, 105, 106, 107, 108, 109};
    private static final int[] VALIDATION_TOKENS = {201, 202, 203, 204, 205, 206, 207, 208, 209};

    private BatchingLesson() {}

    public static void main(String[] args) {
        String part = args.length == 1 ? args[0] : "all";
        if (args.length > 1 || !part.matches("all|1|2|3")) {
            System.err.println("Usage: BatchingLesson [all|1|2|3]");
            System.exit(1);
        }

        if (part.equals("1") || part.equals("all")) {
            System.out.println("Step 1: keep training and validation token streams separate");
            printStream("Train", TRAIN_TOKENS);
            printStream("Validation", VALIDATION_TOKENS);
            System.out.println("The batch function receives one stream at a time, so a batch cannot cross the split.\n");
        }
        if (part.equals("2") || part.equals("all")) {
            System.out.println("Step 2: shift one stream by one token to make inputs and next-token targets");
            Batch batch = makeBatch(TRAIN_TOKENS, 0);
            printMatrix("x (inputs)", batch.inputs());
            printMatrix("y (targets)", batch.targets());
            System.out.println("Each y position is the next ID after its matching x position.\n");
        }
        if (part.equals("3") || part.equals("all")) {
            System.out.println("Step 3: advance the stream cursor and wrap when a full window will not fit");
            int cursor = WINDOW_SIZE;
            System.out.println("After one batch, cursor = " + cursor + ".");
            if (cursor + WINDOW_SIZE + 1 > TRAIN_TOKENS.length) {
                cursor = 0;
                System.out.println("The next B*T+1 window would overrun this toy stream, so restart at position 0.");
            }
            System.out.println("Next batch starts at token position " + cursor + ".");
            System.out.println("Karpathy's loader applies the same rule per shard and advances by B*T*num_processes.\n");
        }
    }

    private static void printStream(String label, int[] tokens) {
        System.out.println(label + " stream (" + tokens.length + " IDs): " + Arrays.toString(tokens));
    }

    private static Batch makeBatch(int[] tokens, int start) {
        if (start + WINDOW_SIZE + 1 > tokens.length) {
            throw new IllegalArgumentException("The stream is too short for one batch.");
        }
        int[][] inputs = new int[BATCH_SIZE][CONTEXT_LENGTH];
        int[][] targets = new int[BATCH_SIZE][CONTEXT_LENGTH];
        for (int row = 0; row < BATCH_SIZE; row++) {
            for (int column = 0; column < CONTEXT_LENGTH; column++) {
                int position = start + row * CONTEXT_LENGTH + column;
                inputs[row][column] = tokens[position];
                targets[row][column] = tokens[position + 1];
            }
        }
        return new Batch(inputs, targets);
    }

    private static void printMatrix(String label, int[][] values) {
        System.out.println(label + " =");
        for (int[] row : values) {
            System.out.println("  " + Arrays.toString(row));
        }
    }

    private record Batch(int[][] inputs, int[][] targets) {}
}
