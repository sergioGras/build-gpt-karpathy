package learning.karpathy.gpt2;

import java.util.Locale;

/** A tiny, explicit example of causal scaled dot-product self-attention. */
public final class AttentionLesson {
    private static final int SEQUENCE_LENGTH = 3;
    private static final int EMBEDDING_SIZE = 2;

    // This is the first sequence's residual stream from Lesson 4.
    private static final double[][] INPUT = {
        {1.1, 0.0},
        {0.0, 1.2},
        {1.3, 1.3}
    };

    // Identity matrices keep this first example focused on attention weights.
    private static final double[][] QUERY_WEIGHTS = {{1.0, 0.0}, {0.0, 1.0}};
    private static final double[][] KEY_WEIGHTS = {{1.0, 0.0}, {0.0, 1.0}};
    private static final double[][] VALUE_WEIGHTS = {{1.0, 0.0}, {0.0, 1.0}};

    private AttentionLesson() {}

    public static void main(String[] args) {
        String part = args.length == 1 ? args[0] : "all";
        if (args.length > 1 || !part.matches("all|1|2|3")) {
            System.err.println("Usage: AttentionLesson [all|1|2|3]");
            System.exit(1);
        }

        AttentionResult result = runAttention(INPUT);
        if (part.equals("1") || part.equals("all")) {
            System.out.println("Step 1: project each input vector into query, key, and value");
            printVectorMatrix("Input residual stream X (T, C)", INPUT);
            printVectorMatrix("Query Q = X W_Q", result.query());
            printVectorMatrix("Key K = X W_K", result.key());
            printVectorMatrix("Value V = X W_V", result.value());
            System.out.println("The toy projections are identity matrices; GPT-2 learns separate projection weights.\n");
        }
        if (part.equals("2") || part.equals("all")) {
            System.out.println("Step 2: compare queries with keys, scale, then mask future positions");
            printScoreMatrix("Scaled scores Q K^T / sqrt(2)", result.scores());
            printScoreMatrix("After causal mask (future scores become -inf)", result.maskedScores());
            System.out.println("Softmax turns each allowed score row into attention weights:\n");
            printScoreMatrix("Attention weights", result.weights());
            System.out.println("Each row sums to 1; masked future positions have weight 0.\n");
        }
        if (part.equals("3") || part.equals("all")) {
            System.out.println("Step 3: use the weights to combine value vectors");
            printVectorMatrix("Context vectors = attention weights x V", result.context());

            double[][] changedInput = copy(INPUT);
            changedInput[2][0] += 100.0;
            changedInput[2][1] -= 100.0;
            AttentionResult changedResult = runAttention(changedInput);
            boolean earlierOutputUnchanged = close(result.context()[0][0], changedResult.context()[0][0])
                    && close(result.context()[0][1], changedResult.context()[0][1]);
            System.out.println("Changing the final input leaves position 0's output unchanged: "
                    + (earlierOutputUnchanged ? "yes" : "no"));
            if (!earlierOutputUnchanged) {
                throw new IllegalStateException("The causal mask allowed information from the future.");
            }
            System.out.println("A decoder token can use its past and present, but cannot see the future.\n");
        }
    }

    private static AttentionResult runAttention(double[][] input) {
        double[][] query = project(input, QUERY_WEIGHTS);
        double[][] key = project(input, KEY_WEIGHTS);
        double[][] value = project(input, VALUE_WEIGHTS);
        double[][] scores = scaledScores(query, key);
        double[][] maskedScores = causalMask(scores);
        double[][] weights = softmaxRows(maskedScores);
        double[][] context = combineValues(weights, value);
        return new AttentionResult(query, key, value, scores, maskedScores, weights, context);
    }

    private static double[][] project(double[][] values, double[][] weights) {
        double[][] projected = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int outputChannel = 0; outputChannel < EMBEDDING_SIZE; outputChannel++) {
                for (int inputChannel = 0; inputChannel < EMBEDDING_SIZE; inputChannel++) {
                    projected[position][outputChannel] += values[position][inputChannel]
                            * weights[inputChannel][outputChannel];
                }
            }
        }
        return projected;
    }

    private static double[][] scaledScores(double[][] query, double[][] key) {
        double[][] scores = new double[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        double scale = Math.sqrt(EMBEDDING_SIZE);
        for (int queryPosition = 0; queryPosition < SEQUENCE_LENGTH; queryPosition++) {
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                    scores[queryPosition][keyPosition] += query[queryPosition][channel]
                            * key[keyPosition][channel];
                }
                scores[queryPosition][keyPosition] /= scale;
            }
        }
        return scores;
    }

    private static double[][] causalMask(double[][] scores) {
        double[][] masked = new double[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        for (int queryPosition = 0; queryPosition < SEQUENCE_LENGTH; queryPosition++) {
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                // A token may use itself and earlier positions, never future positions.
                masked[queryPosition][keyPosition] = keyPosition > queryPosition
                        ? Double.NEGATIVE_INFINITY : scores[queryPosition][keyPosition];
            }
        }
        return masked;
    }

    private static double[][] softmaxRows(double[][] logits) {
        double[][] weights = new double[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        for (int queryPosition = 0; queryPosition < SEQUENCE_LENGTH; queryPosition++) {
            double maximum = Double.NEGATIVE_INFINITY;
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                maximum = Math.max(maximum, logits[queryPosition][keyPosition]);
            }

            double denominator = 0.0;
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                weights[queryPosition][keyPosition] = Math.exp(
                        logits[queryPosition][keyPosition] - maximum);
                denominator += weights[queryPosition][keyPosition];
            }
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                weights[queryPosition][keyPosition] /= denominator;
            }
        }
        return weights;
    }

    private static double[][] combineValues(double[][] weights, double[][] value) {
        double[][] context = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int queryPosition = 0; queryPosition < SEQUENCE_LENGTH; queryPosition++) {
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                    context[queryPosition][channel] += weights[queryPosition][keyPosition]
                            * value[keyPosition][channel];
                }
            }
        }
        return context;
    }

    private static double[][] copy(double[][] values) {
        double[][] result = new double[values.length][];
        for (int row = 0; row < values.length; row++) {
            result[row] = values[row].clone();
        }
        return result;
    }

    private static boolean close(double left, double right) {
        return Math.abs(left - right) < 1e-12;
    }

    private static void printVectorMatrix(String label, double[][] values) {
        System.out.printf("%s (%d x %d):%n", label, values.length, values[0].length);
        printRows(values);
    }

    private static void printScoreMatrix(String label, double[][] values) {
        System.out.printf("%s (%d x %d):%n", label, values.length, values[0].length);
        printRows(values);
    }

    private static void printRows(double[][] values) {
        for (double[] row : values) {
            System.out.print("  [");
            for (int column = 0; column < row.length; column++) {
                if (column > 0) {
                    System.out.print(", ");
                }
                System.out.print(Double.isInfinite(row[column])
                        ? "-inf" : String.format(Locale.ROOT, "%.4f", row[column]));
            }
            System.out.println("]");
        }
    }

    private record AttentionResult(
            double[][] query,
            double[][] key,
            double[][] value,
            double[][] scores,
            double[][] maskedScores,
            double[][] weights,
            double[][] context) {}
}
