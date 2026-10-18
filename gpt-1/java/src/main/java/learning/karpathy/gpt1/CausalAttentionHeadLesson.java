package learning.karpathy.gpt1;

/** Builds one small causal self-attention head from query, key, and value projections. */
public final class CausalAttentionHeadLesson {
    private static final int BATCH_SIZE = 2;
    private static final int TIME_STEPS = 4;
    private static final int CHANNELS = 4;
    private static final int HEAD_SIZE = 2;

    private static final double[][][] INPUT = {
            {
                    { 1.0, 0.0, 2.0, -1.0 }, { 0.0, 1.0, 1.0, 2.0 },
                    { 2.0, -1.0, 0.0, 1.0 }, { 1.0, 2.0, -1.0, 0.0 }
            },
            {
                    { 0.0, 2.0, 1.0, -1.0 }, { 1.0, -1.0, 2.0, 0.0 },
                    { -1.0, 0.0, 1.0, 2.0 }, { 2.0, 1.0, 0.0, -1.0 }
            }
    };

    // Rows are output channels, matching nn.Linear.weight's [out_features, in_features].
    private static final double[][] QUERY_WEIGHT = {
            { 0.5, -0.5, 1.0, 0.0 }, { 0.0, 1.0, 0.5, -0.5 }
    };
    private static final double[][] KEY_WEIGHT = {
            { 1.0, 0.0, 0.5, -0.5 }, { -0.5, 0.5, 0.0, 1.0 }
    };
    private static final double[][] VALUE_WEIGHT = {
            { 0.5, 0.0, 1.0, 0.5 }, { 0.0, 1.0, -0.5, 0.5 }
    };

    private CausalAttentionHeadLesson() {
    }

    public static void main(String[] args) {
        int[] steps = parseSteps(args);
        AttentionResult result = forward(INPUT);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Project each token into query, key, and value vectors");
            System.out.println("Input shape [batch, time, channels] = [2, 4, 4]; each projection has head size 2.");
            System.out.println("These fixed matrices stand in for the trainable weights of three nn.Linear layers.");
            printVectors("Queries", result.queries());
            printVectors("Keys", result.keys());
            printVectors("Values", result.values());
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Compare each query with every key and scale the dot products");
            printScores("Raw query-key scores", result.rawScores());
            printScores("Scaled scores = query dot key / sqrt(head_size)", result.scaledScores());
            System.out.println("Scores have shape [batch, query time, key time] = [2, 4, 4].");
        }

        if (steps[0] <= 3 && steps[1] >= 3) {
            System.out.println("Step 3 - Hide future positions, then normalize each row with softmax");
            printScores("Scores after causal masking", result.maskedScores());
            printScores("Attention weights", result.weights());
            System.out.println("Every row sums to 1; future positions have weight 0.");
        }

        if (steps[1] >= 4) {
            System.out.println("Step 4 - Use attention weights to combine the value vectors");
            printVectors("Single-head output", result.output());
            System.out.println("The output shape is [batch, time, head_size] = [2, 4, 2].");

            double[][][] changedInput = copyInput(INPUT);
            changedInput[0][3][0] += 100.0;
            AttentionResult changedResult = forward(changedInput);
            double earlierDifference = maximumPrefixDifference(
                    result.output(), changedResult.output(), 0, 3);
            double finalDifference = maximumDifference(
                    result.output()[0][3], changedResult.output()[0][3]);
            System.out.printf(
                    "Change a future input token by +100: max change at earlier positions = %.12f%n",
                    earlierDifference);
            System.out.printf("Change at the edited position's output = %.12f%n", finalDifference);
            if (earlierDifference > 1e-10 || finalDifference < 1e-8) {
                throw new IllegalStateException("Causality check failed.");
            }
            System.out.println("Causality check passed: future input changes cannot alter earlier outputs.");
        }
    }

    private static AttentionResult forward(double[][][] input) {
        double[][][] queries = project(input, QUERY_WEIGHT);
        double[][][] keys = project(input, KEY_WEIGHT);
        double[][][] values = project(input, VALUE_WEIGHT);
        double[][][] rawScores = new double[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
        double[][][] scaledScores = new double[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
        double[][][] maskedScores = new double[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
        double[][][] weights = new double[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
        double[][][] output = new double[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                    double score = 0.0;
                    for (int channel = 0; channel < HEAD_SIZE; channel++) {
                        score += queries[batch][time][channel] * keys[batch][keyTime][channel];
                    }
                    rawScores[batch][time][keyTime] = score;
                    scaledScores[batch][time][keyTime] = score / Math.sqrt(HEAD_SIZE);
                    maskedScores[batch][time][keyTime] = keyTime <= time
                            ? scaledScores[batch][time][keyTime]
                            : Double.NEGATIVE_INFINITY;
                }
            }
        }

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                double maximum = maskedScores[batch][time][0];
                for (int keyTime = 1; keyTime <= time; keyTime++) {
                    maximum = Math.max(maximum, maskedScores[batch][time][keyTime]);
                }

                double total = 0.0;
                for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                    double score = maskedScores[batch][time][keyTime];
                    weights[batch][time][keyTime] =
                            Double.isInfinite(score) ? 0.0 : Math.exp(score - maximum);
                    total += weights[batch][time][keyTime];
                }
                for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                    weights[batch][time][keyTime] /= total;
                }
            }
        }

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                for (int channel = 0; channel < HEAD_SIZE; channel++) {
                    for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                        output[batch][time][channel] +=
                                weights[batch][time][keyTime] * values[batch][keyTime][channel];
                    }
                }
            }
        }

        return new AttentionResult(
                queries, keys, values, rawScores, scaledScores, maskedScores, weights, output);
    }

    private static double[][][] project(double[][][] input, double[][] weight) {
        double[][][] projected = new double[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                for (int outputChannel = 0; outputChannel < HEAD_SIZE; outputChannel++) {
                    for (int inputChannel = 0; inputChannel < CHANNELS; inputChannel++) {
                        projected[batch][time][outputChannel] +=
                                input[batch][time][inputChannel]
                                        * weight[outputChannel][inputChannel];
                    }
                }
            }
        }
        return projected;
    }

    private static double[][][] copyInput(double[][][] input) {
        double[][][] copy = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                System.arraycopy(input[batch][time], 0, copy[batch][time], 0, CHANNELS);
            }
        }
        return copy;
    }

    private static double maximumPrefixDifference(
            double[][][] left, double[][][] right, int batch, int prefixLength) {
        double maximum = 0.0;
        for (int time = 0; time < prefixLength; time++) {
            maximum = Math.max(maximum, maximumDifference(left[batch][time], right[batch][time]));
        }
        return maximum;
    }

    private static double maximumDifference(double[] left, double[] right) {
        double maximum = 0.0;
        for (int channel = 0; channel < left.length; channel++) {
            maximum = Math.max(maximum, Math.abs(left[channel] - right[channel]));
        }
        return maximum;
    }

    private static void printVectors(String label, double[][][] values) {
        System.out.printf("%s, shape [%d, %d, %d]:%n",
                label, BATCH_SIZE, TIME_STEPS, HEAD_SIZE);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            System.out.printf("  batch %d%n", batch);
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.printf("    t=%d [%.4f, %.4f]%n",
                        time, values[batch][time][0], values[batch][time][1]);
            }
        }
    }

    private static void printScores(String label, double[][][] values) {
        System.out.printf("%s, shape [%d, %d, %d] (batch 0 shown):%n",
                label, BATCH_SIZE, TIME_STEPS, TIME_STEPS);
        for (int time = 0; time < TIME_STEPS; time++) {
            System.out.print(" ");
            for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                double score = values[0][time][keyTime];
                if (Double.isInfinite(score)) {
                    System.out.printf(" %8s", "-inf");
                } else {
                    System.out.printf(" %8.4f", score);
                }
            }
            System.out.println();
        }
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
        return "Usage: CausalAttentionHeadLesson [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]";
    }

    private record AttentionResult(
            double[][][] queries,
            double[][][] keys,
            double[][][] values,
            double[][][] rawScores,
            double[][][] scaledScores,
            double[][][] maskedScores,
            double[][][] weights,
            double[][][] output) {
    }
}
