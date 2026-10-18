package learning.karpathy.gpt1;

/** Demonstrates parallel causal heads, concatenation, and the output projection. */
public final class MultiHeadAttentionLesson {
    private static final int BATCH_SIZE = 2;
    private static final int TIME_STEPS = 4;
    private static final int CHANNELS = 4;
    private static final int HEADS = 2;
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

    // Each weight matrix is [output channel][input channel], like nn.Linear.weight.
    private static final double[][][] QUERY_WEIGHT = {
            {
                    { 0.5, -0.5, 1.0, 0.0 }, { 0.0, 1.0, 0.5, -0.5 }
            },
            {
                    { 0.0, 0.5, -0.5, 1.0 }, { 1.0, -0.5, 0.0, 0.5 }
            }
    };
    private static final double[][][] KEY_WEIGHT = {
            {
                    { 1.0, 0.0, 0.5, -0.5 }, { -0.5, 0.5, 0.0, 1.0 }
            },
            {
                    { 0.5, 1.0, -0.5, 0.0 }, { -1.0, 0.0, 0.5, 0.5 }
            }
    };
    private static final double[][][] VALUE_WEIGHT = {
            {
                    { 0.5, 0.0, 1.0, 0.5 }, { 0.0, 1.0, -0.5, 0.5 }
            },
            {
                    { 1.0, 0.5, 0.0, -0.5 }, { 0.5, -1.0, 1.0, 0.0 }
            }
    };

    private static final double[][] OUTPUT_WEIGHT = {
            { 0.5, 0.0, 0.5, 0.0 },
            { 0.0, 0.5, 0.0, 0.5 },
            { 0.5, 0.0, -0.5, 0.0 },
            { 0.0, 0.5, 0.0, -0.5 }
    };
    private static final double[] OUTPUT_BIAS = { 0.1, -0.2, 0.05, 0.3 };

    private MultiHeadAttentionLesson() {
    }

    public static void main(String[] args) {
        int[] steps = parseSteps(args);
        AttentionResult result = forward(INPUT);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Run two independent causal attention heads");
            System.out.println("Each head has its own query, key, and value projections and returns two channels per token.");
            System.out.println("Attention weights use scaled query-key scores, a causal mask, and row-wise softmax.");
            printWeights(result.attentionWeights());
            printHeadVectors("Individual head outputs", result.headOutputs());
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Concatenate the head outputs along the channel dimension");
            System.out.println("Concatenation places head 0's two channels before head 1's two channels.");
            printChannels("Concatenated heads", result.concatenated());
            System.out.println("The shape is [batch, time, heads * head_size] = [2, 4, 4].");
        }

        if (steps[0] <= 3 && steps[1] >= 3) {
            System.out.println("Step 3 - Project the concatenated channels back to model width");
            System.out.println("A learned linear projection mixes information from all heads; this example uses fixed weights.");
            printChannels("Projected multi-head output", result.output());
            System.out.println("The model width is 4, so the shape returns to [batch, time, channels] = [2, 4, 4].");
            System.out.println("Dropout is inactive in evaluation mode; C and Java show this deterministic forward pass.");
        }

        if (steps[1] >= 4) {
            double[][][] changedInput = copyInput(INPUT);
            changedInput[0][3][0] += 100.0;
            AttentionResult changedResult = forward(changedInput);
            double earlierDifference = maximumPrefixDifference(
                    result.output(), changedResult.output(), 0, 3);

            System.out.println("Step 4 - Verify the combined block remains causal");
            System.out.printf(
                    "Change a future input token by +100: max change at earlier outputs = %.12f%n",
                    earlierDifference);
            if (earlierDifference > 1e-10) {
                throw new IllegalStateException("Causality check failed.");
            }
            System.out.println("Causality check passed: neither head can pass future information to the past.");
        }
    }

    private static AttentionResult forward(double[][][] input) {
        double[][][][] headOutputs = new double[HEADS][BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
        double[][][][] attentionWeights = new double[HEADS][BATCH_SIZE][TIME_STEPS][TIME_STEPS];

        for (int head = 0; head < HEADS; head++) {
            double[][][] queries = project(input, QUERY_WEIGHT[head]);
            double[][][] keys = project(input, KEY_WEIGHT[head]);
            double[][][] values = project(input, VALUE_WEIGHT[head]);
            double[][][] scores = new double[BATCH_SIZE][TIME_STEPS][TIME_STEPS];

            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                        double score = 0.0;
                        for (int channel = 0; channel < HEAD_SIZE; channel++) {
                            score += queries[batch][time][channel] * keys[batch][keyTime][channel];
                        }
                        scores[batch][time][keyTime] = keyTime <= time
                                ? score / Math.sqrt(HEAD_SIZE)
                                : Double.NEGATIVE_INFINITY;
                    }
                }
            }

            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    double maximum = scores[batch][time][0];
                    for (int keyTime = 1; keyTime <= time; keyTime++) {
                        maximum = Math.max(maximum, scores[batch][time][keyTime]);
                    }

                    double total = 0.0;
                    for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                        double score = scores[batch][time][keyTime];
                        attentionWeights[head][batch][time][keyTime] =
                                Double.isInfinite(score) ? 0.0 : Math.exp(score - maximum);
                        total += attentionWeights[head][batch][time][keyTime];
                    }
                    for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                        attentionWeights[head][batch][time][keyTime] /= total;
                    }
                }
            }

            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    for (int channel = 0; channel < HEAD_SIZE; channel++) {
                        for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                            headOutputs[head][batch][time][channel] +=
                                    attentionWeights[head][batch][time][keyTime]
                                            * values[batch][keyTime][channel];
                        }
                    }
                }
            }
        }

        double[][][] concatenated = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                for (int head = 0; head < HEADS; head++) {
                    for (int channel = 0; channel < HEAD_SIZE; channel++) {
                        concatenated[batch][time][head * HEAD_SIZE + channel] =
                                headOutputs[head][batch][time][channel];
                    }
                }
            }
        }

        double[][][] output = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                for (int outputChannel = 0; outputChannel < CHANNELS; outputChannel++) {
                    output[batch][time][outputChannel] = OUTPUT_BIAS[outputChannel];
                    for (int inputChannel = 0; inputChannel < CHANNELS; inputChannel++) {
                        output[batch][time][outputChannel] +=
                                concatenated[batch][time][inputChannel]
                                        * OUTPUT_WEIGHT[outputChannel][inputChannel];
                    }
                }
            }
        }

        return new AttentionResult(headOutputs, attentionWeights, concatenated, output);
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
            for (int channel = 0; channel < CHANNELS; channel++) {
                maximum = Math.max(
                        maximum, Math.abs(left[batch][time][channel] - right[batch][time][channel]));
            }
        }
        return maximum;
    }

    private static void printHeadVectors(String label, double[][][][] values) {
        System.out.printf("%s, shape [heads, batch, time, head_size] = [%d, %d, %d, %d]:%n",
                label, HEADS, BATCH_SIZE, TIME_STEPS, HEAD_SIZE);
        for (int head = 0; head < HEADS; head++) {
            System.out.printf("  head %d, batch 0:%n", head);
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.printf("    t=%d [%.4f, %.4f]%n",
                        time, values[head][0][time][0], values[head][0][time][1]);
            }
        }
    }

    private static void printWeights(double[][][][] values) {
        System.out.printf("Attention weights, shape [heads, batch, time, time] = [%d, %d, %d, %d] (batch 0):%n",
                HEADS, BATCH_SIZE, TIME_STEPS, TIME_STEPS);
        for (int head = 0; head < HEADS; head++) {
            System.out.printf("  head %d:%n", head);
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.print(" ");
                for (int keyTime = 0; keyTime < TIME_STEPS; keyTime++) {
                    System.out.printf(" %8.4f", values[head][0][time][keyTime]);
                }
                System.out.println();
            }
        }
    }

    private static void printChannels(String label, double[][][] values) {
        System.out.printf("%s, shape [batch, time, channels] = [%d, %d, %d]:%n",
                label, BATCH_SIZE, TIME_STEPS, CHANNELS);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            System.out.printf("  batch %d%n", batch);
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.printf("    t=%d [", time);
                for (int channel = 0; channel < CHANNELS; channel++) {
                    System.out.printf("%s%.4f", channel == 0 ? "" : ", ", values[batch][time][channel]);
                }
                System.out.println("]");
            }
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
        return "Usage: MultiHeadAttentionLesson [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]";
    }

    private record AttentionResult(
            double[][][][] headOutputs,
            double[][][][] attentionWeights,
            double[][][] concatenated,
            double[][][] output) {
    }
}
