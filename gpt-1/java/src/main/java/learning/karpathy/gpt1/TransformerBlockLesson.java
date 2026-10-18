package learning.karpathy.gpt1;

/** Shows how pre-LayerNorm, attention, residual paths, and a feed-forward network form a block. */
public final class TransformerBlockLesson {
    private static final int BATCH_SIZE = 2;
    private static final int TIME_STEPS = 4;
    private static final int CHANNELS = 4;
    private static final int HIDDEN_SIZE = 4 * CHANNELS;
    private static final double EPSILON = 1e-5;

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

    // Fixed Lesson 10 multi-head output for LayerNorm(INPUT), using its shared demo weights.
    private static final double[][][] ATTENTION_OUTPUT = {
            {
                    { 0.9944236133126180, -0.0881970483359226, 0.0500000000000000, -1.6006501782893132 },
                    { 0.3933738827285683, 0.1954140878650754, 0.2038379239277406, -0.6362878460652932 },
                    { -0.5175716661695421, -1.3505289024072524, 1.4923256127023328, -0.1986207426936189 },
                    { 0.6142816558095826, -0.5870337399944243, -0.4857832613323823, 1.4064524206079174 }
            },
            {
                    { 0.3236059033281545, -0.5354088549922318, -0.6208177099844634, 1.0826206616485405 },
                    { 0.6468403211190350, 0.2777975158844621, 0.5677887061817602, -1.9983469601638093 },
                    { 0.6771942604404187, -0.9952272543893472, 0.1465132161629877, -0.0665467975441047 },
                    { -0.2972420175988929, 0.2774110104487157, 0.7685343078948748, -0.3120942225247459 }
            }
    };

    private TransformerBlockLesson() {
    }

    public static void main(String[] args) {
        int[] steps = parseSteps(args);
        double[][][] normalizedInput = layerNorm(INPUT);
        double[][][] afterAttention = add(INPUT, ATTENTION_OUTPUT);
        double[][][] normalizedAfterAttention = layerNorm(afterAttention);
        FeedForward feedForward = new FeedForward();
        FeedForwardResult feedForwardResult = feedForward.forward(normalizedAfterAttention);
        double[][][] blockOutput = add(afterAttention, feedForwardResult.output());

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Normalize each token before its sublayer");
            System.out.println("LayerNorm uses each token's channel mean and population variance; gamma=1 and beta=0 here.");
            printChannels("LayerNorm(input), supplied to Lesson 10 attention", normalizedInput);
            System.out.println("The attention result below is the fixed Lesson 10 example output for this normalized input.");
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Add the attention result through a residual connection");
            printChannels("Attention branch output", ATTENTION_OUTPUT);
            printChannels("After attention residual: input + attention output", afterAttention);
            System.out.println("The residual keeps the original representation available to later layers.");
        }

        if (steps[0] <= 3 && steps[1] >= 3) {
            System.out.println("Step 3 - Normalize, expand, apply ReLU, and project back");
            System.out.println("The position-wise feed-forward network expands 4 channels to 16, then contracts to 4.");
            printChannels("LayerNorm after the attention residual", normalizedAfterAttention);
            System.out.printf("Expanded ReLU representation shape = [%d, %d, %d]. First token: [",
                    BATCH_SIZE, TIME_STEPS, HIDDEN_SIZE);
            for (int hiddenChannel = 0; hiddenChannel < HIDDEN_SIZE; hiddenChannel++) {
                System.out.printf("%s%.4f", hiddenChannel == 0 ? "" : ", ",
                        feedForwardResult.hidden()[0][0][hiddenChannel]);
            }
            System.out.println("]");
            printChannels("Feed-forward output", feedForwardResult.output());
            System.out.println("Dropout follows the second linear layer in training; it is inactive for this evaluation example.");
        }

        if (steps[1] >= 4) {
            double[][][] changedInput = copyInput(normalizedAfterAttention);
            changedInput[0][3][0] += 10.0;
            double[][][] changedFeedForward = feedForward.forward(changedInput).output();
            double earlierDifference = maximumPrefixDifference(
                    feedForwardResult.output(), changedFeedForward, 0, 3);

            System.out.println("Step 4 - Add the feed-forward residual and inspect the block output");
            printChannels("Transformer block output", blockOutput);
            System.out.printf("Block output shape is unchanged: [batch, time, channels] = [%d, %d, %d].%n",
                    BATCH_SIZE, TIME_STEPS, CHANNELS);
            System.out.printf(
                    "Change one token entering the feed-forward network: max change at other tokens = %.12f%n",
                    earlierDifference);
            if (earlierDifference > 1e-10) {
                throw new IllegalStateException("Position-wise feed-forward check failed.");
            }
            System.out.println("Check passed: the feed-forward network mixes channels within each token, not information across time.");
        }
    }

    private static double[][][] layerNorm(double[][][] input) {
        double[][][] output = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                double mean = 0.0;
                for (int channel = 0; channel < CHANNELS; channel++) {
                    mean += input[batch][time][channel];
                }
                mean /= CHANNELS;

                double variance = 0.0;
                for (int channel = 0; channel < CHANNELS; channel++) {
                    double centered = input[batch][time][channel] - mean;
                    variance += centered * centered;
                }
                variance /= CHANNELS;

                for (int channel = 0; channel < CHANNELS; channel++) {
                    output[batch][time][channel] =
                            (input[batch][time][channel] - mean) / Math.sqrt(variance + EPSILON);
                }
            }
        }
        return output;
    }

    private static double[][][] add(double[][][] left, double[][][] right) {
        double[][][] sum = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                for (int channel = 0; channel < CHANNELS; channel++) {
                    sum[batch][time][channel] =
                            left[batch][time][channel] + right[batch][time][channel];
                }
            }
        }
        return sum;
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
        return "Usage: TransformerBlockLesson [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]";
    }

    private static final class FeedForward {
        private final double[][] firstWeight = new double[HIDDEN_SIZE][CHANNELS];
        private final double[] firstBias = new double[HIDDEN_SIZE];
        private final double[][] secondWeight = new double[CHANNELS][HIDDEN_SIZE];
        private final double[] secondBias = { 0.02, -0.01, 0.03, -0.02 };

        private FeedForward() {
            for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
                firstBias[hidden] = (hidden % 3 - 1) * 0.05;
                for (int channel = 0; channel < CHANNELS; channel++) {
                    int pattern = ((hidden + 1) * (channel + 2)) % 7 - 3;
                    firstWeight[hidden][channel] = pattern * 0.1;
                }
            }
            for (int channel = 0; channel < CHANNELS; channel++) {
                for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
                    int pattern = ((channel + 2) * (hidden + 1)) % 5 - 2;
                    secondWeight[channel][hidden] = pattern * 0.05;
                }
            }
        }

        private FeedForwardResult forward(double[][][] input) {
            double[][][] hiddenValues = new double[BATCH_SIZE][TIME_STEPS][HIDDEN_SIZE];
            double[][][] output = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
                        double value = firstBias[hidden];
                        for (int channel = 0; channel < CHANNELS; channel++) {
                            value += input[batch][time][channel] * firstWeight[hidden][channel];
                        }
                        hiddenValues[batch][time][hidden] = Math.max(0.0, value);
                    }
                    for (int channel = 0; channel < CHANNELS; channel++) {
                        double value = secondBias[channel];
                        for (int hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
                            value += hiddenValues[batch][time][hidden] * secondWeight[channel][hidden];
                        }
                        output[batch][time][channel] = value;
                    }
                }
            }
            return new FeedForwardResult(hiddenValues, output);
        }
    }

    private record FeedForwardResult(double[][][] hidden, double[][][] output) {
    }
}
