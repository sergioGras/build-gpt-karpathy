package learning.karpathy.gpt1;

/** Demonstrates LayerNorm across each token's channel dimension. */
public final class LayerNormLesson {
    private static final int BATCH_SIZE = 2;
    private static final int TIME_STEPS = 2;
    private static final int CHANNELS = 4;
    private static final double EPSILON = 1e-5;
    private static final double[][][] INPUT = {
            { { 1.0, 2.0, 3.0, 4.0 }, { 2.0, 4.0, 6.0, 8.0 } },
            { { 10.0, 11.0, 12.0, 13.0 }, { 3.0, 1.0, 4.0, 2.0 } }
    };
    private static final double[] GAMMA = { 1.0, 0.5, 2.0, -1.0 };
    private static final double[] BETA = { 0.0, 1.0, -1.0, 0.5 };

    private LayerNormLesson() {
    }

    public static void main(String[] args) {
        int[] steps = parseSteps(args);
        double[][][] normalized = normalizeInput();

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Choose the dimension being normalized");
            System.out.println("Input shape: (batch=2, time=2, channels=4)");
            System.out.println("LayerNorm computes statistics over each token's channels independently.");
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    double[] stats = rowStatistics(INPUT[batch][time]);
                    System.out.printf("  batch %d, time %d input ", batch, time);
                    printVector(INPUT[batch][time]);
                    System.out.printf(" -> mean %.4f, population variance %.4f%n",
                            stats[0], stats[1]);
                }
            }
            System.out.println("Other batch examples and time positions do not affect these statistics.");
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Center and rescale each token vector");
            System.out.println("x_hat = (x - mean) / sqrt(population_variance + eps), with eps = 1e-5");
            printTokenVectors("Normalized token vectors:", normalized);
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    double[] stats = rowStatistics(normalized[batch][time]);
                    System.out.printf("  normalized mean %.4f, population std %.4f%n",
                            stats[0], Math.sqrt(stats[1]));
                }
            }
            double channelZeroSum = 0.0;
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    channelZeroSum += normalized[batch][time][0];
                }
            }
            System.out.printf("Mean of channel 0 across all tokens (not constrained): %.4f%n",
                    channelZeroSum / (BATCH_SIZE * TIME_STEPS));
        }

        if (steps[1] >= 3) {
            double[][][] output = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < TIME_STEPS; time++) {
                    for (int channel = 0; channel < CHANNELS; channel++) {
                        output[batch][time][channel] =
                                GAMMA[channel] * normalized[batch][time][channel] + BETA[channel];
                    }
                }
            }

            System.out.println("Step 3 - Apply a learned scale and shift per channel");
            System.out.print("gamma: ");
            printVector(GAMMA);
            System.out.print("\nbeta:  ");
            printVector(BETA);
            System.out.println();
            System.out.println("LayerNorm output: gamma * x_hat + beta");
            printTokenVectors("Output token vectors:", output);
            System.out.println("These example gamma/beta values are fixed here; training would update them by gradient descent.");
        }
    }

    private static double[][][] normalizeInput() {
        double[][][] normalized = new double[BATCH_SIZE][TIME_STEPS][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                double[] stats = rowStatistics(INPUT[batch][time]);
                for (int channel = 0; channel < CHANNELS; channel++) {
                    normalized[batch][time][channel] =
                            (INPUT[batch][time][channel] - stats[0])
                                    / Math.sqrt(stats[1] + EPSILON);
                }
            }
        }
        return normalized;
    }

    /** Returns the mean and population variance of one channel vector. */
    private static double[] rowStatistics(double[] row) {
        double mean = 0.0;
        for (double value : row) {
            mean += value;
        }
        mean /= row.length;

        double variance = 0.0;
        for (double value : row) {
            double centered = value - mean;
            variance += centered * centered;
        }
        variance /= row.length; // PyTorch LayerNorm uses the population variance.
        return new double[] { mean, variance };
    }

    private static void printTokenVectors(String label, double[][][] values) {
        System.out.println(label);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_STEPS; time++) {
                System.out.printf("  batch %d, time %d: ", batch, time);
                printVector(values[batch][time]);
                System.out.println();
            }
        }
    }

    private static void printVector(double[] values) {
        System.out.print("[");
        for (int index = 0; index < values.length; index++) {
            System.out.printf("%s%.4f", index == 0 ? "" : ", ", values[index]);
        }
        System.out.print("]");
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException("Usage: LayerNormLesson [all|1|2|3|1-2|1-3]");
        }
        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "1-2" -> new int[] { 1, 2 };
            case "all", "1-3" -> new int[] { 1, 3 };
            default -> throw new IllegalArgumentException("Usage: LayerNormLesson [all|1|2|3|1-2|1-3]");
        };
    }
}
