package learning.karpathy.gpt1;

/** Small array-based examples for the causal weighted aggregation behind self-attention. */
public final class SelfAttentionMathLesson {
    private static final int TIME_STEPS = 4;
    private static final int CHANNELS = 2;

    private SelfAttentionMathLesson() {
    }

    public static void main(String[] args) {
        int[] steps = parseSteps(args);
        double[][] input = {
                { 1.0, 0.0 }, { 0.0, 2.0 }, { 3.0, 1.0 }, { 2.0, 4.0 }
        };

        // Compute all tiny arrays once; the selected part controls which step is shown.
        double[][] loopOutput = averageVisiblePast(input);
        double[][] causalWeights = causalAverageWeights();
        double[][] matrixOutput = applyWeights(causalWeights, input);

        // Negative infinity becomes probability zero after softmax.
        double[][] zeroScores = new double[TIME_STEPS][TIME_STEPS];
        for (int time = 0; time < TIME_STEPS; time++) {
            for (int key = time + 1; key < TIME_STEPS; key++) {
                zeroScores[time][key] = Double.NEGATIVE_INFINITY;
            }
        }
        double[][] uniformWeights = softmaxRows(zeroScores);
        double[][] softmaxOutput = applyWeights(uniformWeights, input);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Average each token with the visible past using loops");
            printVectors("Input vectors", input);
            printVectors("Causal averages", loopOutput);
            System.out.println("At time t, this loop reads positions 0 through t and cannot see the future.");
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Write the same operation as a matrix multiplication");
            printMatrix("Lower-triangular averaging weights", causalWeights);
            printVectors("Weights x input", matrixOutput);
            System.out.printf("Maximum difference from the loop version: %.12f%n",
                    maximumDifference(loopOutput, matrixOutput));
        }

        if (steps[0] <= 3 && steps[1] >= 3) {
            System.out.println("Step 3 - Use a causal mask and softmax to create the weights");
            printMatrix("Scores after masking future positions", zeroScores);
            printMatrix("Softmax weights", uniformWeights);
            System.out.printf("Maximum difference from the loop version: %.12f%n",
                    maximumDifference(loopOutput, softmaxOutput));
        }

        if (steps[1] >= 4) {
            double[][] queryProjection = {
                    { 1.0, 0.5 }, { -0.5, 1.0 }
            };
            double[][] keyProjection = {
                    { 0.5, -0.5 }, { 1.0, 0.5 }
            };
            double[][] queries = project(input, queryProjection);
            double[][] keys = project(input, keyProjection);
            double[][] attentionScores = new double[TIME_STEPS][TIME_STEPS];
            for (int time = 0; time < TIME_STEPS; time++) {
                for (int key = 0; key < TIME_STEPS; key++) {
                    if (key > time) {
                        attentionScores[time][key] = Double.NEGATIVE_INFINITY;
                    } else {
                        double score = 0.0;
                        for (int channel = 0; channel < CHANNELS; channel++) {
                            score += queries[time][channel] * keys[key][channel];
                        }
                        attentionScores[time][key] = score / Math.sqrt(CHANNELS);
                    }
                }
            }

            double[][] attentionWeights = softmaxRows(attentionScores);
            double[][] attentionOutput = applyWeights(attentionWeights, input);

            System.out.println("Step 4 - Let query-key similarity choose the causal weights");
            System.out.println("The fixed projection matrices stand in for learned query/key layers; input vectors act as values.");
            printVectors("Queries", queries);
            printVectors("Keys", keys);
            printMatrix("Scaled query-key scores after the causal mask", attentionScores);
            printMatrix("Input-dependent attention weights", attentionWeights);
            printVectors("Weighted value vectors", attentionOutput);
            System.out.println("Each row attends only to its own position and earlier positions; later positions have weight zero.");
        }
    }

    private static double[][] averageVisiblePast(double[][] input) {
        double[][] output = new double[TIME_STEPS][CHANNELS];
        for (int time = 0; time < TIME_STEPS; time++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                double total = 0.0;
                for (int past = 0; past <= time; past++) {
                    total += input[past][channel];
                }
                output[time][channel] = total / (time + 1);
            }
        }
        return output;
    }

    private static double[][] causalAverageWeights() {
        double[][] weights = new double[TIME_STEPS][TIME_STEPS];
        for (int time = 0; time < TIME_STEPS; time++) {
            for (int past = 0; past <= time; past++) {
                weights[time][past] = 1.0 / (time + 1);
            }
        }
        return weights;
    }

    private static double[][] applyWeights(double[][] weights, double[][] values) {
        double[][] output = new double[TIME_STEPS][CHANNELS];
        for (int time = 0; time < TIME_STEPS; time++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                for (int key = 0; key < TIME_STEPS; key++) {
                    output[time][channel] += weights[time][key] * values[key][channel];
                }
            }
        }
        return output;
    }

    private static double[][] softmaxRows(double[][] scores) {
        double[][] weights = new double[TIME_STEPS][TIME_STEPS];
        for (int time = 0; time < TIME_STEPS; time++) {
            double maximum = scores[time][0];
            for (int key = 1; key < TIME_STEPS; key++) {
                maximum = Math.max(maximum, scores[time][key]);
            }
            double total = 0.0;
            for (int key = 0; key < TIME_STEPS; key++) {
                weights[time][key] = Math.exp(scores[time][key] - maximum);
                total += weights[time][key];
            }
            for (int key = 0; key < TIME_STEPS; key++) {
                weights[time][key] /= total;
            }
        }
        return weights;
    }

    private static double[][] project(double[][] input, double[][] projection) {
        double[][] output = new double[TIME_STEPS][CHANNELS];
        for (int time = 0; time < TIME_STEPS; time++) {
            for (int outputChannel = 0; outputChannel < CHANNELS; outputChannel++) {
                for (int inputChannel = 0; inputChannel < CHANNELS; inputChannel++) {
                    output[time][outputChannel] +=
                            input[time][inputChannel] * projection[inputChannel][outputChannel];
                }
            }
        }
        return output;
    }

    private static double maximumDifference(double[][] left, double[][] right) {
        double maximum = 0.0;
        for (int time = 0; time < TIME_STEPS; time++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                maximum = Math.max(maximum, Math.abs(left[time][channel] - right[time][channel]));
            }
        }
        return maximum;
    }

    private static void printVectors(String label, double[][] vectors) {
        System.out.printf("%s, shape [%d, %d]:%n", label, TIME_STEPS, CHANNELS);
        for (int time = 0; time < TIME_STEPS; time++) {
            System.out.printf("  t=%d [%.3f, %.3f]%n", time, vectors[time][0], vectors[time][1]);
        }
    }

    private static void printMatrix(String label, double[][] matrix) {
        System.out.printf("%s, shape [%d, %d]:%n", label, TIME_STEPS, TIME_STEPS);
        for (double[] row : matrix) {
            System.out.print(" ");
            for (double value : row) {
                if (Double.isInfinite(value)) {
                    System.out.printf(" %8s", "-inf");
                } else {
                    System.out.printf(" %8.4f", value);
                }
            }
            System.out.println();
        }
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException(
                    "Usage: SelfAttentionMathLesson [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]");
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
            default -> throw new IllegalArgumentException(
                    "Usage: SelfAttentionMathLesson [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]");
        };
    }
}
