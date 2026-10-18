package learning.karpathy.gpt1;

/** Shows why scaled dot-product attention divides query-key scores by sqrt(head size). */
public final class AttentionScalingLesson {
    private static final int MAX_HEAD_SIZE = 8;
    private static final int TOY_LOGIT_COUNT = 5;

    private AttentionScalingLesson() {
    }

    public static void main(String[] args) {
        int[] steps = parseSteps(args);

        if (steps[0] <= 1) {
            int[] headSizes = { 1, 2, 4, 8 };
            System.out.println("Step 1 - See how query-key score variance grows with head size");
            System.out.println("Fix q to all +1 values and enumerate every key made of -1/+1 values.");
            System.out.println("This exactly samples independent, zero-mean, unit-variance key components.");
            System.out.println("head size | raw score variance | variance after dividing by sqrt(head size)");

            for (int headSize : headSizes) {
                int keyCount = 1 << headSize;
                double scoreSum = 0.0;
                double squaredScoreSum = 0.0;
                for (int pattern = 0; pattern < keyCount; pattern++) {
                    double score = 0.0;
                    for (int channel = 0; channel < headSize; channel++) {
                        score += (pattern & (1 << channel)) != 0 ? 1.0 : -1.0;
                    }
                    scoreSum += score;
                    squaredScoreSum += score * score;
                }

                double mean = scoreSum / keyCount;
                double variance = squaredScoreSum / keyCount - mean * mean;
                double scaledVariance = variance / headSize;
                System.out.printf("%9d | %18.4f | %39.4f%n",
                        headSize, variance, scaledVariance);
            }
            System.out.println("The dot product sums head_size terms, so its variance grows with head_size.");
            System.out.println("Dividing by sqrt(head_size) divides that variance by head_size.");
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            double[] baseLogits = { 0.1, -0.2, 0.3, -0.2, 0.5 };
            double[] sharpLogits = new double[TOY_LOGIT_COUNT];
            double scale = 8.0; // sqrt(64), used to make the effect easy to see.
            for (int index = 0; index < TOY_LOGIT_COUNT; index++) {
                sharpLogits[index] = baseLogits[index] * scale;
            }
            double[] baseProbabilities = softmax(baseLogits);
            double[] sharpProbabilities = softmax(sharpLogits);

            System.out.println("Step 2 - See how large logits make softmax overly concentrated");
            printVector("Example logits", baseLogits);
            printVector("Softmax at the original scale", baseProbabilities);
            printVector("Logits multiplied by 8 (sqrt of head size 64)", sharpLogits);
            printVector("Softmax at the larger scale", sharpProbabilities);
            System.out.printf("Largest probability: %.4f -> %.4f; entropy: %.4f -> %.4f nats%n",
                    baseProbabilities[4], sharpProbabilities[4],
                    entropy(baseProbabilities), entropy(sharpProbabilities));
            System.out.println("Large score gaps concentrate probability; saturated softmax can produce small gradients.");
        }

        if (steps[1] >= 3) {
            double[] query = new double[MAX_HEAD_SIZE];
            double[][] keys = new double[TOY_LOGIT_COUNT][MAX_HEAD_SIZE];
            double[] rawScores = new double[TOY_LOGIT_COUNT];
            double[] scaledScores = new double[TOY_LOGIT_COUNT];
            for (int channel = 0; channel < MAX_HEAD_SIZE; channel++) {
                query[channel] = 1.0;
            }
            for (int key = 0; key < TOY_LOGIT_COUNT; key++) {
                for (int channel = 0; channel < MAX_HEAD_SIZE; channel++) {
                    keys[key][channel] = channel < 2 * key ? -1.0 : 1.0;
                    rawScores[key] += query[channel] * keys[key][channel];
                }
                scaledScores[key] = rawScores[key] / Math.sqrt(MAX_HEAD_SIZE);
            }

            double[] rawProbabilities = softmax(rawScores);
            double[] scaledProbabilities = softmax(scaledScores);
            System.out.println("Step 3 - Scale actual query-key dot products before softmax");
            printKeyVectors(keys);
            printVector("Raw dot-product scores", rawScores);
            printVector("Raw softmax weights", rawProbabilities);
            printVector("Scores divided by sqrt(8)", scaledScores);
            printVector("Scaled softmax weights", scaledProbabilities);
            System.out.printf("Largest attention weight: %.4f -> %.4f%n",
                    rawProbabilities[0], scaledProbabilities[0]);
            System.out.println("Scaling controls the score range while preserving which keys score higher.");
        }
    }

    private static double[] softmax(double[] scores) {
        double maximum = scores[0];
        for (int index = 1; index < scores.length; index++) {
            maximum = Math.max(maximum, scores[index]);
        }
        double[] probabilities = new double[scores.length];
        double total = 0.0;
        for (int index = 0; index < scores.length; index++) {
            probabilities[index] = Math.exp(scores[index] - maximum);
            total += probabilities[index];
        }
        for (int index = 0; index < scores.length; index++) {
            probabilities[index] /= total;
        }
        return probabilities;
    }

    private static double entropy(double[] probabilities) {
        double result = 0.0;
        for (double probability : probabilities) {
            result -= probability * Math.log(probability);
        }
        return result;
    }

    private static void printVector(String label, double[] values) {
        System.out.print(label + ": [");
        for (int index = 0; index < values.length; index++) {
            System.out.printf("%s%.6g", index == 0 ? "" : ", ", values[index]);
        }
        System.out.println("]");
    }

    private static void printKeyVectors(double[][] keys) {
        System.out.println("Five key vectors (each has eight components):");
        for (int key = 0; key < keys.length; key++) {
            System.out.print("  key " + key + ": [");
            for (int channel = 0; channel < keys[key].length; channel++) {
                System.out.printf("%s%.0f", channel == 0 ? "" : ", ", keys[key][channel]);
            }
            System.out.println("]");
        }
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException(
                    "Usage: AttentionScalingLesson [all|1|2|3|1-2|1-3]");
        }
        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "1-2" -> new int[] { 1, 2 };
            case "all", "1-3" -> new int[] { 1, 3 };
            default -> throw new IllegalArgumentException(
                    "Usage: AttentionScalingLesson [all|1|2|3|1-2|1-3]");
        };
    }
}
