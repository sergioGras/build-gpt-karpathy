package learning.karpathy.gpt1;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/** Runnable Java companion to lesson 5: train a bigram model and sample from it. */
public final class BigramTrainingLesson {
    private static final int BLOCK_SIZE = 8;
    private static final int BATCH_SIZE = 32;
    private static final int TRAINING_STEPS = 100;
    private static final double LEARNING_RATE = 1e-3;
    private static final double WEIGHT_DECAY = 0.01;

    private BigramTrainingLesson() {
    }

    public static void main(String[] args) throws IOException {
        int[] steps = parseSteps(args);
        TinyShakespeareDataset dataset = TinyShakespeareDataset.load(
                TinyShakespeareDataset.findDefaultPath());
        String text = dataset.text();
        CharacterVocabulary vocabulary = CharacterVocabulary.fromText(text);
        int[] tokenIds = vocabulary.encode(text);
        int splitIndex = tokenIds.length * 9 / 10;
        int[] trainData = Arrays.copyOfRange(tokenIds, 0, splitIndex);
        int[] validationData = Arrays.copyOfRange(tokenIds, splitIndex, tokenIds.length);

        // Reuse these fixed batches to show how the same examples improve.
        Random evaluationRandom = new Random(1337);
        int[][] trainEvalInputs = new int[BATCH_SIZE][BLOCK_SIZE];
        int[][] trainEvalTargets = new int[BATCH_SIZE][BLOCK_SIZE];
        int[][] validationInputs = new int[BATCH_SIZE][BLOCK_SIZE];
        int[][] validationTargets = new int[BATCH_SIZE][BLOCK_SIZE];
        sampleBatch(trainData, evaluationRandom, trainEvalInputs, trainEvalTargets);
        sampleBatch(validationData, evaluationRandom, validationInputs, validationTargets);

        BigramLanguageModel model = new BigramLanguageModel(vocabulary.size(), 1337);
        double initialTrainLoss = evaluate(model, trainEvalInputs, trainEvalTargets);
        double initialValidationLoss = evaluate(model, validationInputs, validationTargets);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Establish the starting point");
            System.out.printf("Model parameters: %d x %d = %d scores%n",
                    vocabulary.size(), vocabulary.size(), vocabulary.size() * vocabulary.size());
            System.out.printf("Evaluation batch shapes: x/y = [%d, %d]%n", BATCH_SIZE, BLOCK_SIZE);
            System.out.printf("Initial fixed-batch loss: train %.4f; validation %.4f%n",
                    initialTrainLoss, initialValidationLoss);
            System.out.println("The same evaluation batches are reused for before/after comparisons.");
        }

        if (steps[1] >= 2) {
            if (steps[0] <= 2) {
                System.out.println("Step 2 - Train the table with AdamW for 100 mini-batches");
                System.out.println("AdamW keeps moving averages of gradients and squared gradients.");
                System.out.println("step | train loss | validation loss");
            }

            Random trainingRandom = new Random(2025);
            int[][] batchInputs = new int[BATCH_SIZE][BLOCK_SIZE];
            int[][] batchTargets = new int[BATCH_SIZE][BLOCK_SIZE];
            for (int step = 0; step <= TRAINING_STEPS; step++) {
                if (steps[0] <= 2 && step % 25 == 0) {
                    printLossLine(step,
                            evaluate(model, trainEvalInputs, trainEvalTargets),
                            evaluate(model, validationInputs, validationTargets));
                }
                if (step == TRAINING_STEPS) {
                    break;
                }

                sampleBatch(trainData, trainingRandom, batchInputs, batchTargets);
                model.trainBatch(batchInputs, batchTargets, LEARNING_RATE, WEIGHT_DECAY);
            }
        }

        if (steps[1] >= 3) {
            int[] generatedIds = model.generate(0, 200);
            System.out.println("Step 3 - Sample text from the trained bigram model");
            System.out.printf("Start token ID: 0; generated text:%n%s%n",
                    vocabulary.decode(generatedIds));
            System.out.println("The model now prefers observed character pairs, but remembers only one character.");
        }
    }

    private static double evaluate(BigramLanguageModel model, int[][] inputs, int[][] targets) {
        return model.crossEntropy(model.forward(inputs), targets);
    }

    private static void printLossLine(int step, double trainLoss, double validationLoss) {
        System.out.printf("%4d | %10.4f | %15.4f%n", step, trainLoss, validationLoss);
    }

    private static void sampleBatch(int[] source, Random random, int[][] inputs, int[][] targets) {
        for (int batch = 0; batch < inputs.length; batch++) {
            int start = random.nextInt(source.length - BLOCK_SIZE);
            System.arraycopy(source, start, inputs[batch], 0, BLOCK_SIZE);
            System.arraycopy(source, start + 1, targets[batch], 0, BLOCK_SIZE);
        }
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException(
                    "Usage: BigramTrainingLesson [all|1|2|3|1-2|1-3]");
        }
        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "1-2" -> new int[] { 1, 2 };
            case "all", "1-3" -> new int[] { 1, 3 };
            default -> throw new IllegalArgumentException(
                    "Usage: BigramTrainingLesson [all|1|2|3|1-2|1-3]");
        };
    }
}
