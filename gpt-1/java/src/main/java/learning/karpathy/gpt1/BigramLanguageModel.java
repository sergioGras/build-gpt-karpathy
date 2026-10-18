package learning.karpathy.gpt1;

import java.util.Random;

/** A table that scores the next character using only the current character. */
public final class BigramLanguageModel {
    private final int vocabularySize;
    private final float[][] tokenEmbeddingTable;
    private final float[][] firstMoment;
    private final float[][] secondMoment;
    private final Random random;
    private long optimizerStep;

    public BigramLanguageModel(int vocabularySize, long seed) {
        if (vocabularySize < 1) {
            throw new IllegalArgumentException("Vocabulary must not be empty.");
        }
        this.vocabularySize = vocabularySize;
        this.tokenEmbeddingTable = new float[vocabularySize][vocabularySize];
        this.firstMoment = new float[vocabularySize][vocabularySize];
        this.secondMoment = new float[vocabularySize][vocabularySize];
        this.random = new Random(seed);

        // Each row is one input character; each column scores a possible next character.
        for (int inputId = 0; inputId < vocabularySize; inputId++) {
            for (int nextId = 0; nextId < vocabularySize; nextId++) {
                tokenEmbeddingTable[inputId][nextId] = (float) random.nextGaussian();
            }
        }
    }

    /** Return logits with shape [batch][time][vocabulary]. */
    public float[][][] forward(int[][] inputIds) {
        float[][][] logits = new float[inputIds.length][][];
        for (int batch = 0; batch < inputIds.length; batch++) {
            logits[batch] = new float[inputIds[batch].length][];
            for (int time = 0; time < inputIds[batch].length; time++) {
                int inputId = inputIds[batch][time];
                checkTokenId(inputId);
                logits[batch][time] = tokenEmbeddingTable[inputId].clone();
            }
        }
        return logits;
    }

    /** Stable mean cross-entropy over all batch and time positions. */
    public double crossEntropy(float[][][] logits, int[][] targetIds) {
        if (logits.length == 0 || logits.length != targetIds.length) {
            throw new IllegalArgumentException("Logits and targets must have matching batch sizes.");
        }

        double totalLoss = 0.0;
        int positionCount = 0;
        for (int batch = 0; batch < logits.length; batch++) {
            if (logits[batch].length != targetIds[batch].length) {
                throw new IllegalArgumentException("Logits and targets must have matching time steps.");
            }
            for (int time = 0; time < logits[batch].length; time++) {
                int targetId = targetIds[batch][time];
                checkTokenId(targetId);
                float[] row = logits[batch][time];
                double maximum = row[0];
                for (int tokenId = 1; tokenId < vocabularySize; tokenId++) {
                    maximum = Math.max(maximum, row[tokenId]);
                }

                // log-sum-exp avoids taking log of tiny softmax probabilities.
                double exponentialSum = 0.0;
                for (float score : row) {
                    exponentialSum += Math.exp(score - maximum);
                }
                totalLoss += Math.log(exponentialSum) + maximum - row[targetId];
                positionCount++;
            }
        }
        if (positionCount == 0) {
            throw new IllegalArgumentException("At least one target position is required.");
        }
        return totalLoss / positionCount;
    }

    /**
     * One AdamW step. Unlike PyTorch's autograd, this method writes the
     * softmax-cross-entropy gradient for each selected lookup-table row itself.
     */
    public void trainBatch(int[][] inputIds, int[][] targetIds,
                           double learningRate, double weightDecay) {
        if (inputIds.length == 0 || inputIds.length != targetIds.length
                || learningRate <= 0.0 || weightDecay < 0.0) {
            throw new IllegalArgumentException("Inputs, targets, and optimizer settings are invalid.");
        }

        double[][] gradient = new double[vocabularySize][vocabularySize];
        int positionCount = 0;
        for (int batch = 0; batch < inputIds.length; batch++) {
            if (inputIds[batch].length == 0
                    || inputIds[batch].length != targetIds[batch].length) {
                throw new IllegalArgumentException("Input and target rows must have equal lengths.");
            }
            for (int time = 0; time < inputIds[batch].length; time++) {
                int inputId = inputIds[batch][time];
                int targetId = targetIds[batch][time];
                checkTokenId(inputId);
                checkTokenId(targetId);

                float[] scores = tokenEmbeddingTable[inputId];
                double maximum = scores[0];
                for (int tokenId = 1; tokenId < vocabularySize; tokenId++) {
                    maximum = Math.max(maximum, scores[tokenId]);
                }
                double exponentialSum = 0.0;
                for (float score : scores) {
                    exponentialSum += Math.exp(score - maximum);
                }

                // For softmax cross-entropy, dLoss/dLogit = p - one_hot(target).
                for (int nextId = 0; nextId < vocabularySize; nextId++) {
                    gradient[inputId][nextId] +=
                            Math.exp(scores[nextId] - maximum) / exponentialSum;
                }
                gradient[inputId][targetId] -= 1.0;
                positionCount++;
            }
        }

        double gradientScale = 1.0 / positionCount;
        optimizerStep++;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        double beta1Correction = 1.0 - Math.pow(beta1, optimizerStep);
        double beta2Correction = 1.0 - Math.pow(beta2, optimizerStep);

        for (int inputId = 0; inputId < vocabularySize; inputId++) {
            for (int nextId = 0; nextId < vocabularySize; nextId++) {
                double currentGradient = gradient[inputId][nextId] * gradientScale;
                // AdamW applies decoupled weight decay before its adaptive update.
                tokenEmbeddingTable[inputId][nextId] *= (float) (1.0 - learningRate * weightDecay);
                firstMoment[inputId][nextId] = (float) (
                        beta1 * firstMoment[inputId][nextId]
                                + (1.0 - beta1) * currentGradient);
                secondMoment[inputId][nextId] = (float) (
                        beta2 * secondMoment[inputId][nextId]
                                + (1.0 - beta2) * currentGradient * currentGradient);

                double correctedFirst = firstMoment[inputId][nextId] / beta1Correction;
                double correctedSecond = secondMoment[inputId][nextId] / beta2Correction;
                double update = learningRate * correctedFirst
                        / (Math.sqrt(correctedSecond) + epsilon);
                tokenEmbeddingTable[inputId][nextId] -= (float) update;
            }
        }
    }

    /** Sample next characters from the current character's softmax distribution. */
    public int[] generate(int startToken, int newTokenCount) {
        checkTokenId(startToken);
        int[] outputIds = new int[newTokenCount + 1];
        outputIds[0] = startToken;
        for (int generated = 0; generated < newTokenCount; generated++) {
            float[] scores = tokenEmbeddingTable[outputIds[generated]];
            double maximum = scores[0];
            for (int tokenId = 1; tokenId < vocabularySize; tokenId++) {
                maximum = Math.max(maximum, scores[tokenId]);
            }

            double probabilitySum = 0.0;
            for (float score : scores) {
                probabilitySum += Math.exp(score - maximum);
            }
            double draw = random.nextDouble() * probabilitySum;
            int chosenId = vocabularySize - 1;
            for (int tokenId = 0; tokenId < vocabularySize; tokenId++) {
                draw -= Math.exp(scores[tokenId] - maximum);
                if (draw <= 0.0) {
                    chosenId = tokenId;
                    break;
                }
            }
            outputIds[generated + 1] = chosenId;
        }
        return outputIds;
    }

    public int vocabularySize() {
        return vocabularySize;
    }

    private void checkTokenId(int tokenId) {
        if (tokenId < 0 || tokenId >= vocabularySize) {
            throw new IllegalArgumentException("Token ID outside the model vocabulary: " + tokenId);
        }
    }
}
