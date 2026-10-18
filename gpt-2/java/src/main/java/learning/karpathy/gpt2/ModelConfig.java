package learning.karpathy.gpt2;

/** Dimensions only: constructing this record does not allocate a GPT-2 model. */
public record ModelConfig(int vocabSize, int contextLength, int layers,
                          int heads, int embeddingSize) {
    public ModelConfig {
        if (vocabSize <= 0 || contextLength <= 0 || layers <= 0
                || heads <= 0 || embeddingSize <= 0 || embeddingSize % heads != 0) {
            throw new IllegalArgumentException("Positive dimensions and embeddingSize divisible by heads are required");
        }
    }

    public int headSize() {
        return embeddingSize / heads;
    }

    public static ModelConfig tiny() {
        return new ModelConfig(256, 32, 2, 4, 32);
    }

    public static ModelConfig gpt2() {
        return new ModelConfig(50257, 1024, 12, 12, 768);
    }
}
