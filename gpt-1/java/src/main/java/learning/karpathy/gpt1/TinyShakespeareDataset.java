package learning.karpathy.gpt1;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/** The text and source path for the first tutorial's training corpus. */
public final class TinyShakespeareDataset {
    private final Path source;
    private final String text;

    private TinyShakespeareDataset(Path source, String text) {
        this.source = source;
        this.text = text;
    }

    public static TinyShakespeareDataset load(Path source) throws IOException {
        String text = Files.readString(source, StandardCharsets.UTF_8);
        return new TinyShakespeareDataset(source, text);
    }

    public Path source() {
        return source;
    }

    public int characterCount() {
        return text.length();
    }

    public String preview(int characterLimit) {
        int end = Math.min(characterLimit, text.length());
        return text.substring(0, end);
    }
}
