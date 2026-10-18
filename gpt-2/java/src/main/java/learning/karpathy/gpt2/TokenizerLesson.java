package learning.karpathy.gpt2;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;

public final class TokenizerLesson {
    private static final int GPT2_END_OF_TEXT_ID = 50_256;

    private TokenizerLesson() {}

    public static void main(String[] args) {
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out), true,
                StandardCharsets.UTF_8));
        String part = args.length == 1 ? args[0] : "all";
        if (args.length > 1 || !part.matches("all|1|2|3")) {
            System.err.println("Usage: TokenizerLesson [all|1|2|3]");
            System.exit(1);
        }

        // GPT-2's vocabulary and token IDs correspond to the r50k_base encoding.
        EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
        Encoding encoding = registry.getEncoding(EncodingType.R50K_BASE);

        if (part.equals("1") || part.equals("all")) {
            System.out.println("Step 1: turn a short English string into GPT-2 token IDs");
            showEncoding(encoding, "Hello, world!");
        }
        if (part.equals("2") || part.equals("all")) {
            System.out.println("Step 2: spaces and UTF-8 text affect the byte sequence");
            showEncoding(encoding, "hello");
            showEncoding(encoding, " hello");
            showEncoding(encoding, "Hola, ¿qué tal? 😊");
        }
        if (part.equals("3") || part.equals("all")) {
            System.out.println("Step 3: add the GPT-2 end-of-text ID between documents");
            showDocumentBoundary(encoding);
        }
    }

    private static void showEncoding(Encoding encoding, String text) {
        var tokenIds = encoding.encodeOrdinary(text);
        System.out.println("Text: " + text);
        System.out.println("IDs:  " + tokenIds);
        System.out.println("Decoded: " + encoding.decode(tokenIds));
        System.out.println();
    }

    private static void showDocumentBoundary(Encoding encoding) {
        String document = "The dragon learned Python.";
        IntArrayList documentIds = encoding.encodeOrdinary(document);
        IntArrayList sequence = new IntArrayList(documentIds.size() + 1);
        sequence.add(GPT2_END_OF_TEXT_ID);
        for (int index = 0; index < documentIds.size(); index++) {
            sequence.add(documentIds.get(index));
        }

        System.out.println("Document text: " + document);
        System.out.println("IDs including the boundary: " + sequence);
        System.out.println("ID 50256 is prepended by the data pipeline before ordinary document tokens.\n");
    }
}
