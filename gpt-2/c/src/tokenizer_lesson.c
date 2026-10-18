#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tiktoken.h>

enum { GPT2_END_OF_TEXT_ID = 50256 };

static bool show_encoding(CoreBPE *encoding, const char *text) {
    size_t token_count = 0;
    Rank *tokens = tiktoken_corebpe_encode_ordinary(encoding, text, &token_count);
    if (token_count > 0 && tokens == NULL) {
        fprintf(stderr, "The tokenizer could not encode the input.\n");
        return false;
    }

    printf("Text: %s\nIDs:  [", text);
    for (size_t i = 0; i < token_count; ++i) {
        printf("%s%" PRIu32, i == 0 ? "" : ", ", tokens[i]);
    }
    puts("]");

    char *decoded = tiktoken_corebpe_decode(encoding, tokens, token_count);
    if (decoded == NULL) {
        fprintf(stderr, "The tokenizer could not decode the token sequence.\n");
        tiktoken_free(tokens);
        return false;
    }
    printf("Decoded: %s\n\n", decoded);

    /* The library owns both returned buffers; free them with its matching API. */
    tiktoken_free(decoded);
    tiktoken_free(tokens);
    return true;
}

static bool show_document_boundary(CoreBPE *encoding) {
    const char *document = "The dragon learned Python.";
    size_t document_token_count = 0;
    Rank *document_tokens = tiktoken_corebpe_encode_ordinary(
        encoding, document, &document_token_count);
    if (document_token_count > 0 && document_tokens == NULL) {
        fprintf(stderr, "The tokenizer could not encode the document.\n");
        return false;
    }

    Rank *sequence = malloc((document_token_count + 1) * sizeof(*sequence));
    if (sequence == NULL) {
        fprintf(stderr, "Could not allocate the document token sequence.\n");
        tiktoken_free(document_tokens);
        return false;
    }
    sequence[0] = GPT2_END_OF_TEXT_ID;
    if (document_token_count > 0) {
        memcpy(sequence + 1, document_tokens,
            document_token_count * sizeof(*sequence));
    }

    printf("Document text: %s\nIDs including the boundary: [", document);
    for (size_t i = 0; i < document_token_count + 1; ++i) {
        printf("%s%" PRIu32, i == 0 ? "" : ", ", sequence[i]);
    }
    puts("]");
    puts("ID 50256 is prepended by the data pipeline before ordinary document tokens.\n");

    free(sequence);
    tiktoken_free(document_tokens);
    return true;
}

static bool show_part(CoreBPE *encoding, const char *part) {
    if (strcmp(part, "1") == 0 || strcmp(part, "all") == 0) {
        puts("Step 1: turn a short English string into GPT-2 token IDs");
        if (!show_encoding(encoding, "Hello, world!")) {
            return false;
        }
    }
    if (strcmp(part, "2") == 0 || strcmp(part, "all") == 0) {
        puts("Step 2: spaces and UTF-8 text affect the byte sequence");
        if (!show_encoding(encoding, "hello")) {
            return false;
        }
        if (!show_encoding(encoding, " hello")) {
            return false;
        }
        if (!show_encoding(encoding, "Hola, ¿qué tal? 😊")) {
            return false;
        }
    }
    if (strcmp(part, "3") == 0 || strcmp(part, "all") == 0) {
        puts("Step 3: add the GPT-2 end-of-text ID between documents");
        if (!show_document_boundary(encoding)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    const char *part = argc == 2 ? argv[1] : "all";
    if (argc > 2 || (strcmp(part, "all") != 0 && strcmp(part, "1") != 0
            && strcmp(part, "2") != 0 && strcmp(part, "3") != 0)) {
        fprintf(stderr, "Usage: tokenizer_lesson [all|1|2|3]\n");
        return 1;
    }

    CoreBPE *encoding = tiktoken_get_bpe_from_model("gpt2");
    if (encoding == NULL) {
        fprintf(stderr, "Could not load the GPT-2 byte-pair encoding.\n");
        return 1;
    }

    const bool succeeded = show_part(encoding, part);
    tiktoken_destroy_corebpe(encoding);
    return succeeded ? 0 : 1;
}
