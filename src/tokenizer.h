#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef struct {
    char *text;
    int32_t type;
} Token;

typedef struct {
    char *left;
    char *right;
} Merge;

typedef struct {
    Token *tokens;
    uint64_t vocab_size;

    Merge *merges;
    uint64_t merge_count;

    int32_t eos_id;
    int32_t bos_id;
    int32_t pad_id;

    /* Special tokens (control / user-defined, e.g. <|im_start|>, <think>). -1 = not in vocab. */
    int32_t *special_ids;
    size_t special_count;
    int32_t im_start_id, im_end_id, endoftext_id;
    int32_t think_open_id, think_close_id;
    int32_t tool_open_id, tool_close_id;
} Tokenizer;

int tokenizer_load_metadata(
    FILE *f,
    uint64_t metadata_count,
    Tokenizer *tok
);

size_t tokenizer_encode(
    const Tokenizer *tok,
    const char *text,
    int32_t **out_ids
);

char *tokenizer_decode(
    const Tokenizer *tok,
    const int32_t *ids,
    size_t count
);

/* Like tokenizer_encode, but literal special-token text (e.g. "<|im_start|>") becomes the single
 * special token id. Only use it on trusted template text, never on user input. */
size_t tokenizer_encode_special(const Tokenizer *tok, const char *text, int32_t **out_ids);

int tokenizer_is_special(const Tokenizer *tok, int32_t id);
/* eos, <|im_end|> or <|endoftext|> */
int tokenizer_is_stop(const Tokenizer *tok, int32_t id);

void tokenizer_free(Tokenizer *tok);

#endif