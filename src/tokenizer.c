#include "tokenizer.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <errno.h>

/*
============================================================
GGUF readers
============================================================
*/

static uint8_t read_u8(FILE *f)
{
    uint8_t x;
    if(fread(&x,sizeof(x),1,f)!=1)
    {
        fprintf(stderr,"Failed reading u8\n");
        exit(1);
    }
    return x;
}

static uint32_t read_u32(FILE *f)
{
    uint32_t x;
    if(fread(&x,sizeof(x),1,f)!=1)
    {
        fprintf(stderr,"Failed reading u32\n");
        exit(1);
    }
    return x;
}

static uint64_t read_u64(FILE *f)
{
    uint64_t x;
    if(fread(&x,sizeof(x),1,f)!=1)
    {
        fprintf(stderr,"Failed reading u64\n");
        exit(1);
    }
    return x;
}

static int32_t read_i32(FILE *f)
{
    int32_t x;
    if(fread(&x,sizeof(x),1,f)!=1)
    {
        fprintf(stderr,"Failed reading i32\n");
        exit(1);
    }
    return x;
}

static char *read_string(FILE *f)
{
    uint64_t len =
        read_u64(f);
    char *s =
        malloc(len+1);
    if(!s)
    {
        fprintf(stderr,"Out of memory\n");
        exit(1);
    }
    if (fread(s, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "Failed reading string data\n");
        free(s);
        exit(1);
    }
    s[len] = '\0';
    return s;
}

static char *dup_string(const char *s)
{
    size_t len=strlen(s);
    char *out=malloc(len+1);
    if(!out)
    {
        fprintf(stderr,"Out of memory\n");
        exit(1);
    }
    memcpy(out,s,len+1);
    return out;
}

/*
============================================================
Metadata skipping
============================================================
*/

/* Lock persistent tokenizer strings; report only failures. */

static void lock_tokenizer_string(const char *s)
{
    if (!s)
    return;
    size_t bytes = strlen(s) + 1;
    (void)mlock(s, bytes);
}

static void skip_value(
    FILE *f,
    uint32_t type
)
{
    switch(type)
    {
        case 0:
        case 1:
        case 7:
            fseek(f,1,SEEK_CUR);
            break;
        case 2:
        case 3:
            fseek(f,2,SEEK_CUR);
            break;
        case 4:
        case 5:
        case 6:
            fseek(f,4,SEEK_CUR);
            break;
        case 8:
        {
            uint64_t len=read_u64(f);
            fseek(
                f,
                len,
                SEEK_CUR
            );
            break;
        }
        case 9:
        {
            uint32_t element_type =
                read_u32(f);
            uint64_t count =
                read_u64(f);
            for(uint64_t i=0;i<count;i++)
                skip_value(
                    f,
                    element_type
                );
            break;
        }
        case 10:
        case 11:
        case 12:
            fseek(
                f,
                8,
                SEEK_CUR
            );
            break;
        default:
            fprintf(
                stderr,
                "Unknown metadata type %u\n",
                type
            );
            exit(1);
    }
}

/*
============================================================
Arrays
============================================================
*/

static void load_string_array(
    FILE *f,
    char ***out,
    uint64_t *count
)
{
    uint32_t element_type =
        read_u32(f);
    if(element_type!=8)
    {
        fprintf(stderr,
            "Expected string array\n");
        exit(1);
    }
    uint64_t n =
        read_u64(f);
    char **items =
        calloc(
            n,
            sizeof(char*)
        );
    for(uint64_t i=0;i<n;i++)
        items[i]=read_string(f);
    *out=items;
    *count=n;
}

static void load_i32_array(
    FILE *f,
    int32_t **out,
    uint64_t *count
)
{
    uint32_t element_type =
        read_u32(f);
    uint64_t n =
        read_u64(f);
    int32_t *items =
        calloc(
            n,
            sizeof(int32_t)
        );
    for(uint64_t i=0;i<n;i++)
        items[i]=read_i32(f);
    *out=items;
    *count=n;
}

/*
============================================================
Tokenizer metadata loader
============================================================
*/

static const char *const k_special_names[] = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"
};

static int is_special_candidate(const Tokenizer *tok, uint64_t i)
{
    const char *s = tok->tokens[i].text;
    if (!s)
        return 0;
    size_t n = strlen(s);
    if (n < 3)
        return 0;
    for (size_t k = 0; k < sizeof(k_special_names) / sizeof(k_special_names[0]); k++)
        if (strcmp(s, k_special_names[k]) == 0)
            return 1;
    return (tok->tokens[i].type == 3 || tok->tokens[i].type == 4) &&
           s[0] == '<' && s[n - 1] == '>';
}

static int32_t special_lookup(const Tokenizer *tok, const char *name)
{
    for (size_t k = 0; k < tok->special_count; k++)
        if (strcmp(tok->tokens[tok->special_ids[k]].text, name) == 0)
            return tok->special_ids[k];
    return -1;
}

static void build_special_table(Tokenizer *tok)
{
    tok->im_start_id = tok->im_end_id = tok->endoftext_id = -1;
    tok->think_open_id = tok->think_close_id = -1;
    tok->tool_open_id = tok->tool_close_id = -1;
    tok->special_ids = NULL;
    tok->special_count = 0;
    if (!tok->tokens)
        return;
    size_t count = 0;
    for (uint64_t i = 0; i < tok->vocab_size; i++)
        if (is_special_candidate(tok, i))
            count++;
    tok->special_ids = malloc((count ? count : 1) * sizeof(int32_t));
    if (!tok->special_ids)
        return;
    for (uint64_t i = 0; i < tok->vocab_size; i++)
        if (is_special_candidate(tok, i))
            tok->special_ids[tok->special_count++] = (int32_t)i;
    tok->im_start_id    = special_lookup(tok, "<|im_start|>");
    tok->im_end_id      = special_lookup(tok, "<|im_end|>");
    tok->endoftext_id   = special_lookup(tok, "<|endoftext|>");
    tok->think_open_id  = special_lookup(tok, "<think>");
    tok->think_close_id = special_lookup(tok, "</think>");
    tok->tool_open_id   = special_lookup(tok, "<tool_call>");
    tok->tool_close_id  = special_lookup(tok, "</tool_call>");
}

int tokenizer_load_metadata(
    FILE *f,
    uint64_t metadata_count,
    Tokenizer *tok
)
{
    tok->bos_id = tok->eos_id = tok->pad_id = -1;
    for(uint64_t i=0;i<metadata_count;i++)
    {
        char *key =
            read_string(f);
        uint32_t type =
            read_u32(f);
        if(strcmp(
            key,
            "tokenizer.ggml.tokens"
        )==0)
        {
            char **tokens;
            uint64_t count;
            load_string_array(
                f,
                &tokens,
                &count
            );
            tok->vocab_size =
                count;
            if (count > SIZE_MAX / sizeof(Token)) {
                fprintf(stderr, "[TOKENIZER] ERROR: token array size overflow\n");
                exit(1);
            }
            size_t token_array_bytes = (size_t)count * sizeof(Token);
            tok->tokens = calloc(count, sizeof(Token));
            if (!tok->tokens) {
                fprintf(stderr, "Tokenizer: token array allocation failed\n");
                exit(1);
            }
            (void)mlock(tok->tokens, token_array_bytes);
            for(uint64_t j=0;j<count;j++)
            {
                tok->tokens[j].text = tokens[j];
                lock_tokenizer_string(tok->tokens[j].text);
            }
            free(tokens);
        }
        else if(strcmp(
            key,
            "tokenizer.ggml.token_type"
        )==0)
        {
            int32_t *types;
            uint64_t count;
            load_i32_array(
                f,
                &types,
                &count
            );
            for(uint64_t j=0;
                j<count && j<tok->vocab_size;
                j++)
            {
                tok->tokens[j].type =
                    types[j];
            }
            free(types);
        }
        else if(strcmp(
            key,
            "tokenizer.ggml.merges"
        )==0)
        {
            char **merges;
            uint64_t count;
            load_string_array(
                f,
                &merges,
                &count
            );
            tok->merge_count =
                count;
            if (count > SIZE_MAX / sizeof(Merge)) {
                fprintf(stderr, "[TOKENIZER] ERROR: merge array size overflow\n");
                exit(1);
            }
            size_t merge_array_bytes = (size_t)count * sizeof(Merge);
            tok->merges = calloc(count, sizeof(Merge));
            if (!tok->merges) {
                fprintf(stderr, "Tokenizer: merge array allocation failed\n");
                exit(1);
            }
            (void)mlock(tok->merges, merge_array_bytes);
            for(uint64_t j=0;j<count;j++)
            {
                char *space =
                    strchr(
                        merges[j],
                        ' '
                    );
                if(space)
                {
                    *space=0;
                    tok->merges[j].left =
                        dup_string(
                            merges[j]
                        );
                    tok->merges[j].right =
                        dup_string(
                            space+1
                        );
                    lock_tokenizer_string(tok->merges[j].left);
                    lock_tokenizer_string(tok->merges[j].right);
                }
                free(merges[j]);
            }
            free(merges);
        }
        else if(strcmp(
            key,
            "tokenizer.ggml.bos_token_id"
        )==0)
        {
            tok->bos_id =
                read_i32(f);
        }
        else if(strcmp(
            key,
            "tokenizer.ggml.eos_token_id"
        )==0)
        {
            tok->eos_id =
                read_i32(f);
        }
        else if(strcmp(
            key,
            "tokenizer.ggml.padding_token_id"
        )==0)
        {
            tok->pad_id =
                read_i32(f);
        }
        else
        {
            skip_value(
                f,
                type
            );
        }
        free(key);
    }
    build_special_table(tok);
    return 0;
}

/*
============================================================
Vocabulary lookup
============================================================
*/

static int find_token(
    const Tokenizer *tok,
    const char *piece
)
{
    for(uint64_t i=0;i<tok->vocab_size;i++)
    {
        if(strcmp(
            tok->tokens[i].text,
            piece
        )==0)
        {
            return i;
        }
    }
    return -1;
}

/*
============================================================
Byte level helpers
============================================================
*/

static int utf8_encode(
    uint32_t cp,
    char out[5]
)
{
    if(cp <= 0x7F)
    {
        out[0]=cp;
        out[1]=0;
        return 1;
    }
    else if(cp <= 0x7FF)
    {
        out[0]=0xC0 | (cp>>6);
        out[1]=0x80 | (cp&0x3F);
        out[2]=0;
        return 2;
    }
    else if(cp <= 0xFFFF)
    {
        out[0]=0xE0 | (cp>>12);
        out[1]=0x80 | ((cp>>6)&0x3F);
        out[2]=0x80 | (cp&0x3F);
        out[3]=0;
        return 3;
    }
    else
    {
        out[0]=0xF0 | (cp>>18);
        out[1]=0x80 | ((cp>>12)&0x3F);
        out[2]=0x80 | ((cp>>6)&0x3F);
        out[3]=0x80 | (cp&0x3F);
        out[4]=0;
        return 4;
    }
}

/*
============================================================
BPE merge search
============================================================
*/

static int merge_rank(
    const Tokenizer *tok,
    const char *a,
    const char *b
)
{
    for(uint64_t i=0;i<tok->merge_count;i++)
    {
        if(strcmp(
            tok->merges[i].left,
            a
        )==0 &&
        strcmp(
            tok->merges[i].right,
            b
        )==0)
        {
            return i;
        }
    }
    return -1;
}
typedef struct
{
    char *text;
} BPEPiece;

static uint32_t byte_to_unicode_codepoint(uint8_t b)
{
    /*
    * Directly represented bytes.
    *
    * ! through \\\~
    * ¡ through ¬
    * ® through ÿ
    */
    if ((b >= 33 && b <= 126) ||
        (b >= 161 && b <= 172) ||
        (b >= 174 && b <= 255)) {
        return b;
    }
    /*
    * GPT-2 assigns the remaining bytes to unicode
    * codepoints starting at U+0100.
    */
    static int initialized = 0;
    static uint32_t table[256];
    if (!initialized) {
        for (int i = 0; i < 256; i++)
            table[i] = 0;
        uint32_t n = 0;
        for (uint32_t b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) ||
                (b >= 161 && b <= 172) ||
                (b >= 174 && b <= 255)) {
                table[b] = b;
            } else {
                table[b] = 256 + n;
                n++;
            }
        }
        initialized = 1;
    }
    return table[b];
}

static char *byte_to_bpe_symbol(uint8_t b)
{
    char tmp[5];
    uint32_t cp =
        byte_to_unicode_codepoint(b);
    size_t len =
        utf8_encode(cp, tmp);
    char *s =
        malloc(len + 1);
    if (!s) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    memcpy(s, tmp, len);
    s[len] = '\0';
    return s;
}

/*
============================================================
BPE encoder
============================================================
*/

size_t tokenizer_encode(
    const Tokenizer *tok,
    const char *text,
    int32_t **out_ids)
{
    *out_ids = NULL;
    size_t text_len = strlen(text);
    if (text_len == 0)
        return 0;
    /*
    * --------------------------------------------------------
    * Step 1:
    *
    * Qwen-style pre-tokenization.
    *
    * This is the ASCII/common-text implementation.
    *
    * We deliberately keep punctuation, words and whitespace
    * as separate chunks so BPE cannot merge across them.
    * --------------------------------------------------------
    */
    BPEPiece *pieces =
        calloc(text_len + 1, sizeof(BPEPiece));
    if (!pieces) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    size_t piece_count = 0;
    size_t pos = 0;
    while (pos < text_len) {
        unsigned char c =
            (unsigned char)text[pos];
        /*
        * Newline sequence.
        */
        if (c == '\n' || c == '\r') {
            size_t start = pos;
            while (pos < text_len &&
                (text[pos] == '\n' ||
                    text[pos] == '\r')) {
                pos++;
            }
            size_t len = pos - start;
            /*
            * Byte-level encode this chunk.
            */
            size_t capacity = len * 4 + 1;
            char *mapped =
                malloc(capacity);
            if (!mapped) {
                fprintf(stderr, "Out of memory\n");
                exit(1);
            }
            size_t out = 0;
            for (size_t i = start; i < pos; i++) {
                char *sym =
                    byte_to_bpe_symbol(
                        (uint8_t)text[i]
                    );
                size_t n = strlen(sym);
                memcpy(mapped + out, sym, n);
                out += n;
                free(sym);
            }
            mapped[out] = '\0';
            pieces[piece_count++].text = mapped;
            continue;
        }
        /*
        * Whitespace.
        *
        * Keep a leading space attached to the following
        * word. This produces things like:
        *
        *     " world" -> "Ġworld"
        */
        if (c == ' ' || c == '\t') {
            size_t start = pos;
            while (pos < text_len &&
                (text[pos] == ' ' ||
                    text[pos] == '\t')) {
                pos++;
            }
            /*
            * If whitespace is followed by normal text,
            * attach it to that following chunk.
            */
            size_t word_start = pos;
            while (pos < text_len) {
                unsigned char x =
                    (unsigned char)text[pos];
                if (x == ' ' ||
                    x == '\t' ||
                    x == '\n' ||
                    x == '\r')
                    break;
                /*
                * ASCII punctuation ends this chunk.
                */
                if (x < 128 &&
                    strchr(
                        ".,!?;:()[]{}\"'<>/\\|@#$%*^&+=~`",
                        x
                    )) {
                    break;
                }
                pos++;
            }
            if (word_start > start) {
                size_t len = pos - start;
                char *mapped =
                    malloc(len * 4 + 1);
                if (!mapped) {
                    fprintf(stderr, "Out of memory\n");
                    exit(1);
                }
                size_t out = 0;
                for (size_t i = start; i < pos; i++) {
                    char *sym =
                        byte_to_bpe_symbol(
                            (uint8_t)text[i]
                        );
                    size_t n = strlen(sym);
                    memcpy(mapped + out, sym, n);
                    out += n;
                    free(sym);
                }
                mapped[out] = '\0';
                pieces[piece_count++].text = mapped;
                continue;
            }
            /*
            * Trailing whitespace.
            */
            size_t len = pos - start;
            char *mapped =
                malloc(len * 4 + 1);
            if (!mapped) {
                fprintf(stderr, "Out of memory\n");
                exit(1);
            }
            size_t out = 0;
            for (size_t i = start; i < pos; i++) {
                char *sym =
                    byte_to_bpe_symbol(
                        (uint8_t)text[i]
                    );
                size_t n = strlen(sym);
                memcpy(mapped + out, sym, n);
                out += n;
                free(sym);
            }
            mapped[out] = '\0';
            pieces[piece_count++].text = mapped;
            continue;
        }
        /*
        * ASCII punctuation.
        */
        if (c < 128 &&
            strchr(
                ".,!?;:()[]{}\"'<>/\\|@#$%*^&+=~`",
                c
            )) {
            char *mapped =
                byte_to_bpe_symbol(c);
            pieces[piece_count++].text = mapped;
            pos++;
            continue;
        }
        /*
        * Normal text.
        *
        * Consume until whitespace or punctuation.
        *
        * UTF-8 bytes are kept together here; the subsequent
        * byte-level conversion handles the actual bytes.
        */
        size_t start = pos;
        while (pos < text_len) {
            unsigned char x =
                (unsigned char)text[pos];
            if (x == ' ' ||
                x == '\t' ||
                x == '\n' ||
                x == '\r')
                break;
            if (x < 128 &&
                strchr(
                    ".,!?;:()[]{}\"'<>/\\|@#$%*^&+=~`",
                    x
                ))
                break;
            pos++;
        }
        size_t len = pos - start;
        char *mapped =
            malloc(len * 4 + 1);
        if (!mapped) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
        size_t out = 0;
        for (size_t i = start; i < pos; i++) {
            char *sym =
                byte_to_bpe_symbol(
                    (uint8_t)text[i]
                );
            size_t n = strlen(sym);
            memcpy(mapped + out, sym, n);
            out += n;
            free(sym);
        }
        mapped[out] = '\0';
        pieces[piece_count++].text = mapped;
    }
    /*
    * --------------------------------------------------------
    * Step 2:
    *
    * BPE.
    *
    * Each pre-tokenized chunk is merged independently.
    * --------------------------------------------------------
    */
    BPEPiece *final_pieces =
        calloc(text_len * 2 + 1, sizeof(BPEPiece));
    if (!final_pieces) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    size_t final_count = 0;
    for (size_t p = 0; p < piece_count; p++) {
        char *chunk = pieces[p].text;
        /*
        * Split the byte-level UTF-8 representation into
        * Unicode symbols.
        *
        * For the GPT-2 byte mapping, every original byte
        * becomes exactly one UTF-8 symbol.
        */
        size_t chunk_len = strlen(chunk);
        BPEPiece *work =
            calloc(chunk_len + 1, sizeof(BPEPiece));
        if (!work) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
        size_t work_count = 0;
        for (size_t i = 0; i < chunk_len;) {
            unsigned char c =
                (unsigned char)chunk[i];
            size_t n;
            if (c < 0x80)
                n = 1;
            else if ((c & 0xE0) == 0xC0)
                n = 2;
            else if ((c & 0xF0) == 0xE0)
                n = 3;
            else
                n = 4;
            work[work_count].text =
                malloc(n + 1);
            if (!work[work_count].text) {
                fprintf(stderr, "Out of memory\n");
                exit(1);
            }
            memcpy(
                work[work_count].text,
                chunk + i,
                n
            );
            work[work_count].text[n] = '\0';
            work_count++;
            i += n;
        }
        /*
        * Repeatedly apply the highest priority merge.
        */
        while (work_count > 1) {
            int best_rank = -1;
            size_t best_pos = 0;
            for (size_t i = 0;
                i + 1 < work_count;
                i++) {
                int rank =
                    merge_rank(
                        tok,
                        work[i].text,
                        work[i + 1].text
                    );
                if (rank >= 0 &&
                    (best_rank < 0 ||
                    rank < best_rank)) {
                    best_rank = rank;
                    best_pos = i;
                }
            }
            if (best_rank < 0)
                break;
            size_t left_len =
                strlen(work[best_pos].text);
            size_t right_len =
                strlen(work[best_pos + 1].text);
            char *merged =
                malloc(left_len + right_len + 1);
            if (!merged) {
                fprintf(stderr, "Out of memory\n");
                exit(1);
            }
            memcpy(
                merged,
                work[best_pos].text,
                left_len
            );
            memcpy(
                merged + left_len,
                work[best_pos + 1].text,
                right_len
            );
            merged[left_len + right_len] = '\0';
            free(work[best_pos].text);
            free(work[best_pos + 1].text);
            work[best_pos].text = merged;
            for (size_t i = best_pos + 1;
                i + 1 < work_count;
                i++) {
                work[i] = work[i + 1];
            }
            work_count--;
        }
        /*
        * Move final BPE pieces into global list.
        */
        for (size_t i = 0; i < work_count; i++) {
            final_pieces[final_count++] =
                work[i];
        }
        free(work);
        free(chunk);
    }
    free(pieces);
    /*
    * --------------------------------------------------------
    * Step 3:
    *
    * Convert BPE pieces to vocabulary IDs.
    * --------------------------------------------------------
    */
    if (final_count == 0) {
        free(final_pieces);
        return 0;
    }
    if (final_count > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "Tokenizer: encoded ID vector size overflow\n");
        for (size_t i = 0; i < final_count; i++)
            free(final_pieces[i].text);
        free(final_pieces);
        return 0;
    }
    size_t ids_bytes = final_count * sizeof(int32_t);
    int32_t *ids = malloc(ids_bytes);
    if (!ids) {
        fprintf(stderr, "Tokenizer: encoded ID vector allocation failed (%zu bytes)\n", ids_bytes);
        for (size_t i = 0; i < final_count; i++)
            free(final_pieces[i].text);
        free(final_pieces);
        return 0;
    }
    size_t output_count = 0;
    for (size_t i = 0; i < final_count; i++) {
        int id =
            find_token(
                tok,
                final_pieces[i].text
            );
        if (id < 0) {
            fprintf(
                stderr,
                "Tokenizer: no vocabulary entry for piece "
                "\"%s\"\n",
                final_pieces[i].text
            );
            /*
            * Clean up exactly once.
            */
            for (size_t j = i;
                j < final_count;
                j++) {
                free(final_pieces[j].text);
            }
            /*
            * Pieces before i were already consumed below
            * in the normal path, so don't free them again.
            */
            free(ids);
            free(final_pieces);
            *out_ids = NULL;
            return 0;
        }
        ids[output_count++] = id;
        free(final_pieces[i].text);
    }
    free(final_pieces);
    *out_ids = ids;
    return output_count;
}

/*
============================================================
Decode
============================================================
*/

static int bpe_symbol_to_byte(
    uint32_t cp,
    uint8_t *out
)
{
    /*
    * GPT-2 byte -> unicode mapping is:
    *
    * Printable bytes stay unchanged.
    * Remaining bytes map to U+0100 + n.
    */
    if ((cp >= 33 && cp <= 126) ||
        (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) {
        *out = (uint8_t)cp;
        return 1;
    }
    if (cp >= 256 && cp <= 511) {
        static int initialized = 0;
        static uint8_t inverse[256];
        if (!initialized) {
            uint32_t n = 0;
            for (uint32_t b = 0; b < 256; b++) {
                if ((b >= 33 && b <= 126) ||
                    (b >= 161 && b <= 172) ||
                    (b >= 174 && b <= 255)) {
                    continue;
                }
                inverse[n++] = (uint8_t)b;
            }
            initialized = 1;
        }
        *out = inverse[cp - 256];
        return 1;
    }
    return 0;
}

static size_t utf8_decode_cp(
    const char *s,
    uint32_t *cp
)
{
    const unsigned char *p =
        (const unsigned char *)s;
    if (p[0] < 0x80) {
        *cp = p[0];
        return 1;
    }
    if ((p[0] & 0xE0) == 0xC0) {
        *cp =
            ((uint32_t)(p[0] & 0x1F) << 6) |
            (uint32_t)(p[1] & 0x3F);
        return 2;
    }
    if ((p[0] & 0xF0) == 0xE0) {
        *cp =
            ((uint32_t)(p[0] & 0x0F) << 12) |
            ((uint32_t)(p[1] & 0x3F) << 6) |
            (uint32_t)(p[2] & 0x3F);
        return 3;
    }
    *cp =
        ((uint32_t)(p[0] & 0x07) << 18) |
        ((uint32_t)(p[1] & 0x3F) << 12) |
        ((uint32_t)(p[2] & 0x3F) << 6) |
        (uint32_t)(p[3] & 0x3F);
    return 4;
}

char *tokenizer_decode(
    const Tokenizer *tok,
    const int32_t *ids,
    size_t count
)
{
    if (!tok || !ids)
        return NULL;
    /*
    * Allocate generously because each token may contain
    * several UTF-8 encoded GPT-2 byte symbols.
    */
    size_t capacity = 4096;
    size_t length = 0;
    char *out = malloc(capacity);
    if (!out)
        return NULL;
    for (size_t i = 0; i < count; i++) {
        int32_t id = ids[i];
        if (id < 0 ||
            (uint64_t)id >= tok->vocab_size)
            continue;
        const char *token =
            tok->tokens[id].text;
        /*
        * Preserve control / special tokens literally.
        */
        if (tok->tokens[id].type == 3 ||
            tok->tokens[id].type == 4 ||
            tok->tokens[id].type == 5) {
            size_t n = strlen(token);
            if (length + n + 1 > capacity) {
                while (length + n + 1 > capacity)
                    capacity *= 2;
                char *tmp =
                    realloc(out, capacity);
                if (!tmp) {
                    free(out);
                    return NULL;
                }
                out = tmp;
            }
            memcpy(
                out + length,
                token,
                n
            );
            length += n;
            continue;
        }
        /*
        * Byte tokens such as <0xAB>.
        */
        if (strncmp(token, "<0x", 3) == 0 &&
            strlen(token) == 6 &&
            token[5] == '>') {
            unsigned int byte;
            if (sscanf(token + 3, "%2x", &byte) == 1) {
                if (length + 2 > capacity) {
                    capacity *= 2;
                    char *tmp =
                        realloc(out, capacity);
                    if (!tmp) {
                        free(out);
                        return NULL;
                    }
                    out = tmp;
                }
                out[length++] =
                    (char)byte;
                continue;
            }
        }
        /*
        * GPT-2 byte-level decoding.
        *
        * Example:
        *
        *   Ġ  -> ' '
        *   Ċ  -> '\n'
        *   æĥ³ -> UTF-8 bytes
        */
        size_t pos = 0;
        size_t token_len = strlen(token);
        while (pos < token_len) {
            uint32_t cp;
            size_t n =
                utf8_decode_cp(
                    token + pos,
                    &cp
                );
            uint8_t byte;
            if (bpe_symbol_to_byte(cp, &byte)) {
                if (length + 2 > capacity) {
                    capacity *= 2;
                    char *tmp =
                        realloc(out, capacity);
                    if (!tmp) {
                        free(out);
                        return NULL;
                    }
                    out = tmp;
                }
                out[length++] =
                    (char)byte;
            } else {
                /*
                * Not a GPT-2 byte symbol.
                * Preserve the original UTF-8 bytes.
                */
                if (length + n + 1 > capacity) {
                    while (length + n + 1 > capacity)
                        capacity *= 2;
                    char *tmp =
                        realloc(out, capacity);
                    if (!tmp) {
                        free(out);
                        return NULL;
                    }
                    out = tmp;
                }
                memcpy(
                    out + length,
                    token + pos,
                    n
                );
                length += n;
            }
            pos += n;
        }
    }
    out[length] = '\0';
    return out;
}

/*
============================================================
Free
============================================================
*/

void tokenizer_free(
    Tokenizer *tok
)
{
    for(uint64_t i=0;i<tok->vocab_size;i++)
    {
        if (tok->tokens && tok->tokens[i].text) {
            size_t bytes = strlen(tok->tokens[i].text) + 1;
            (void)munlock(tok->tokens[i].text, bytes);
        }
        free(tok->tokens ? tok->tokens[i].text : NULL);
    }
    if (tok->tokens) {
        size_t bytes = (size_t)tok->vocab_size * sizeof(Token);
        if (bytes != 0)
            (void)munlock(tok->tokens, bytes);
        free(tok->tokens);
    }
    for(uint64_t i=0;i<tok->merge_count;i++)
    {
        if (tok->merges) {
            if (tok->merges[i].left) {
                size_t bytes = strlen(tok->merges[i].left) + 1;
                (void)munlock(tok->merges[i].left, bytes);
            }
            if (tok->merges[i].right) {
                size_t bytes = strlen(tok->merges[i].right) + 1;
                (void)munlock(tok->merges[i].right, bytes);
            }
            free(tok->merges[i].left);
            free(tok->merges[i].right);
        }
    }
    if (tok->merges) {
        size_t bytes = (size_t)tok->merge_count * sizeof(Merge);
        if (bytes != 0)
            (void)munlock(tok->merges, bytes);
        free(tok->merges);
    }
    free(tok->special_ids);
    memset(
        tok,
        0,
        sizeof(Tokenizer)
    );
}

/*
============================================================
Special tokens
============================================================
*/

int tokenizer_is_special(const Tokenizer *tok, int32_t id)
{
    if (!tok || !tok->special_ids)
        return 0;
    for (size_t k = 0; k < tok->special_count; k++)
        if (tok->special_ids[k] == id)
            return 1;
    return 0;
}

int tokenizer_is_stop(const Tokenizer *tok, int32_t id)
{
    if (!tok || id < 0)
        return 0;
    return id == tok->eos_id || id == tok->im_end_id || id == tok->endoftext_id;
}

static int ids_push(int32_t **ids, size_t *n, size_t *cap, const int32_t *src, size_t cnt)
{
    if (*n + cnt > *cap) {
        size_t nc = *cap ? *cap : 64;
        while (nc < *n + cnt)
            nc *= 2;
        int32_t *t = realloc(*ids, nc * sizeof(int32_t));
        if (!t)
            return -1;
        *ids = t;
        *cap = nc;
    }
    memcpy(*ids + *n, src, cnt * sizeof(int32_t));
    *n += cnt;
    return 0;
}

static int encode_segment(const Tokenizer *tok, const char *s, size_t len,
                          int32_t **ids, size_t *n, size_t *cap)
{
    if (len == 0)
        return 0;
    char *seg = malloc(len + 1);
    if (!seg)
        return -1;
    memcpy(seg, s, len);
    seg[len] = '\0';
    int32_t *part = NULL;
    size_t cnt = tokenizer_encode(tok, seg, &part);
    free(seg);
    if (!part || cnt == 0) {
        free(part);
        return -1;
    }
    int rc = ids_push(ids, n, cap, part, cnt);
    free(part);
    return rc;
}

size_t tokenizer_encode_special(const Tokenizer *tok, const char *text, int32_t **out_ids)
{
    *out_ids = NULL;
    if (!tok || !text)
        return 0;
    size_t len = strlen(text), seg = 0, pos = 0, n = 0, cap = 0;
    int32_t *ids = NULL;
    while (pos < len) {
        int32_t sid = -1;
        size_t slen = 0;
        if (text[pos] == '<') {                       /* longest special token starting here */
            for (size_t k = 0; k < tok->special_count; k++) {
                int32_t id = tok->special_ids[k];
                const char *s = tok->tokens[id].text;
                size_t l = strlen(s);
                if (l > slen && strncmp(text + pos, s, l) == 0) {
                    sid = id;
                    slen = l;
                }
            }
        }
        if (sid < 0) {
            pos++;
            continue;
        }
        if (encode_segment(tok, text + seg, pos - seg, &ids, &n, &cap) != 0 ||
            ids_push(&ids, &n, &cap, &sid, 1) != 0)
            goto fail;
        pos += slen;
        seg = pos;
    }
    if (encode_segment(tok, text + seg, len - seg, &ids, &n, &cap) != 0)
        goto fail;
    if (n == 0) {
        free(ids);
        return 0;
    }
    *out_ids = ids;
    return n;
fail:
    free(ids);
    return 0;
}