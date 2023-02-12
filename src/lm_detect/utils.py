def group_subwords(encodings):
    grouped_token_indices = []
    for word_id in encodings.word_ids():
        if word_id is not None:
            start, end = encodings.word_to_tokens(word_id)
            if start == end - 1:
                tokens = [start]
            else:
                tokens = [start, end-1]
            if len(grouped_token_indices) == 0 or grouped_token_indices[-1] != tokens:
                grouped_token_indices.append(tokens)
    return grouped_token_indices # list of lists of subword indices

def decode_words(encodings, tokenizer):
    words_token_indices = group_subwords(encodings)
    # get encodings.input_ids from encodings corresponding to the token_word
    word_input_ids = []
    for subword_idxs in words_token_indices:
        word_input_ids.append(encodings.input_ids[0][subword_idxs[0]:subword_idxs[-1]+1])
    return {
        'words': tokenizer.batch_decode(word_input_ids),
        'words_token_indices': words_token_indices,
    }

#print(decode_words(encodings, tokenizer))

def encodings_words_indexof(encodings, needle: str or list, tokenizer):
    words = decode_words(encodings, tokenizer)
    matches = {}
    for i, w in enumerate(words['words']):
        if isinstance(needle, list):
            for n in needle:
                if n in w:
                    matches.setdefault(n, [])
                    matches[n].append(words['words_token_indices'][i])
        else:
            if needle in w:
                matches.setdefault(needle, [])
                matches[needle].append(words['words_token_indices'][i])
    return matches
