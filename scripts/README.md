


# Trivial N-grams

This script computes the most frequent n-grams in a dataset. It produces a pickle file with the n-grams and their counts.

```` bash
python scripts/trivial_ngrams.py \
        --dataset_name andstor/the_pile_github \
        --dataset_config_name python \
        --text_column_name text \
        --tokenizer_name gpt2 \
        --preprocessing_num_workers 10 \
        --max_ngram 4 \
        --num_top_ngrams 1000 \
        --dataset_split train \
        --output_dir tmp
```