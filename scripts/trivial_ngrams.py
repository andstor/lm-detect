import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import os
from collections import Counter
from nltk.util import ngrams
import pickle
import json
import logging

from accelerate.logging import get_logger
get_logger("transformers").setLevel(logging.ERROR)
logger = get_logger(__name__)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Do inference with a transformer model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="The name of the split to use.",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of the dataset to use.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_batch_size",  # "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached dataset"
    )
    parser.add_argument(
        "--num_top_ngrams",
        type=int,
        default=500,
        help="The k most common ngrams to return.",
    )
    parser.add_argument(
        "--max_ngram",
        type=int,
        default=3,
        help="The maximum ngram to calculate.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need a dataset name.")
    
    return args


def main():
    """
    Generate new data by sampling from the original data.
    """

    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    if args.seed is not None:
        set_seed(args.seed)



    # Write the generation config to disk
    if accelerator.is_main_process:
        if args.output_dir is not None:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Need a output directory.")
    
    accelerator.wait_for_everyone()

    # 
    # Load the dataset
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    #    dataset = raw_datasets.with_format("torch", columns=[text_column])

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, split=args.dataset_split)

    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing the datasets.
    column_names = raw_dataset.column_names

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    else:
        text_column_name = "text" if "text" in column_names else column_names[0]
        logger.warning(f"Using column {text_column_name} as text column.")
    

    def top_ngrams_function(examples):
        """Return the k most common n-grams in the corpus."""
        k=args.num_top_ngrams
        # Extract trivially shared n-grams
        top_ngrams = Counter()
        for example in examples["ngrams"]:
            unpic_example = pickle.loads(example)
            frequencies = Counter(unpic_example)
            #frequencies = Counter([tuple(gram) for gram in example]) # convert lists to tuples
            most_common = dict(frequencies.most_common(k))
            top_ngrams += Counter(most_common)
        most_common = dict(top_ngrams.most_common(k))
        pic_top_ngrams = pickle.dumps(most_common)
        #safe_top_ngrams = [[list(gram), c] for gram, c in top_ngrams] # convert tuples of tuple to lists of list
        return {"top_ngrams": [pic_top_ngrams]}

    def calc_ngrams_function(examples):
        max_ngram = args.max_ngram
        ngrams_list = []
        for example in examples["input_ids"]:
            all_ngrams = []
            for j in range(1, max_ngram+1):
                n_grams = list(ngrams(example, j))
                #safe_n_grams = [list(gram) for gram in n_grams] # convert tuples to lists
                #all_ngrams.extend(safe_n_grams)
                all_ngrams.extend(n_grams)
            pic_all_ngrams = pickle.dumps(all_ngrams)
            ngrams_list.append(pic_all_ngrams)
        examples["ngrams"] = ngrams_list
        return examples
    
    # Tokenize the data
    def tokenize_function(examples):        
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            keep_in_memory=False,
            desc="Running tokenizer on dataset",
        )
        ngrams_dataset = tokenized_dataset.map(
            calc_ngrams_function,
            batched=True,
            remove_columns=tokenized_dataset.column_names,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            keep_in_memory=False,
            desc=f"Calculating ngrams of order 1 to {args.max_ngram}",
        )
        
        top_ngrams_dataset = ngrams_dataset.map(
            top_ngrams_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=ngrams_dataset.column_names,
            keep_in_memory=False,
            desc="Calculating top ngrams",
        )


        top_ngrams = Counter()
        for example in top_ngrams_dataset["top_ngrams"]:
            unpic_count_dict = pickle.loads(example)
            #count_dict = {tuple(gram): c for gram, c in example} # convert lists of list to tuples of tuple
            top_ngrams += Counter(unpic_count_dict)

        # Decode the token ngrams to string ngrams
        encoded_dict = dict(top_ngrams.most_common(args.num_top_ngrams))
        tokens_dict = {}
        for token_ngram, count in encoded_dict.items():
            decoded_gram = tokenizer.convert_ids_to_tokens(token_ngram)
            tokens_dict[tuple(decoded_gram)] = count
        
        decoded_dict = {}
        for token_ngram, count in encoded_dict.items():
            decoded_gram = tokenizer.batch_decode(token_ngram)
            decoded_dict[tuple(decoded_gram)] = count
        
        # Save data to pickle
        output_file = Path(args.output_dir) / f"trivial_ngrams.ids.{args.dataset_split}.pkl"
        output_file.write_bytes(pickle.dumps(encoded_dict))

        output_file = Path(args.output_dir) / f"trivial_ngrams.tokens.{args.dataset_split}.pkl"
        output_file.write_bytes(pickle.dumps(tokens_dict))

        output_file = Path(args.output_dir) / f"trivial_ngrams.text.{args.dataset_split}.pkl"
        output_file.write_bytes(pickle.dumps(decoded_dict))

if __name__ == "__main__":
    main()
