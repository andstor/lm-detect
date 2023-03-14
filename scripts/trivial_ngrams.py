import multiprocessing
import os
import time
from multiprocessing import Pool, Queue, Process, Manager
import random
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.util import ngrams
from collections import Counter
import multiprocessing
import multiprocessing.util
import logging
from threading import Thread
import pickle
import argparse
from pathlib import Path

#multiprocessing.util._logger = multiprocessing.util.log_to_stderr(logging.DEBUG)
logging.getLogger('transformers').setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
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
        "--revision",
        type=str,
        default="main",
        help="The specific revision of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default="text",
        help="The name of the text column in the dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="The number of processes to use for data loading.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="The number of sub samples to use for data loading.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp",
        help="The output directory where the trivial tokens will be saved.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_ngram",
        type=int,
        default=3,
        help="The maximum ngram to calculate.",
    )
    parser.add_argument(
        "--num_top_ngrams",
        type=int,
        default=500,
        help="The k most common ngrams to return.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Whether to stream the dataset.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="The name of the split to use.",
    )


    args = parser.parse_args()

    if not args.tokenizer_name:
        raise ValueError("You need to specify a tokenizer name")

    return args


def worker_main(queue_in, queue_out, args):
    print(os.getpid(),"starting worker")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    while True:
        item = queue_in.get(block=True) #block=True means make a blocking call to wait for items in queue
        if item is None:
            break

        ngrams = calc_ngrams(item, tokenizer, args.max_ngram)
        t_ngrams = trivial_ngrams(ngrams, args.num_top_ngrams)
        
        queue_out.put(t_ngrams)
    
    queue_out.put(None)


def calc_ngrams(text, tokenizer, max_ngram=3):
    tokenized = tokenizer.tokenize(text)
    all_ngrams = []
    for j in range(1, max_ngram+1):
        n_grams = list(ngrams(tokenized, j))
        all_ngrams.extend(n_grams)
    return all_ngrams

def trivial_ngrams(ngrams, k=500):
    """Return the k most common n-grams in the corpus."""
    # Extract trivially shared n-grams
    frequencies = Counter(ngrams) # tokenized_corpus is a list of strings
    return dict(frequencies.most_common(k))

def merge_dicts(*dict_args):
    """Sum all the values in the dictionaries."""
    result = Counter()
    for dictionary in dict_args:
        result += Counter(dictionary)
    return dict(result)



def producer_main(que_in, args):
    print(os.getpid(),"starting dataloader")
    datasets = load_dataset(args.dataset_name, args.dataset_config_name, streaming=args.stream, revision=args.revision)
    dataset = iter(datasets[args.split_name])

    for i, row in enumerate(dataset):
        if args.max_samples is not None and i >= args.max_samples:
            break
        que_in.put(row[args.text_column_name])

    for i in range(args.num_proc):
        que_in.put(None)


def main():
    args = parse_args()
    print(args)
    with Manager() as manager:

        que_in = manager.Queue(maxsize=args.num_proc*2)
        que_out = manager.Queue(maxsize=args.num_proc*2)
        
        # Start producer
        producer = Process(target=producer_main, args=(que_in, args))
        producer.start()

        workers = [Process(target=worker_main, args=(que_in, que_out, args)) for i in range(args.num_proc-1)]
        for w in workers:
            w.start()
        
        
        pbar = tqdm(total=args.max_samples)
        #Collect results
        frequencies = Counter()
        #for i in range(args.max_samples):
        exit_signals = 0
        while exit_signals < args.num_proc:
            t_ngrams = que_out.get()
            if t_ngrams is None:
                exit_signals += 1
            frequencies += Counter(t_ngrams)
            pbar.update(1)
        pbar.close()
        data = trivial_ngrams(frequencies)


        # prevent adding anything more to the queue and wait for queue to empty
        #que_in.close()
        #que_in.join_thread()

        #que_out.close()
        #que_out.join_thread()

        # Wait for producer to finish
        producer.join()

        for w in workers:
            w.join()

        # Save data to pickle
        output_file = Path(args.output_dir) / f"trivial_ngrams_{args.split_name}.pkl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(pickle.dumps(data))
        


        # prevent adding anything more to the process pool and wait for all processes to finish

if __name__ == '__main__':
    main()

    """
    python trivial_ngrams.py \
        --dataset_name andstor/the_pile_github \
        --dataset_config_name python \
        --text_column_name text \
        --tokenizer_name gpt2 \
        --num_proc 10 \
        --max_ngram 4 \
        --num_top_ngrams 1000 \
        --split_name train \
        --output_dir tmp

    """


"""
--stream \
--revision refs/pr/8 \
    """