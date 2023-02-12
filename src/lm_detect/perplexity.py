import torch
from tqdm import tqdm

from .utils import encodings_words_indexof

class Perplexity:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device

    def __call__(self, text, stride=50, ignore_words=[]):
        """Calculates the perplexity of a input text."""
        
        return self.perplexity(text, stride, ignore_words)

    def perplexity(self, text, stride, ignore_words):
        """
        Calculates the perplexity of a input text.
        :param text: The text to calculate the perplexity of.
        :param stride: Amount to move the sliding window when calculating the perplexity. A stride of 1 is the slowest but most accurate.
        :return: The perplexity of the input text.
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        
        ignore_indices = encodings_words_indexof(encodings, ignore_words, self.tokenizer)
        print(ignore_indices)
        
        max_length = self.model.config.n_positions
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100 # def cross_entropy(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
            
            # set all target_ids that are in ignore_indices to -100
            for key, indices in ignore_indices.items():
                for idx in indices:
                    if idx[0] >= begin_loc and idx[-1] < end_loc:
                        target_ids[:, idx[0]-begin_loc:idx[-1]-begin_loc+1] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.item()