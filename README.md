# lm-detect

> Zero-Shot Machine-Generated Text Detection

lm-detect provides tooling for automatically detecting AI-written text.

## Installation

```bash
pip install lm-detect
```

## Usage

```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModel.from_pretrained("gpt2")

>>> from lm_detect import Perplexity
>>> ppl = Perplexity(model, tokenizer)
>>> ppl("Lorem ipsum dolor sit amet")
6.536835670471191
```

## Authors

lm-detect is written by [André Storhaug](https://github.com/andstor) <andr3.storhaug@gmail.com>

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE) file for details.
