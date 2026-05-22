# GPT Tokenizer

A from-scratch implementation of BPE (Byte Pair Encoding) tokenization, following Andrej Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) tutorial.

## Overview

Two tokenizer implementations are included:

- **BasicTokenizer** (`basic.py`) — minimal BPE tokenizer operating on raw bytes
- **RegexTokenizer** (`regex_tokenizer.py`) — extends the basic tokenizer with GPT-4's regex pre-tokenization pattern to split text before encoding, matching the behavior of `tiktoken`

## How BPE Works

1. Encode text as raw bytes (256 base tokens)
2. Count the most frequent adjacent byte pair
3. Merge that pair into a new token and repeat for a target vocabulary size
4. Decode by reversing the merges

## Files

| File | Description |
|------|-------------|
| `basic.py` | BasicTokenizer class |
| `regex_tokenizer.py` | RegexTokenizer class with GPT-4 regex splitting |
| `test_tokenizer.py` | Test cases adapted from Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe) |
| `taylorswift.txt` | Sample corpus used for training |
| `tokenizer_dev.ipynb` | Development notebook comparing vocab across tokenizers |

## Reference

- [Let's build the GPT Tokenizer — Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
- [minbpe](https://github.com/karpathy/minbpe)
