# GPT Tokenizer

A from-scratch implementation of **BPE (Byte Pair Encoding)** tokenization, following Andrej Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE). Both tokenizers are built directly on raw bytes — no `tiktoken`, no Hugging Face — to understand exactly how subword vocabularies are learned and how text round-trips through encode/decode.

## Overview

Two implementations, building from simplest to most realistic:

- **`BasicTokenizer`** (`basic.py`) — minimal BPE operating on the raw UTF-8 byte stream. Merges the most frequent adjacent pair repeatedly until the target vocab size is reached.
- **`RegexTokenizer`** (`regex_tokenizer.py`) — extends the basic approach with GPT-4's regex pre-tokenization pattern, which splits text into chunks (words, numbers, contractions, punctuation runs) *before* applying BPE. Merges never cross chunk boundaries, matching the behavior of the GPT-4 tokenizer.

## How BPE works

1. Encode the text as raw bytes — 256 base tokens.
2. Count the most frequent adjacent token pair.
3. Merge that pair into a new token id and record the merge.
4. Repeat until the vocabulary reaches the target size.
5. To decode, replay the merges in reverse to reconstruct the original bytes.

`vocab_size` includes the 256 base byte tokens, so e.g. `vocab_size=276` learns 20 merges.

## Why the regex split matters

Pure byte-level BPE will happily merge across boundaries you don't want — gluing a word to the following space or punctuation. The `RegexTokenizer` first splits text with the GPT-4 pattern so that BPE only operates *within* sensible units. This keeps the learned vocabulary cleaner and mirrors how production tokenizers like `tiktoken` behave.

## Setup

```bash
pip install regex pytest
```

`regex` is required for the GPT-4 split pattern (the standard `re` module doesn't support the `\p{L}` Unicode property escapes); `pytest` is only needed to run the tests.

## Usage

```python
from regex_tokenizer import RegexTokenizer

tokenizer = RegexTokenizer()
with open("taylorswift.txt", encoding="utf-8") as f:
    tokenizer.train(f.read(), vocab_size=276)

ids = tokenizer.encode("hello world!!!? (안녕하세요!) 😉")
print(ids)                      # list of token ids
print(tokenizer.decode(ids))    # -> "hello world!!!? (안녕하세요!) 😉"
```

`BasicTokenizer` exposes the same `train` / `encode` / `decode` interface.

Running either file directly trains on `taylorswift.txt` and dumps the full learned vocabulary (with each merged token and its parent pair) to the `tests/` directory for inspection:

```bash
python basic.py            # writes tests/test.md
python regex_tokenizer.py  # writes tests/test2.md
```

## Tests

A pytest suite checks the core invariant of any tokenizer — that `decode(encode(text)) == text` — for **both** implementations across several cases: an empty string, a single character, mixed-script text (Latin, Korean, and an emoji), and the full corpus file. Test cases are adapted from Karpathy's [minbpe](https://github.com/karpathy/minbpe).

```bash
python -m pytest test_tokenizer.py -v
```

All 8 cases (4 inputs × 2 tokenizers) pass, confirming lossless round-trips including on multi-byte Unicode.

## Files

| File | Description |
|------|-------------|
| `basic.py` | `BasicTokenizer` — byte-level BPE |
| `regex_tokenizer.py` | `RegexTokenizer` — BPE with GPT-4 regex pre-tokenization |
| `test_tokenizer.py` | Encode/decode round-trip tests (adapted from minbpe) |
| `taylorswift.txt` | Sample corpus used for training |
| `tokenizer_dev.ipynb` | Development notebook comparing vocab across tokenizers |
| `tests/` | Dumped vocabularies from each tokenizer |

## References

- [Let's build the GPT Tokenizer — Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
- [minbpe](https://github.com/karpathy/minbpe)
