def get_stats(ids, counts = None):
    counts = {}

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


class BasicTokenizer():
    def __init__(self):
        super().__init__()

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))

        for i in range(num_merges):
            stats = get_stats(tokens)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f" merging pair {top_pair} into a new token {idx}")
            tokens = merge(tokens, top_pair, idx)
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({self.vocab[idx]}) had {stats[top_pair]} occurrences")

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')

        return text
    
    def encode(self, text):
        tokens = list(text.encode('utf-8'))

        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)

        return tokens



if __name__ == "__main__":
    with open("taylorswift.txt", "r", encoding = "utf-8") as f:
        text = f.read()
    
    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=276, verbose=False)

    lines = []

    lines.append("\n# Full Vocab\n")
    for idx, token_bytes in tokenizer.vocab.items():
        token_str = token_bytes.decode('utf-8', errors='replace')
        if idx in tokenizer.merges.values():
            pair = next(p for p, i in tokenizer.merges.items() if i == idx)
            lines.append(f"{idx}: `{token_str}` (merged from {pair[0]} + {pair[1]})\n")
        else:
            lines.append(f"{idx}: `{token_str}`\n")

    with open("tests/test.md", "w", encoding="utf-8") as f:
        f.writelines(lines)