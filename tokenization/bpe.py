import collections

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = None

    def train(self, data):
        # Initialize with single characters
        substrings = collections.Counter()
        for word in data:
            for char in word:
                substrings[char] += 1

        pairs = collections.Counter()
        for word in data:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] += 1

        while len(substrings) < self.vocab_size:
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_substring = best_pair[0] + best_pair[1]
            substrings[new_substring] = pairs[best_pair]

            # Update pairs
            new_pairs = collections.Counter()
            for word in data:
                word = word.replace(best_pair[0] + best_pair[1], new_substring)
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    new_pairs[pair] += 1
            pairs = new_pairs

        self.vocab = set(substrings.keys())

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            for j in range(len(text), i, -1):
                if text[i:j] in self.vocab:
                    tokens.append(text[i:j])
                    i = j - 1
                    break
            i += 1
        return tokens

# 测试代码
bpe_tokenizer = BPETokenizer(vocab_size=10)
bpe_tokenizer.train(["aaabadaaabac"])
print(bpe_tokenizer.tokenize("aaabadaaabac"))
