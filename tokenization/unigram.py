import collections
import heapq

class UnigramTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = None

    def train(self, data):
        # Initialize with all possible substrings
        substrings = collections.Counter()
        for word in data:
            for i in range(len(word)):
                for j in range(i+1, len(word)+1):
                    substrings[word[i:j]] += 1

        # Create a priority queue with negative frequencies to simulate a max-heap
        heap = [(-freq, substr) for substr, freq in substrings.items()]
        heapq.heapify(heap)

        # Reduce vocab size to the target size
        while len(heap) > self.vocab_size:
            heapq.heappop(heap)

        self.vocab = set(sub for _, sub in heap)

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
unigram_tokenizer = UnigramTokenizer(vocab_size=5)
unigram_tokenizer.train(["aaabadaaabac"])
print(unigram_tokenizer.tokenize("aaabadaaabac"))
