from tokenization import BPE

corpus = []

with open("data/TheGoldenTouch.txt", "r") as f:
    for line in f.readlines():
        corpus += line.strip().split(" ")
print(corpus,len(corpus))

bpe = BPE(vocab_size=100)
bpe.fit(corpus)
print(bpe.encode("golden"))
print(bpe.decode(bpe.encode("golden")))
with open("data/TheGoldenTouch_BPE_vocab.txt", "w") as f:
    for word in bpe.vocab:
        f.write(word + "\n")
