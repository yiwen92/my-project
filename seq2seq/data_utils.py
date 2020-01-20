import os
from collections import defaultdict, Counter
from tqdm import tqdm

def gen_correct_data():
    TOKENS = [("<PAD>", 1000000), ("<UNK>", 1000000), ("<S>", 1000000), ("</S>", 1000000)]
    word_freq = defaultdict(int)
    file_path = "data/nlpcc2018+hsk/"
    for file_name in os.listdir(file_path):
        print("read file: %s" % (file_name))
        for i, line in enumerate(tqdm(open(file_path + file_name, encoding="utf8").readlines())):
            for e in line.strip().split():
                word_freq[e] += 1
    sorted_word_freq = sorted(word_freq.items(), key=lambda d: d[1], reverse=True)
    vocab_src, vocab_trg = TOKENS + sorted_word_freq, TOKENS + sorted_word_freq
    with open(file_path + "vocab.src", "w", encoding="utf8") as fin:
        for w, f in vocab_src:
            fin.write(str(w) + " " + str(f) + "\n")
    with open(file_path + "vocab.trg", "w", encoding="utf8") as fin:
        for w, f in vocab_trg:
            fin.write(str(w) + " " + str(f) + "\n")

if __name__ == "__main__":
    gen_correct_data()