import os, re, math, json
from collections import defaultdict
from config import conf
from seg_utils import Tokenizer
from data_utils import VOCAB_SIZE, special_symbols
from tqdm import tqdm

def handle_sort_search_data():
    pass

def get_corpus():
    seg = Tokenizer()
    matchObj = re.compile(r'(.+)&([0-9]+)', re.M | re.I)
    word_freq, idf, query_num, query_len = defaultdict(int), defaultdict(int), 0, defaultdict(int)
    print("read file: %s\nwrite file: %s" % (conf.original_corpus, conf.pretrain_corpus))
    for filename in os.listdir(conf.original_corpus):
        corpus = []
        print("read file %s" % (filename))
        freqth = 1
        if filename == "jdtitle": freqth = 6
        if not os.path.exists(conf.pretrain_corpus): os.makedirs(conf.pretrain_corpus)
        for line in tqdm(open(conf.original_corpus + filename, encoding="utf8").readlines()):
            matchRes = matchObj.match(line)
            if not matchRes: continue
            text, freq = matchRes.group(1), int(matchRes.group(2))
            if freq < freqth: continue
            text = re.sub(u"[=—】★一\-【◆④\t ]{1,}|\d[、.）)．]|[(（]\d[）)]|[0-9]{3,}", "", text)
            senten2term, word_seg = seg.tokenize(text)
            if len(senten2term) < 2: continue
            query_len[len(senten2term)] += 1
            for e in senten2term:
                word_freq[e] += 1
            for e in set(senten2term):
                idf[e] += 1
            query_num +=1
            corpus.append(text.strip().lower() + "\n")
        if corpus:
            with open(conf.pretrain_corpus + filename + ".txt", "w", encoding="utf8") as fout:
                fout.write(''.join(corpus))
    idf_value = {k: round(math.log10(query_num / (v + 1)), 3) for k, v in idf.items()}
    sorted_idf_value = sorted(idf_value.items(), key=lambda d: d[1], reverse=True)
    sorted_word_freq = sorted(special_symbols.items(), key=lambda d: d[1]) + [("▁", -1)] + sorted(word_freq.items(), key=lambda d: d[1], reverse=True)
    top_word_freq = sorted_word_freq[: VOCAB_SIZE]
    vocab = {w: i for i, (w, f) in enumerate(top_word_freq)}
    with open(conf.vocab, "w", encoding="utf8") as fin:
        json.dump(vocab, fin, ensure_ascii=False, indent=2)
    with open(conf.idf, "w", encoding="utf8") as fin:
        json.dump(idf_value, fin, ensure_ascii=False, indent=2)
    with open(conf.query_len, "w", encoding="utf8") as fin:
        json.dump(query_len, fin, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    get_corpus()