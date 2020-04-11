import random, json
import numpy as np
from tqdm import tqdm
from utils import clean_line, re_en
from collections import defaultdict
from config import MAX_NUM_NEG, conf, VOCAB_SIZE
from seg_utils import Tokenizer

class TrainData():
    def __init__(self):
        self.tokenizer = Tokenizer()

    def gen_train_samples(self, file_path):
        train_set = {}
        np.random.seed(8)
        # 加载数据，以文本为单位
        important_tokens = []
        text = open(file_path, encoding="utf8").readlines()[:10]
        print("select important tokens...")
        for e in tqdm(text, total=len(text)):
            tmp = self.tokenizer.select_important_tokens(clean_line(e.strip()))
            if len(tmp) < 10: continue
            important_tokens.append(tmp)
        # 采样正负样本，同一个文本中的词为正样本，不同文本中的词为负样本
        print("sample train set...")
        num_neg = min(len(important_tokens) - 1, MAX_NUM_NEG)
        for cur_index, cur_ele in tqdm(enumerate(important_tokens), total=len(important_tokens)):
            np.random.shuffle(cur_ele)
            cut_index = int(len(cur_ele) / 3)
            lhs, rhs = cur_ele[: cut_index], cur_ele[cut_index:]
            for word in lhs:
                if word in train_set: continue
                positive_entity = random.sample(rhs, 1)[0]
                negative_entitys = []
                negative_indexes = [i for i in range(len(important_tokens)) if i != cur_index]
                negs = np.random.choice(negative_indexes, size=num_neg, replace=False)
                for neg_index in negs:
                    while True:
                        neg_tmp = random.sample(important_tokens[neg_index], 1)[0]
                        if neg_tmp != word and neg_tmp not in negative_entitys: break
                    negative_entitys.append(neg_tmp)
                train_set[word] = [positive_entity, negative_entitys]
        # 产生字典
        token_freq = defaultdict(int); token_freq['UNKNOWN'] = 1e8
        for k, (p, n) in train_set.items():
            tmp = [k, p] + n
            for t in tmp:
                if re_en.fullmatch(t): token_freq[t] += 1
                else:
                    for e in list(t):
                        token_freq[e] += 1
        sorted_token_freq = sorted(token_freq.items(), key=lambda d: d[1], reverse=True)[:VOCAB_SIZE]
        word2id = {w: i for i, (w, f) in enumerate(sorted_token_freq)}
        json.dump(word2id, open(conf.vocab, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        json.dump(train_set, open(conf.train_samples, "w", encoding="utf8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    td = TrainData()
    td.gen_train_samples("data/train.txt")