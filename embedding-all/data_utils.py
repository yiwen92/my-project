import random, json
import numpy as np
from tqdm import tqdm
from utils import clean_line, re_en, token2list
from collections import defaultdict
from config import MAX_NUM_NEG, conf, VOCAB_SIZE, SEQ_LEN
from seg_utils import Tokenizer

word2id = json.load(open(conf.vocab, encoding="utf8"))

class TrainData():
    def __init__(self):
        self.tokenizer = Tokenizer()

    def get_train_corp(self):
        text = [e.strip().split("\t")[33].replace("\\n", "").lower() for e in open("data/jddata_1000", encoding="utf8").readlines()]
        with open("data/train.txt", "w", encoding="utf8") as fin:
            fin.write("\n".join(text))

    def gen_train_samples(self, file_path):
        sample_set = {}
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
                if word in sample_set: continue
                positive_entity = random.sample(rhs, 1)[0]
                negative_entitys = []
                negative_indexes = [i for i in range(len(important_tokens)) if i != cur_index]
                negs = np.random.choice(negative_indexes, size=num_neg, replace=False)
                for neg_index in negs:
                    while True:
                        neg_tmp = random.sample(important_tokens[neg_index], 1)[0]
                        if neg_tmp != word and neg_tmp not in negative_entitys: break
                    negative_entitys.append(neg_tmp)
                sample_set[word] = [positive_entity, negative_entitys]
        # 产生字典
        token_freq = defaultdict(int); token_freq['UNKNOWN'] = 1e8
        for k, (p, n) in sample_set.items():
            tmp = [k, p] + n
            for t in tmp:
                if re_en.fullmatch(t): token_freq[t] += 1
                else:
                    for e in list(t):
                        token_freq[e] += 1
        sorted_token_freq = sorted(token_freq.items(), key=lambda d: d[1], reverse=True)[:VOCAB_SIZE]
        word2id = {w: i for i, (w, f) in enumerate(sorted_token_freq)}
        if conf.over_write_vocab: json.dump(word2id, open(conf.vocab, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        _keys_ = list(sample_set.keys())
        train_set = {k: sample_set[k] for k in _keys_[:int(len(_keys_) * conf.train_valid_ratio)]}
        valid_set = {k: sample_set[k] for k in _keys_[int(len(_keys_) * conf.train_valid_ratio):]}
        json.dump(train_set, open(conf.train_samples, "w", encoding="utf8"), ensure_ascii=False, indent=2)
        json.dump(valid_set, open(conf.valid_samples, "w", encoding="utf8"), ensure_ascii=False, indent=2)

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]

def seq2ids(text, max_length=SEQ_LEN):
    """将文本转换为id表示"""
    data = token2list(text)
    data_id = [word2id.get(x, 0) for x in data]
    if len(data_id) > max_length: x_pad = data_id[: max_length]
    else: x_pad = data_id + [0] * (max_length - len(data_id))
    return x_pad

class train_sample():
    def __init__(self, entity, pos_entity, neg_entitys):
        self.entity = entity
        self.pos_entity = pos_entity
        self.neg_entitys = neg_entitys

def gen_train_samples(file_path):
    train_samples = json.load(open(file_path, encoding="utf8"))
    samples = []
    for k, (p, n) in train_samples.items():
        kid = seq2ids(k)
        pid = seq2ids(p)
        nid = [seq2ids(e) for e in n]
        ts = train_sample(kid, pid, nid)
        samples.append(ts)
    X = np.array([e.entity for e in samples])
    Y = np.array([[e.pos_entity] + e.neg_entitys for e in samples])
    return X, Y

if __name__ == "__main__":
    td = TrainData()
    td.gen_train_samples("data/train.txt")