import jieba, multiprocessing, copy
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

MODEL_PATH = "w2v_model.bin"
VEC_PATH = "w2v_vector.bin"
PUNCTUATION_LIST = ".。,，,、?？:：;；{}[]【】“‘’”《》/!！%……（）()<>@#$~^￥%&*\"\'=+-_——「」"

def clean_line(line_segment):
    res = []
    for e in line_segment:
        if e in PUNCTUATION_LIST: continue
        res.append(e)
    return res

def get_word2vec():
    sentences = [list(jieba.cut(line.strip())) for line in open("../data/corpus", encoding="utf8").readlines()]
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model.save(MODEL_PATH)
    model.wv.save(VEC_PATH)

MODEL = Word2Vec.load(MODEL_PATH)
WV = KeyedVectors.load("w2v_vector.bin", mmap='r')

def textRank(sc, sims, thr_sim=0.7, sc_pow=0.7):
    assert (len(sc) == len(sims))

    def retainMax(arr):
        m = np.max(arr) * thr_sim
        return [0 if x < m else x for x in arr]

    sc_ = sc
    sims = (sims / np.sum(sims, axis=1, keepdims=True)).astype(np.float)
    sims = np.apply_along_axis(retainMax, 1, sims)
    sc = np.array(sc) / np.sum(sc)

    for _ in range(1000):
        ss = copy.deepcopy(sc)
        for i in range(len(sc)):
            s = 0
            for j in range(len(sc)):
                if i == j: continue
                s += sc[j] * sims[j][i]
            sc[i] = 0.15 + 0.85 * s
        if np.sum(np.abs(sc - ss)) < 1e-8:
            break

        # for i in range(len(sc)):sc[i] *= math.pow(sc_[i], sc_pow)
    print(sc)
    return sc

def cal_sim(word1, word2):
    if word1 not in WV.vocab or word2 not in WV.vocab:
        return 0.0
    index1, index2 = WV.vocab.get(word1).index, WV.vocab.get(word2).index
    vec1, vec2 = WV.vectors[index1], WV.vectors[index2]
    return cosine_similarity([vec1], [vec2])[0][0]

def text_rank(word_list):
    sim_mat = np.zeros([len(word_list), len(word_list)])
    for i in range(len(word_list)):
        for j in range(len(word_list)):
            sim_mat[i][j] = cal_sim(word_list[i], word_list[j])
    # 迭代得到句子的textrank值，排序并取出摘要"""
    # 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
    nx_graph = nx.from_numpy_array(sim_mat)
    # 得到所有句子的textrank值
    scores = nx.pagerank(nx_graph)
    # 根据textrank值对未处理的句子进行排序
    ranked_words = sorted(((scores[i], s) for i, s in enumerate(word_list)), reverse=True)
    _scores = textRank(np.ones([len(word_list)]), sim_mat)
    _ranked_words = sorted(((_scores[i], s) for i, s in enumerate(word_list)), reverse=True)
    return ranked_words


if __name__ == "__main__":
    texts = [list(jieba.cut(line.strip())) for line in open("../data/corpus", encoding="utf8").readlines()]
    #get_word2vec()
    res = text_rank(list(set(texts[4])))
    pass