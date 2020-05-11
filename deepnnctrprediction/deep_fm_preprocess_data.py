#coding=utf8
"""
特征工程参考(https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py)完成
-对数值型特征，normalize处理
-对类别型特征，对长尾(出现频次低于200)的进行过滤
"""
import os
import sys
import random
import collections
import argparse
from multiprocessing import Pool as ThreadPool

# 13个连续型列，26个类别型列
continous_features = range(1, 14)
categorial_features = range(14, 40)

# 对连续值进行截断处理(取每个连续值列的95%分位数)
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    类别型特征编码字典
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(e) for e in self.dicts]
        #return map(len, self.dicts)


class ContinuousFeatureGenerator:
    """
    对连续值特征做最大最小值normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature       # [sys.maxint] * num_feature
        self.max = [-sys.maxsize] * num_feature      # [-sys.maxint] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


def preprocess(input_dir, output_dir):
    """
    对连续型和类别型特征进行处理
    """

    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(input_dir + 'train.txt', continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(input_dir + 'train.txt', categorial_features)#, cutoff=150)

    output = open(output_dir + 'feature_map', 'w')
    for i in continous_features:
        output.write("{0} {1}\n".format('I' + str(i), i))
    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [dists.num_feature]
    for i in range(1, len(categorial_features) + 1):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)
        for key, val in dicts.dicts[i - 1].items():
            output.write("{0} {1}\n".format('C' + str(i) + '|' + str(key), categorial_feature_offset[i - 1] + val + 1))

    random.seed(0)

    # 90%的数据用于训练，10%的数据用于验证
    with open(output_dir + 'tr.libsvm', 'w') as out_train:
        with open(output_dir + 'va.libsvm', 'w') as out_valid:
            with open(input_dir + 'train.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_vals = []
                    for i in range(0, len(continous_features)):
                        val = dists.gen(i, features[continous_features[i]])
                        feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        feat_vals.append(str(val) + ':1')

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    with open(output_dir + 'te.libsvm', 'w') as out:
        with open(input_dir + 'test.txt', 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1]) + categorial_feature_offset[i]
                    feat_vals.append(str(val) + ':1')

                out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))


if __name__ == "__main__":
    input_dir = './criteo_data/'
    output_dir = './criteo_data/'
    print("开始数据处理与特征工程...")
    preprocess(input_dir, output_dir)