#!/usr/bin/python
import math
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from qw import query_weight
from generate_train_samples import get_feature

data_path, TASK, MODEL_FILE = "get_jdcv_data/", "query_weight", "rank_model/query_weight_xgb.model"
#data_path, TASK, MODEL_FILE = "rank/", "mq2008", "rank/xgb.model"

def load_group_data(group_data_file):
    group_data = []
    with open(group_data_file, "r", encoding="utf8") as f:
        data = f.readlines()
        for line in data:
            group_data.append(int(line.split("\n")[0]))
    return group_data

def train(model_file):
    trans_data(data_path)
    #  This script demonstrate how to do ranking with xgboost.train
    x_train, y_train = load_svmlight_file(data_path + TASK + ".train")
    x_valid, y_valid = load_svmlight_file(data_path + TASK + ".valid")
    x_test, y_test = load_svmlight_file(data_path + TASK + ".test")
    print("train data shape: [%d, %d]" % (x_train.shape[0], x_train.shape[1]))

    group_train = load_group_data(data_path + TASK + ".train.group")
    group_valid = load_group_data(data_path + TASK + ".valid.group")
    group_test = load_group_data(data_path + TASK + ".test.group")

    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test, y_test)

    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)
    test_dmatrix.set_group(group_test)

    params = {'booster':'gbtree', 'objective': 'rank:pairwise', 'eta': 0.01, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': 6, \
              'eval_metric':'ndcg@1'}     # ndcg@1, logloss
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=400, evals=[(train_dmatrix, 'train'), (valid_dmatrix, 'valid'), (test_dmatrix, 'test')])
    pred = xgb_model.predict(test_dmatrix)
    print("save model to %s" % (model_file))
    xgb_model.dump_model(model_file + ".txt")
    xgb_model.save_model(model_file)
    # save figures
    plt.clf()
    xgb.plot_importance(xgb_model)
    plt.savefig('rank_model/feature_importance.png', dpi=800, format='png')

xgb_model = xgb.Booster(model_file=MODEL_FILE)
def predict(x, n_feature=32):
    feature = np.array([0.0] * n_feature)
    for e in x.split(" ")[1:]:
        index, value = int(e.split(":")[0]), float(e.split(":")[1])
        feature[index-1] = value
    feature_csr = sparse.csr_matrix(feature)
    input = DMatrix(feature_csr)
    prob = xgb_model.predict(input)
    return prob

def save_data(group_data,output_feature,output_group):
    if len(group_data) == 0: return
    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

def trans_data(path):
    for mode in ['train', 'test', 'valid']:
        fi = open(path + mode + ".txt", encoding="utf8")
        output_feature = open(path + TASK + "." + mode, "w", encoding="utf8")
        output_group = open(path + TASK + "." + mode + ".group", "w", encoding="utf8")
        group_data = []
        group = ""
        for line in fi:
            if not line: break
            if "#" in line: line = line[:line.index("#")]
            splits = line.strip().split(" ")
            if splits[1] != group:
                save_data(group_data, output_feature, output_group)
                group_data = []
            group = splits[1]
            group_data.append(splits)
        save_data(group_data, output_feature, output_group)
        fi.close(); output_feature.close(); output_group.close()
    pass

def rank_query(query="产品策划"):
    qw = query_weight(1000000)
    res0 = qw.run_step(query)
    weight_attn, weight_idf, weight_lm = qw.weight_attn, qw.weight_idf, qw.weight_lm
    sen2terms = [k for k, v in weight_attn]
    tmp, score_sum = [], 1e-8
    for term in sen2terms:
        feature_vector, _ = get_feature(term, sen2terms, weight_attn, weight_idf, weight_lm)
        feature = np.array(feature_vector)
        feature_csr = sparse.csr_matrix(feature)
        input = DMatrix(feature_csr)
        score = xgb_model.predict(input)[0]
        prob = 1.0 / (1 + math.exp(-1 * score))
        tmp.append((term, prob))
        score_sum += prob
    res = [(k, round(v / score_sum, 3)) for k, v in tmp]
    sorted_res0 = sorted(res0, key=lambda d: d[1], reverse=True)
    sorted_res = sorted(res, key=lambda d: d[1], reverse=True)
    return res

if __name__ == "__main__":
    query = "电话销售"
    fea = ["6 1:1 5:1 10:3 14:1 20:1 23:1 24:1 30:0.332 31:0.202 32:0.029"]
    fea.append("5 3:1 4:1 10:2 11:0.143 12:1 16:1 23:1 24:1 30:0.094 31:0.11 32:0.074")
    fea.append("4 3:1 4:1 10:2 11:0.286 14:1 15:1 23:1 27:1 30:0.18 31:0.177 32:0.289")
    #f4 = "0 3:1 4:1 10:1 11:0.5 14:1 15:1 22:1 24:1 30:0.199 31:0.301 32:0.857"
    #trans_data(data_path)
    #train(model_file=MODEL_FILE)    ;   exit()
    #res = {i: predict(f) for i, f in enumerate(fea)}
    #predict(fea[0])
    rank_query(query)
    pass