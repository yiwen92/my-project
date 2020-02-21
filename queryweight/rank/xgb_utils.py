#!/usr/bin/python
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

MODEL_FILE = "xgb.model"

def train(model_file):
    #  This script demonstrate how to do ranking with xgboost.train
    x_train, y_train = load_svmlight_file("mq2008.train")
    x_valid, y_valid = load_svmlight_file("mq2008.vali")
    x_test, y_test = load_svmlight_file("mq2008.test")

    group_train = []
    with open("mq2008.train.group", "r", encoding="utf8") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))

    group_valid = []
    with open("mq2008.vali.group", "r", encoding="utf8") as f:
        data = f.readlines()
        for line in data:
            group_valid.append(int(line.split("\n")[0]))

    group_test = []
    with open("mq2008.test.group", "r", encoding="utf8") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))

    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)

    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)

    params = {'objective': 'rank:pairwise', 'eta': 0.01, 'gamma': 1.0, 'min_child_weight': 0.1, 'max_depth': 8}
    xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4, evals=[(valid_dmatrix, 'validation')])
    pred = xgb_model.predict(test_dmatrix)
    xgb_model.dump_model(model_file + ".txt")
    xgb_model.save_model(model_file)
    # save figures
    plt.clf()
    xgb.plot_importance(xgb_model)
    plt.savefig('feature_importance.png', dpi=800, format='png')

def predict(x, n_feature=46):
    xgb_model = xgb.Booster(model_file=MODEL_FILE)
    feature = np.array([0.0] * n_feature)
    for e in x.split(" ")[1:]:
        index, value = int(e.split(":")[0]), float(e.split(":")[1])
        feature[index-1] = value
    feature_csr = sparse.csr_matrix(feature)
    input = DMatrix(feature_csr)
    prob = xgb_model.predict(input)
    a=1

if __name__ == "__main__":
    #train(model_file=MODEL_FILE) #; exit()
    f="2 1:0.066262 2:0.142857 3:0.333333 5:0.066851 11:0.067003 12:0.142857 13:0.333333 15:0.068163 16:0.051200 17:0.106383 18:0.312500 19:0.190476 20:0.051186 21:0.752123 22:0.519954 23:0.647139 24:0.552374 25:0.472882 26:0.157865 27:0.005263 28:0.083147 29:0.498579 30:0.176608 31:0.362464 32:0.135578 37:0.754945 38:0.537199 39:0.647018 40:0.562377 41:0.285714 42:0.141414 44:0.000687 45:0.000991"
    f="1 1:0.218112 2:0.285714 3:0.333333 5:0.219337 11:0.216254 12:0.285714 13:0.333333 15:0.217611 16:0.217767 17:0.297872 18:0.500000 19:0.142857 20:0.218064 21:0.783012 22:0.554006 23:0.712058 24:0.638314 25:0.391877 26:0.036784 27:0.305970 28:0.051190 29:0.375081 30:0.116636 31:0.361047 32:0.089577 37:0.786565 38:0.570912 39:0.712515 40:0.647727 41:0.285714 42:0.212121 44:0.014392 45:0.006938 46:0.001199"
    f="0 3:0.250000 13:0.259494 18:0.100000 19:1.000000 22:0.470407 24:0.367837 29:0.675712 30:0.296249 31:0.005599 32:0.227130 38:0.329330 40:0.150330 41:1.000000 42:0.826087"
    predict(f)
    pass