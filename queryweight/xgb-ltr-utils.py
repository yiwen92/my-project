from sklearn.datasets import load_svmlight_file
from xgboost import DMatrix
import xgboost as xgb
from sklearn.externals import joblib
import matplotlib.pyplot as plt

DATA_PATH, TASK = "get_jdcv_data/", "query_weight"

def load_group_data(group_data_file):
    group_data = []
    with open(group_data_file, "r", encoding="utf8") as f:
        data = f.readlines()
        for line in data:
            group_data.append(int(line.split("\n")[0]))
    return group_data

class xgbLtr:
    def __init__(self):
        self.train_file = DATA_PATH + TASK + ".train"
        self.valid_file = DATA_PATH + TASK + ".valid"
        self.test_file = DATA_PATH + TASK + ".test"
        self.model_path = "rank_model/"
        self.model_name = TASK + "_xgb1.model"

    def load_data(self):
        x_train, y_train = load_svmlight_file(self.train_file)
        x_valid, y_valid = load_svmlight_file(self.valid_file)
        x_test, y_test = load_svmlight_file(self.test_file)
        print("train data shape: [%d, %d]" % (x_train.shape[0], x_train.shape[1]))

        group_train = load_group_data(DATA_PATH + TASK + ".train.group")
        group_valid = load_group_data(DATA_PATH + TASK + ".valid.group")
        group_test = load_group_data(DATA_PATH + TASK + ".test.group")

        self.train_dmatrix = DMatrix(x_train, y_train)
        self.valid_dmatrix = DMatrix(x_valid, y_valid)
        self.test_dmatrix = DMatrix(x_test, y_test)

        self.train_dmatrix.set_group(group_train)
        self.valid_dmatrix.set_group(group_valid)
        self.test_dmatrix.set_group(group_test)

    def train(self):
        params = {'booster': 'gbtree', 'objective': 'rank:pairwise', 'eta': 0.01, 'gamma': 1.0, 'min_child_weight': 0.1,
                  'max_depth': 2, 'eval_metric': 'ndcg@1'}  # ndcg@1, logloss
        xgb_model = xgb.train(params, self.train_dmatrix, num_boost_round=4,
                              evals=[(self.train_dmatrix, 'train'), (self.valid_dmatrix, 'valid'), (self.test_dmatrix, 'test')])
        pred = xgb_model.predict(self.test_dmatrix)
        print("save model to %s" % (self.model_path))
        xgb_model.dump_model(self.model_path + self.model_name + ".txt")
        xgb_model.save_model(self.model_path + self.model_name)
        joblib.dump(xgb_model, self.model_path + '/xgb_clf.m')
        # save figures
        plt.clf()
        xgb.plot_importance(xgb_model)
        plt.savefig(self.model_path + '/feature_importance.png', dpi=800, format='png')

    def plotXgboostTree(self):
        xgb_model = xgb.Booster(model_file=self.model_path + self.model_name)
        xgbclf = joblib.load(self.model_path + '/xgb_clf.m')
        #plt.clf();    xgb.plot_tree(xgbclf, num_trees=0, fmap='./xgb.fmap');    plt.savefig('xgb_tree.png', dpi=800, format='png'); exit(0)
        for i in range(4):
            #plt.clf()
            xgb.plot_tree(xgb_model, num_trees = i, fmap = './get_jdcv_data/feature.fmap')
            fig = plt.gcf()
            fig.set_size_inches(150, 100)
            fig.savefig('xgb_tree_'+ str(i) +'.png')
            #plt.savefig('xgb_tree_' + str(i) + '.png', dpi=800, format='png')
            a=1
        pass


if __name__ == "__main__":
    xgb_ltr = xgbLtr()  ; xgb_ltr.plotXgboostTree()
    #xgb_ltr.load_data()
    #xgb_ltr.train()
    pass