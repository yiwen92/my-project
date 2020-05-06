import math
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

def parse_xgb_dict(xgb_dump):
    xgb_tree_path_dict = {}
    tree_num = -1
    with open(xgb_dump, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.split('[')[0] == 'booster':
                tree_num += 1
                root = True
                if tree_num not in xgb_tree_path_dict:
                    xgb_tree_path_dict[tree_num] = {'decision_nodes': {}, 'root': -1}
            else:
                node_id = line.strip().split(':')[0]
                if root:
                    xgb_tree_path_dict[tree_num]['root'] = node_id
                    root = False
                arr = line.split('[')
                if len(arr) == 1:  # leaf node
                    leaf_value = line.split('=')[-1]
                    if node_id not in xgb_tree_path_dict[tree_num]['decision_nodes']:
                        xgb_tree_path_dict[tree_num]['decision_nodes'][node_id] = [leaf_value]
                else:   # tree node
                    tmp = arr[1].split(']')
                    fid = tmp[0]
                    feat_id, split_thr = fid.split('<')
                    jump_nodes = tmp[1].strip().split(',')
                    yes_node = jump_nodes[0].split('=')[-1]
                    no_node = jump_nodes[1].split('=')[-1]
                    missing_node = jump_nodes[2].split('=')[-1]
                    if node_id not in xgb_tree_path_dict[tree_num]['decision_nodes']:
                        xgb_tree_path_dict[tree_num]['decision_nodes'][node_id] = [int(feat_id.split('f')[-1]),
                                                                                   split_thr, yes_node, no_node,
                                                                                   missing_node]
        return xgb_tree_path_dict
id2fea = { 0: "性别", 1: "年龄", 2: "当前状态", 3: "薪水比例", 4: "城市匹配", 5: "学位匹配", 6: "工作经验匹配", 7: "行业匹配", 8: "公司距离", 9: "跳槽距离" }
def trvelSingleTree(treeDict, features):
    decision_nodes = treeDict['decision_nodes']
    root_node = treeDict['root']
    cur_decision = decision_nodes[root_node]
    boosting_value = 0.0  # logit value
    nodes = []
    if len(cur_decision) == 1:
        nodes.append([root_node, 'boosting_value', float(cur_decision[0])])
    else:
        nodes.append([root_node, id2fea[cur_decision[0]], features[cur_decision[0]], cur_decision[1], cur_decision[2], cur_decision[3]])
    while True:
        if len(cur_decision) == 1:  # leaf node
            boosting_value += float(cur_decision[0])
            break
        else:
            feat_id = cur_decision[0]
            split_thr = float(cur_decision[1])
            yes_node = cur_decision[2]
            no_node = cur_decision[3]
            if features[feat_id] < split_thr:
                cur_decision = decision_nodes[yes_node]
                if len(cur_decision) == 1:
                    nodes.append([yes_node, 'boosting_value', float(cur_decision[0])])
                else:
                    nodes.append([yes_node, id2fea[cur_decision[0]], features[cur_decision[0]], cur_decision[1], cur_decision[2], cur_decision[3]])
            else:
                cur_decision = decision_nodes[no_node]
                if len(cur_decision) == 1:
                    nodes.append([no_node, 'boosting_value', float(cur_decision[0])])
                else:
                    nodes.append([no_node, id2fea[cur_decision[0]], features[cur_decision[0]], cur_decision[1], cur_decision[2], cur_decision[3]])
    prob = 1.0 / (1 + math.exp(-1 * boosting_value))
    return prob, nodes

def trveltree(tree_dict, Features, tree_feature):
    tree_result = {}
    for f in tree_feature:
        indexs = []
        if len(tree_feature[f]) > 4:
            delt = int(len(tree_feature[f]) / 3)
            indexs.append(0);   indexs.append(delt - 1);    indexs.append(2*delt - 1);  indexs.append(len(tree_feature[f]) - 2)
        else:
            indexs = [i for i in range(len(tree_feature[f]))]
        if len(indexs) == 1:
            features = Features.copy()
            features[f] = tree_feature[f][indexs[0]]
            if f not in tree_result: tree_result[f] = []
            prob, nodpath = trvelSingleTree(tree_dict, features)
            tree_result[f].append((tree_feature[f][indexs[0]], prob, nodpath))
        for i in range(len(indexs) - 1):
            if f == 0:
                a=1
            low = indexs[i];    high = indexs[i+1]
            v = ( tree_feature[f][low] + tree_feature[f][high] ) / 2
            features = Features.copy()
            features[f] = v
            prob, nodpath = trvelSingleTree(tree_dict, features)
            if f not in tree_result:  tree_result[f] = []
            tree_result[f].append((tree_feature[f][low], tree_feature[f][high], prob, nodpath))
    return tree_result

def predict_proba_(xgb_tree_path_dict,input_X, tree_features):
    features = input_X[0]
    boosting_value = 0.0  # logit value
    hit_feats = []
    path_ids = []
    tfs = {}
    for tree_num in xgb_tree_path_dict:

        tf = trveltree(xgb_tree_path_dict[tree_num], features, tree_features[tree_num])
        tfs[tree_num] = tf
        if tree_num == 42:
            aa=1
        sub_tree_path = []
        sub_hit_nodes = {}
        tree_info = xgb_tree_path_dict[tree_num]
        decision_nodes = tree_info['decision_nodes']
        root_node = tree_info['root']
        cur_decision = decision_nodes[root_node]
        while True:
            if len(cur_decision) == 1: # leaf node
                boosting_value += float(cur_decision[0])
                break
            else:
                feat_id = cur_decision[0]
                sub_tree_path.append(feat_id)
                if feat_id not in sub_hit_nodes:
                    sub_hit_nodes[feat_id] = 0
                sub_hit_nodes[feat_id] += 1
                split_thr = float(cur_decision[1])
                yes_node = cur_decision[2]
                no_node = cur_decision[3]
                missing_node = cur_decision[4]
                if features[feat_id] < split_thr:
                    cur_decision = decision_nodes[yes_node]
                else:
                    cur_decision = decision_nodes[no_node]
        path_ids.append(sub_tree_path)
        hit_feats.append(sub_hit_nodes)
    prob = 1.0 /  ( 1 + math.exp( -1 * boosting_value) )
    return {'prob':prob}

def plotXgboostTree():
    xgb_model = xgb.Booster(model_file="./rank_model/query_weight_xgb1.model")
    xgbclf = joblib.load('./rank_model/xgb_clf.m')
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

def static_tree_feature(xgb_dict):
    tree_features = []
    # static feature
    for i in xgb_dict:
        tree_feature = {}
        tree = xgb_dict[i]['decision_nodes']
        for node in tree:
            if len(tree[node]) <2:
                continue
            if tree[node][0] not in tree_feature:
                tree_feature[tree[node][0]] = []
            tree_feature[tree[node][0]].append(float(tree[node][1]))
        tree_features.append(tree_feature)
    # unique and sort
    static_tree_features = []
    for tf in tree_features:
        stf  ={}
        for e in tf:
            sl = list(set(tf[e]))
            sl.sort()
            stf[e] = sl
        static_tree_features.append(stf)
    return static_tree_features


def handxgbmodel(feature):
    #xgb_dict = parse_xgb_dict('./xgb.dump')
    xgb_dict = parse_xgb_dict('D:/Java Project/trunk/dict/intention_predict/xgb.dump')
    tree_features = static_tree_feature(xgb_dict)
    result = predict_proba_(xgb_dict, np.array([feature], dtype='float64'), tree_features)

    pass

if __name__ == '__main__':
    plotXgboostTree()
    #feature = [ 2, 5, 1, 0, 0, 1, 0, 0, -1, -1]
#    origionProb = joblib.load('./xgb_clf.m').predict_proba(np.array([feature]))[:, 1]
    '''
    jl = joblib.load('./xgb_clf.m')
    x = np.array([feature])
    oP = jl.predict_proba(x)
    '''
    feature = [ 2, 0, -1, 0, 2, 1, 2, 1, -1, -1]
    handxgbmodel(feature)