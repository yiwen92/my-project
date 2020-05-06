#!/usr/bin python
#-*- coding:utf-8 -*-
'''
@author:slack
date:2016/09/27
'''
from __future__ import division
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import time, os, random, sys, json
import math
#import hyperopt.tpe
#import kaggle_auc
import matplotlib.pyplot as plt
from plot_roc_prc import *
#import hpsklearn.components
#import hpsklearn.demo_support
import logging
#random.seed(1)

from numpy import arange
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

#reload(sys)
#sys.setdefaultencoding('utf8')

total = 0
split_ratio = 0.3     #0.3	test set

def init():
    logging.basicConfig(level=logging.INFO,
            format='[%(levelname)s] %(asctime)s %(filename)s [line:%(lineno)d] %(message)s',
            datefmt='[%Y-%m_%d %H:%M:%S]',
            filename='model.log',
            filemode='a')

"""
mem = joblib.Memory("./mycache")
@mem.cache
def cache_data(libsvm_dataset):
    data = load_svmlight_file(libsvm_dataset)
    return data[0],data[1]
"""
def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}

def dump_text_model(xgb, model_path):
    bst = xgb._Booster
    bst.dump_model(model_path+'xgb.dump') #, model_path+'xgb.fmap')
    xgdump = bst.get_dump()
    #print xgdump
    print('dump model ok')


#choosing some samples and random split into train set and test set
def split_dataset(libSvmFile, trainFileName, testFileName, testSetRatio, lines):
    dataFile = open(libSvmFile, 'r')
    dataList = dataFile.readlines()
    totalLines = len(dataList)
    testFileLength = int(testSetRatio*totalLines)
    trainFileLength = totalLines - testFileLength
    List = [i for i in range(totalLines)]           #List = range(totalLines)
    random.shuffle(List)
    trainFile = open(trainFileName, 'w')
    testFile = open(testFileName, 'w')
    posSampleCnt = 0
    for i in range(totalLines):
        if float(dataList[List[i]].split(' ')[0]) > 0.0:
            posSampleCnt = posSampleCnt + 1
        if i < trainFileLength:
            trainFile.write(dataList[List[i]])
        else:
            testFile.write(dataList[List[i]])
    dataFile.close()
    trainFile.close()
    testFile.close()
    logging.info('positive samples : %d' % posSampleCnt)
    return posSampleCnt

#calculate the positive and negative samples counts
def pos_neg_dist(libSvmFile):
    global total
    dataFile = open(libSvmFile, 'r')
    dataList = dataFile.readlines()
    total = len(dataList)
    posSampleCnt = 0
    negSampleCnt = 0
    for i in range(len(dataList)):
        if float(dataList[i].split(' ')[0]) > 0.0:
            posSampleCnt = posSampleCnt + 1
        else:
            negSampleCnt = negSampleCnt + 1
    logging.info('P/N = %d/%d' % (posSampleCnt,negSampleCnt))

#training xgboost and using xgboost to encode test set features
def xgboost_lr_train_test(libsvmFileNameInitial, valid_data_path, model_path):
    print('train file:',libsvmFileNameInitial,'model file',model_path)
    posSampleCnt = split_dataset(libsvmFileNameInitial, model_path+'data_train_edr', model_path+'data_test_edr', split_ratio, total)
    X_train, y_train = load_svmlight_file(model_path+'data_train_edr')
    logging.info(X_train.shape)	#;print('X_train',X_train.shape;exit())
    X_test, y_test = load_svmlight_file(model_path+'data_test_edr')
    X_valid, y_valid = load_svmlight_file(valid_data_path)
    #print('X_train\n',X_train,'\ny_train',y_train,'\nX_train',X_train.shape,'y_train',len(y_train);exit())

    #training xgboost
    negPosRatio = (total - posSampleCnt)/posSampleCnt   # sum_neg / sum_pos
    print('total',total,'posSampleCnt',posSampleCnt,'negPosRatio',negPosRatio)
    #gbdt clf
    '''
    xgbclf = xgb.XGBClassifier(\
            nthread=4,
            scale_pos_weight=negPosRatio,
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            subsample=0.5,
            reg_lambda = 5,
            colsample_bytree=0.6,
            missing=0.0,
            silent=True)
    '''

    xgbclf = xgb.XGBClassifier( \
            #objective = 'multi:softprob',
            nthread=4,\
            scale_pos_weight=negPosRatio,\
            learning_rate=0.1,\
            n_estimators=100,\
            max_depth=6,\
            subsample=0.1,\
            reg_lambda = 5,\
            colsample_bytree=0.6,\
            missing=0.0,\
            silent=True)
##############################################################################################
    '''
    param_grid = {
        'max_depth': range(3, 10, 2),
        #'n_estimators': range(100, 200, 2),
        'learning_rate': np.linspace(0.01, 0.1, 5)
    }
    gsearch = GridSearchCV(estimator=xgbclf, param_grid=param_grid, cv=5)
    gsearch.fit(X_train, y_train)
    xgbclf = xgb.XGBClassifier(nthread=4, scale_pos_weight=negPosRatio, reg_lambda=5, colsample_bytree=0.6, missing=0.0, silent=True, subsample=0.1,
                               learning_rate=gsearch.best_params_['learning_rate'],
                               #n_estimators=gsearch.best_estimator_['n_estimators'],
                               max_depth=gsearch.best_params_['max_depth']       )
    print(gsearch.best_params_, gsearch.best_score_)
    '''
##############################################################################################
    #gsearch.
#    xgbclf = xgb.XGBClassifier()
    xgbclf.fit(X_train, y_train)

    pos_neg_dist(model_path+'data_train_edr')
    pos_neg_dist(model_path+'data_test_edr')
    #save model
    joblib.dump(xgbclf,model_path+'xgb_clf.m')
    dump_text_model(xgbclf, model_path)
    xgbclf.predict()
    y_pred_valid = xgbclf.predict_proba(X_valid)[:, 1]
    fpr, tpr, dist = roc_curve(y_valid, y_pred_valid)
    roc_auc = auc(fpr, tpr)
    print('valid roc_auc', roc_auc)
    '''
    plt.clf()
    xgb.plot_tree(xgbclf, num_trees = 99)
    plt.savefig('xgb_tree.png', dpi=800,format='png')
    '''
    """
    feat_names = json.load(open('../models/xgb.feat.name','r'))
    fimp = get_xgb_imp(xgbclf,feat_names)
    for fm in sorted(fimp.iteritems(),key = lambda d:d[1],reverse=True):
        print(fm[1],'\t',fm[0])
    """
    # save figures
    plt.clf()
    xgb.plot_importance(xgbclf)
    plt.savefig('feature_importance.png', dpi=800,format='png')
    print('train set:',X_train.shape,'\ttest set:',X_test.shape)

    logging.info(xgbclf.feature_importances_/ sum (xgbclf.feature_importances_))
    y_pred_train = xgbclf.predict_proba(X_train)[:, 1]
    try:        xgb_train_auc = roc_auc_score(y_train, y_pred_train)
    except:     xgb_train_auc = 1
    logging.info('xgboost train auc: %.5f' % xgb_train_auc)
    #auc
    y_pred_test = xgbclf.predict_proba(X_test)[:, 1]
    try:        xgb_test_auc = roc_auc_score(y_test, y_pred_test)
    except:     xgb_test_auc = 1
    logging.info('xgboost test auc: %.5f' % xgb_test_auc)
    print('xgboost train auc: %.5f' % xgb_train_auc,',\ttest auc: %.5f' % xgb_test_auc)
    #acc,f1,pr
    print('[gbdt]')
    y_pred = xgbclf.predict(X_test)	#;print('y_pred',y_pred;exit())
    y_true = y_test
    print('acc',accuracy_score(y_true,y_pred))
    print('p',precision_score(y_true,y_pred))
    print('r',recall_score(y_true,y_pred))
    print('f',f1_score(y_true,y_pred))

    #print('y_pred_train\n',y_pred_train,'\ny_train\n',y_train)
    #print('y_pred_test\n',y_pred_test,'\ny_test\n',y_test)

    plot_roc(y_test,y_pred_test)

#    exit()

    #using xgboost to encode train set and test set features
    X_train_leaves = xgbclf.apply(X_train)	#;print('X_train_leaves\n',X_train_leaves)
    gbdt_leaf_enc = OneHotEncoder()	#;print('gbdt_leaf_enc',gbdt_leaf_enc)
    gbdt_leaf_enc.fit(X_train_leaves)
    joblib.dump(gbdt_leaf_enc,'./model/leaf_enc.m')
    #X_test_leaves = xgbclf.apply(X_test)
    X_valid_leaves = xgbclf.apply(X_valid)

    X_train_s = gbdt_leaf_enc.transform(X_train_leaves)
    #X_test_s = gbdt_leaf_enc.transform(X_test_leaves)
    X_valid_s = gbdt_leaf_enc.transform(X_valid_leaves)

    #fittting lr using just xgboost encoded feature
    lr = LogisticRegression(n_jobs=-1, C=0.4, penalty='l1')
    lr.fit(X_train_s, y_train)
    joblib.dump(lr, './model/lr_clf.m')
    y_pred_train = lr.predict_proba(X_train_s)[:, 1]
    lr_train_auc = roc_auc_score(y_train, y_pred_train)
    logging.info('LR Train AUC: %.5f' % lr_train_auc) ;print('LR Train AUC: %.5f' % lr_train_auc)
    #y_pred_test = lr.predict_proba(X_test_s)[:,1]
    #lr_test_auc = roc_auc_score(y_test, y_pred_test)
    y_pred_valid = lr.predict_proba(X_valid_s)[:, 1]
    lr_valid_auc = roc_auc_score(y_valid, y_pred_valid)


    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    print('fpr',fpr)
    print('tpr',tpr)
    """
    logging.info('LR Test AUC: %.5f' % lr_valid_auc);print('LR Test AUC: %.5f' % lr_valid_auc)

    #plot_roc(y_test,y_pred_test)

    #plot_prc(y_test,y_pred_test)
    """
    (rows, cols) = X_leaves.shape
    cum_count = np.zeros((1, cols), dtype=np.int32)
    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(X_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(X_leaves[:, j])) + cum_count[0][j-1]
    logging.info('Transform features genenrated by xgboost...')
    for j in range(cols):
        keyMapDict = {}
        if j == 0:
            initial_index = 1
        else:
            initial_index = cum_count[0][j-1]+1
        for i in range(rows):
            if keyMapDict.has_key(X_leaves[i, j]) == False:
                keyMapDict[X_leaves[i, j]] = initial_index
                X_leaves[i, j] = initial_index
                initial_index = initial_index + 1
            else:
                X_leaves[i, j] = keyMapDict[X_leaves[i, j]]
    #writing encoded features into file
    logging.info('Write xgboost learned features to file ...')
    xgbFeatureLibsvm = open('./model_train/xgb_feature_libsvm', 'w')
    for i in range(rows):
        if i < train_rows:
            xgbFeatureLibsvm.write(str(y_train[i]))
        else:
            xgbFeatureLibsvm.write(str(y_test[i-train_rows]))
        for j in range(cols):
            xgbFeatureLibsvm.write(' '+str(X_leaves[i, j])+':1.0')
        xgbFeatureLibsvm.write('\n')
    xgbFeatureLibsvm.close()
    """

#using xgboost encoded feature in lr to calculate auc
def xgb_feature_lr_train_test(xgbfeaturefile, origin_libsvm_file):
    split_dataset(origin_libsvm_file, './model_train/data_train_edr', './model_train/data_test_edr', split_ratio, 1100000)
    split_dataset(xgbfeaturefile, './model_train/xgb_feature_train_libsvm','./model_train/xgb_feature_test_libsvm', split_ratio, 1100000)
    X_train_origin, y_train_origin = load_svmlight_file('./model_train/data_train_edr')
    X_test_origin, y_test_origin = load_svmlight_file('./model_train/data_test_edr')
    X_train, y_train = load_svmlight_file('./model_train/xgb_feature_train_libsvm')
    logging.info(X_train.shape)
    X_test, y_test = load_svmlight_file('./model_train/xgb_feature_test_libsvm')
    logging.info(X_test.shape)

    #fittting lr using just xgboost encoded feature
    lr = LogisticRegression(n_jobs=-1, C=0.2, penalty='l1')
    lr.fit(X_train, y_train)
    joblib.dump(lr, '../models/lr_clf.m')
    y_pred_train = lr.predict_proba(X_train)[:, 1]
    lr_train_auc = roc_auc_score(y_train, y_pred_train)
    pos_neg_dist('./model_train/xgb_feature_train_libsvm')
    logging.info('LR Train AUC: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test)[:,1]
    lr_test_auc = roc_auc_score(y_test, y_pred_test)

    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    print('fpr',fpr)
    print('tpr',tpr)
    """
    pos_neg_dist('./model_train/xgb_feature_test_libsvm')
    logging.info('LR Test AUC: %.5f' % lr_test_auc)

    #plot_roc(y_test,y_pred_test)

    #plot_prc(y_test,y_pred_test)

    plot_prc_roc(y_test,y_pred_test)




    #acc,f1,pr
    print('[gbdt->lr]')
    y_pred = lr.predict(X_test)
    y_true = y_test
    print('acc',accuracy_score(y_true,y_pred))
    print('p',precision_score(y_true,y_pred))
    print('r',recall_score(y_true,y_pred))
    print('f',f1_score(y_true,y_pred))
    # fitting lr using xgboost encoded feature and original feature
    """
    X_train_ext = hstack([X_train_origin, X_train])
    logging.info(X_train_ext.shape)
    del(X_train)
    del(X_train_origin)
    X_test_ext = hstack([X_test_origin, X_test])
    logging.info(X_test_ext.shape)
    del(X_test)
    del(X_test_origin)
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr.fit(X_train_ext, y_train)
    joblib.dump(lr, './model_train/lr_ext.m')
    y_pred_train = lr.predict_proba(X_train_ext)[:, 1]
    lr_train_auc = roc_auc_score(y_train, y_pred_train)
    logging.info('LR Ext Train AUC: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test_ext)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_pred_test)
    logging.info('LR Ext Test AUC: %.5f' % lr_test_auc)
    """

#using gbdt, gbdt+lr to calculate auc
def gbdt_lr_train_test(libsvmFileName):
    split_dataset(libsvmFileName, './model_train/label_feature_data_train', './model_train/label_feature_data_test',split_ratio, total)
    X_train, y_train = load_svmlight_file('./model_train/label_feature_data_train')
    X_test, y_test = load_svmlight_file('./model_train/label_feature_data_test')
    gbclf = GradientBoostingClassifier(n_estimators=30, max_depth=4, verbose=0)
    tuned_parameter = [{'n_estimators':[30, 40, 50,60], 'max_depth':[3, 4, 5, 6, 7, 8, 9], 'max_features':[0.4,0.5,0.6,0.7,0.8,0.9]}]
    gs_clf = GridSearchCV(gbclf, tuned_parameter, cv=5, scoring='roc_auc')
    gs_clf.fit(X_train.toarray(), y_train)
    logging.info('best parameters set found: ')
    logging.info(gs_clf.best_params_)
    y_pred_gbdt = gs_clf.predict_proba(X_test.toarray())[:, 1]
    gbdt_auc = roc_auc_score(y_test, y_pred_gbdt)
    logging.info('gbdt auc: %.5f' % gbdt_auc)
    X_train_leaves = gbclf.apply(X_train)[:,:,0]
    (train_rows, cols) = X_train_leaves.shape
    X_test_leaves = gbclf.apply(X_test)[:,:,0]
    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))
    lr = LogisticRegression()
    lr.fit(X_trans[:train_rows, :], y_train)
    y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    gbdtlr_auc1 = roc_auc_score(y_test, y_pred_gbdtlr1)
    logging.info('gbdt+lr auc 1: %.5f' % gbdtlr_auc1)
    lr = LogisticRegression(n_jobs=-1)
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    lr.fit(X_train_ext, y_train)
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])
    y_pred_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
    gbdtlr_auc2 = roc_auc_score(y_test, y_pred_gbdtlr2)
    logging.info('gbdt+lr auc 2: %.5f' % gbdtlr_auc2)

#using lr to calculate auc on original data and cross featured data
def lr_train_test(libsvmFileInitial, libsvmFileCross):
    split_dataset(libsvmFileInitial, './model_train/data_train_th500', './model_train/data_test_th500', split_ratio, 1100000)
    split_dataset(libsvmFileCross, './model_train/data_cross_train_th500', './model_train/data_cross_test_th500', split_ratio, 1100000)
    X_train_origin, y_train_origin = load_svmlight_file('./model_train/data_train_th500')
    logging.info(X_train_origin.shape)
    X_test_origin, y_test_origin = load_svmlight_file('./model_train/data_test_th500')
    logging.info(X_test_origin.shape)
    lr = LogisticRegression(C=0.1, penalty='l2')
    lr.fit(X_train_origin, y_train_origin)
    y_pred_train = lr.predict_proba(X_train_origin)[:, 1]
    lr_train_auc = roc_auc_score(y_train_origin, y_pred_train)
    logging.info('lr train auc origin: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test_origin)[:, 1]
    lr_test_auc = roc_auc_score(y_test_origin, y_pred_test)
    logging.info('lr test auc origin: %.5f' % lr_test_auc)
    X_train_cross, y_train_cross = load_svmlight_file('./model_train/data_cross_train_th500')
    logging.info(X_train_cross.shape)
    X_test_cross, y_test_cross = load_svmlight_file('./model_train/data_cross_test_th500')
    logging.info(X_test_cross.shape)
    lr = LogisticRegression(C=0.1, penalty='l2')
    lr.fit(X_train_cross, y_train_cross)
    y_pred_train = lr.predict_proba(X_train_cross)[:, 1]
    lr_train_auc = roc_auc_score(y_train_cross, y_pred_train)
    logging.info('lr train auc cross: %.5f' % lr_train_auc)
    y_pred_test = lr.predict_proba(X_test_cross)[:, 1]
    lr_test_auc = roc_auc_score(y_test_cross, y_pred_test)
    logging.info('lr test auc cross: %.5f' % lr_test_auc)

#using hyperopt-sklearn to automatically tune the parameters of gbdt
'''
def hyper_opt(libsvmFile):
    split_dataset(libsvmFile, './model_train/data_train_edr', './model_train/data_test_edr', split_ratio, 100000)
    X_train, y_train = load_svmlight_file('./model_train/data_train_edr')
    X_train = X_train.toarray()
    estimator = hpsklearn.HyperoptEstimator(None,
                                            classifier=hpsklearn.components.any_classifier('clf'),
                                            algo=hyperopt.tpe.suggest,
                                            trial_timeout=10.0,
                                            max_evals=10)
    fit_iterator = estimator.fit_iter(X_train, y_train)
    fit_iterator.next()
    plot_helper = hpsklearn.demo_support.PlotHelper(estimator, mintodate_ylim=(0.0,0.1))
    while len(estimator.trials.trials) < estimator.max_evals:
        fit_iterator.send(1)
        plot_helper.post_iter()
    plot_helper.post_loop()
    estimator.retrain_best_model_on_full_data(X_train, y_train)
    logging.info('Best classifier: \n %s' % str(estimator.best_model()))
'''

if __name__ == '__main__':
    train_data_path = './data/train.data'
    train_data_path = './data/multi_class_train.data'
    #train_data_path = './data/train_data_70956'
    #train_data_path = './data/train_data_35640'
    #train_data_path = './data/static_feature'
    train_data_path = './data/job_intention_train_28707'
    train_data_path = './data/simjdtraindata'
    valid_data_path = './data/validata'
    valid_data_path = './data/simjdtraindata1'

    model_path = './model/'
    #model_path = './model_job_intention/'

    init()
    logging.info('_handler_edr_XgBoost_LR__')
    pos_neg_dist(train_data_path)	#;print(123);exit()
    #split_dataset('../data/train.data', 0.2, 'lr_data_train', 'lr_data_test', 600000)
    xgboost_lr_train_test(train_data_path, valid_data_path, model_path)	;print('Finish Train');exit()
    #lr_train_test('./model_train/data_libsvm_th500', './model_train/data_cross_libsvm_th500')
    #xgb_feature_lr_train_test('./model_train/xgb_feature_libsvm', '../data/train.data')
    #gbdt_lr_train_test('../data/train.data')
    #hyper_opt('../data/train.data')
