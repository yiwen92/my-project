#!/usr/bin/env python
#coding:utf8
#from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

from datetime import date, timedelta
import random
import tensorflow as tf

from utils import readdata, FIELD_SIZES, read_libsvm, calScore
from models import LR
import progressbar
from utils import slice_data, slice_libsvm
from sklearn.metrics import roc_auc_score
import numpy as np
import os, shutil, math

random.seed(0)

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("train", False, "Whether train the model")
tf.app.flags.DEFINE_boolean("clean", False, "clean model file")
tf.app.flags.DEFINE_string("data_dir", './data/simjdtraindata', "data dir")
tf.app.flags.DEFINE_string("model_dir", './model_LR/LR', "model check point dir")
tf.app.flags.DEFINE_integer("feature_size", 28, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 18, "Number of features")
tf.app.flags.DEFINE_float("split_ratio", 0.8, "Split ratio of train set")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_integer("num_round", 20000, "Number of round")
tf.app.flags.DEFINE_integer("min_round", 200, "Number of min round")
tf.app.flags.DEFINE_integer("early_stop_round", 2000, "Number of early stop round")

if FLAGS.dt_dir == "":  FLAGS.dt_dir = (date.today() + timedelta(1 - 1)).strftime('%Y%m%d')
FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

if FLAGS.train:
    data = read_libsvm(FLAGS.data_dir);    random.shuffle(data)
    train_data = data[:(int)(len(data) * FLAGS.split_ratio)];    test_data = data[(int)(len(data) * FLAGS.split_ratio):]
    print('read finish');  print('train data size:', (len(train_data), len(train_data[0][0])));  print('test data size:', (len(test_data), len(test_data[0][0])))
    train_size = len(train_data); test_size = len(test_data)
    min_round = FLAGS.min_round;  num_round = FLAGS.num_round;  early_stop_round = FLAGS.early_stop_round;  batch_size = FLAGS.batch_size

lr_params = {
    'field_size': FLAGS.field_size,
    'feature_size': FLAGS.feature_size,
    'learning_rate': FLAGS.learning_rate,
    'optimizer': FLAGS.optimizer
}

model = LR(**lr_params)

if FLAGS.clean and os.path.isdir(os.path.dirname(FLAGS.model_dir)) and FLAGS.train:
    print('Cleaning ckpt file...')
    shutil.rmtree(os.path.dirname(FLAGS.model_dir))

ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.model_dir))
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(model.sess, ckpt.model_checkpoint_path)  # 读取保存的模型
else:
    print("Created model with fresh parameters.")
    model.sess.run(tf.global_variables_initializer())  # 创建新的模型，并初始化模型

def train():
    history_score = [];  max_score = -1
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                feat_ids, feat_vals, label = slice_libsvm(train_data, j * batch_size, batch_size)
                #a = model.run_step(model.pred_prob, feat_ids, feat_vals, label)
                _, l = model.run_step(fetches, feat_ids, feat_vals, label)
                ls.append(l)
        elif batch_size == -1:
            feat_ids, feat_vals, label = slice_libsvm(train_data)
            _, l = model.run_step(fetches, feat_ids, feat_vals, label)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            feat_ids, feat_vals, label = slice_libsvm(train_data, j * 10000, 10000)
            preds = model.run_step(model.pred_prob, feat_ids, feat_vals, label)
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            feat_ids, feat_vals, label = slice_libsvm(test_data, j * 10000, 10000)
            preds = model.run_step(model.pred_prob, feat_ids, feat_vals, label)
            #auc = model.run_step(model.auc, feat_ids, feat_vals, label)
            test_preds.extend(preds)
        train_true = [];    test_true = []
        for e in train_data:
            train_true.append(e[2])
        for e in test_data:
            test_true.append(e[2])
        train_score = roc_auc_score(train_true, train_preds)
        test_score = roc_auc_score(test_true, test_preds)
        trprecision, trrecall, tracc = calScore(train_true, train_preds)
        teprecision, terecall, teacc = calScore(test_true, test_preds)
        print('[%d]\tloss: %f\ttrain-auc: %f\teval-auc: %f\t\tprecision: %f\trecall: %f\ttrain-acc: %f\ttest-acc: %f'
              % (i, np.mean(ls), train_score, test_score, teprecision, terecall, tracc, teacc))
        history_score.append(test_score)
        if test_score > max_score:
            model.save_model(FLAGS.model_dir)
            max_score = test_score
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[-1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (np.argmax(history_score), np.max(history_score)))
                model.save_model(FLAGS.model_dir)
                break

def predict_nn(feature):
    # b = model.sess.run(model.b); w1 = model.sess.run(model.w1)
    fea_ids = []; fea_vals = []
    fea_id = [int(e.split(':')[0]) - 1 for e in feature]
    fea_val = [float(e.split(':')[1]) for e in feature]
    fea_ids.append(fea_id); fea_vals.append(fea_val)
    pred_prob = model.run_step(model.pred_prob, fea_ids, fea_vals, mode = 'predict')
    return pred_prob

if __name__ == "__main__":
    feature = ['2:1', '4:0.95388', '5:0.777509', '6:0', '9:1', '10:2', '11:5', '12:4', '13:4', '16:1', '19:1',
           '20:1.95', '21:3.5', '22:-1.0', '23:-1', '26:1', '27:1', '28:1']
    #predict_nn(model, feature)    ;   exit()

    if FLAGS.train:
        print(lr_params)
        train()
    else:
        predict_nn(feature)
