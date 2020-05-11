#!/usr/bin/env python
#coding:utf8
#from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

from datetime import date, timedelta
import random
import tensorflow as tf

from utils import readdata, FIELD_SIZES, read_libsvm, calScore
from models import FNN
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
tf.app.flags.DEFINE_string("data_dir", './data/simjdtraindata1', "data dir")
tf.app.flags.DEFINE_string("model_dir", './model_FNN/FNN', "model check point dir")
tf.app.flags.DEFINE_integer("feature_size", 28+3, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 18+3, "Number of features")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Number of features")
tf.app.flags.DEFINE_float("split_ratio", 0.8, "Split ratio of train set")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")         # 1024,512,256,128,64
tf.app.flags.DEFINE_string("active_function", 'relu,relu,relu', "active function")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
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
fields = [1] * 18; fields[0] = 3; fields[4] = 3; fields[9] = 3; fields[10] = 3; fields[15] = 3; a=sum(fields)
localid = []
for e in fields:
    localid.extend([i for i in range(e)])
fnn_params = {
    'fields': fields,
    'feature_size': FLAGS.feature_size,
    'embedding_size': FLAGS.embedding_size,
    'learning_rate': FLAGS.learning_rate,
    'optimizer': FLAGS.optimizer,
    'layer_sizes': FLAGS.deep_layers,
    'layer_acts': FLAGS.active_function,
    'drop_out': FLAGS.dropout,
    'train': FLAGS.train
}

model = FNN(**fnn_params)

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
                features, label = slice_libsvm(train_data, j * batch_size, batch_size, True, FLAGS.feature_size)
                _, l = model.runstep(fetches, features, label)
                ls.append(l)
        elif batch_size == -1:
            features, label = slice_libsvm(train_data, dense=True, featureNum=FLAGS.feature_size)
            _, l = model.runstep(fetches, features, label)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            features, label = slice_libsvm(train_data, j * 10000, 10000, True, FLAGS.feature_size)
            preds = model.runstep(model.pred_prob, features, label)
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            features, label = slice_libsvm(test_data, j * 10000, 10000, True, FLAGS.feature_size)
            preds = model.runstep(model.pred_prob, features, label)
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
    fea = [0] * FLAGS.feature_size
    for e in feature:
        fea[int(e.split(':')[0]) - 1] = float(e.split(':')[1])
    features = []
    features.append(fea)
    pred_prob = model.runstep(model.pred_prob, features, mode = 'predict')
    return pred_prob

if __name__ == "__main__":
    feature = ['2:1', '4:0.95388', '5:0.777509', '6:0', '9:1', '10:2', '11:5', '12:4', '13:4', '16:1', '19:1',
           '20:1.95', '21:3.5', '22:-1.0', '23:-1', '26:1', '27:1', '28:1', '29:0.3', '30:0.4', '31:0.5']

    #predict_nn(model, feature)    ;   exit()

    if FLAGS.train:
        print(fnn_params)
        train()
    else:
        predict_nn(feature)
