#!/usr/bin/env python

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from datetime import date, timedelta
import random
import tensorflow as tf

from utils import readdata, FIELD_SIZES
from models import DeepFFM
import progressbar
from utils import slice_data
from sklearn.metrics import roc_auc_score
import numpy as np
import os, shutil

random.seed(0)

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("train", False, "Whether train the model")
tf.app.flags.DEFINE_boolean("clean", False, "clean model file")
tf.app.flags.DEFINE_string("data_dir", './data/simjdtraindata1', "data dir")
tf.app.flags.DEFINE_string("model_dir", './model/DeepFM', "model check point dir")
tf.app.flags.DEFINE_integer("feature_size", 19, "Number of features")
tf.app.flags.DEFINE_integer("latent_vector_size", 40, "Number of features")
tf.app.flags.DEFINE_float("split_ratio", 0.7, "Split ratio of train set")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0, "L2 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("active_function", 'relu,relu,relu', "active function")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")

tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")

if FLAGS.dt_dir == "":  FLAGS.dt_dir = (date.today() + timedelta(1 - 1)).strftime('%Y%m%d')
FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

if FLAGS.train:
    data = readdata(FLAGS.data_dir, FLAGS.feature_size);    random.shuffle(data)
    train_data = data[:(int)(len(data) * FLAGS.split_ratio)];    test_data = data[(int)(len(data) * FLAGS.split_ratio):]
    print('read finish');  print('train data size:', (len(train_data), len(train_data[0][0])));  print('test data size:', (len(test_data), len(test_data[0][0])))
    train_size = len(train_data); test_size = len(test_data)
    min_round = 1;  num_round = 20;  early_stop_round = 5;  batch_size = FLAGS.batch_size


deep_ffm_params = {
    'train': FLAGS.train,
    'feature2field': FIELD_SIZES,
    'layer_sizes': FLAGS.deep_layers,
    'layer_acts': FLAGS.active_function,
    'drop_out': FLAGS.dropout,
    'opt_algo': FLAGS.optimizer,
    'learning_rate': FLAGS.learning_rate,
    'l2_reg': FLAGS.l2_reg,
    'feature_size': FLAGS.feature_size,
    'field_size': len(set(FIELD_SIZES)),
    'k': FLAGS.latent_vector_size
}
model = DeepFFM(**deep_ffm_params)

def train(model):
    history_score = [];  max_score = -1
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = slice_data(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = slice_data(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            X_i, _ = slice_data(train_data, j * 10000, 10000)
            preds = model.run(model.pred_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            X_i, _ = slice_data(test_data, j * 10000, 10000)
            preds = model.run(model.pred_prob, X_i, mode='test')
            test_preds.extend(preds)
        train_true = [];    test_true = []
        for e in train_data:
            train_true.append(e[1])
        for e in test_data:
            test_true.append(e[1])
        train_score = roc_auc_score(train_true, train_preds)
        test_score = roc_auc_score(test_true, test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if test_score > max_score:
            model.save_model(FLAGS.model_dir)
            max_score = test_score
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[-1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (np.argmax(history_score), np.max(history_score)))
                model.save_model(FLAGS.model_dir)
                break

def predict(model):
    feature_input = []
    featch = []
    feature_input.append([1, 0.941897666667, 0.930052666667, 1.0127358379, 0, 2, 3, 9, 5, 5, 1, 0, 1, 1, 2.0, 1.75, -1.0, -1, 1])
    #b = model.sess.run(model.b); w1 = model.sess.run(model.w1)

    aa = model.sess.run(model.pred_prob, {model.X: feature_input})

    res = {}
    for i in range(1000):
        a= model.sess.run(model.pred_prob, {model.X : feature_input})   #;a=a.tolist()
        if a[0] not in res:  res[a[0]] = 0
        res[a[0]] += 1
    s=1



if __name__ == "__main__":
    if FLAGS.clean and os.path.isdir(os.path.dirname(FLAGS.model_dir)) and FLAGS.train:
        print('Cleaning ckpt file...')
        shutil.rmtree(os.path.dirname(FLAGS.model_dir))

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.model_dir))
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(model.sess, ckpt.model_checkpoint_path)        # 读取保存的模型
    else:
        print("Created model with fresh parameters.")
        model.sess.run(tf.global_variables_initializer())  # 创建新的模型，并初始化模型
    if FLAGS.train:
        print(deep_ffm_params)
        train(model)
    else:
        predict(model)
