#!/usr/bin/env python
#coding:utf8
#from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

from datetime import date, timedelta
import random
import tensorflow as tf

from utils import readdata, FIELD_SIZES, read_libsvm, calScore
from models import DeepFFM, DeepAFM
import progressbar
from utils import slice_data, slice_libsvm
from sklearn.metrics import roc_auc_score
import numpy as np
import os, shutil, math

random.seed(0)

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("train", True, "Whether train the model")
tf.app.flags.DEFINE_boolean("clean", False, "clean model file")
tf.app.flags.DEFINE_string("data_dir", './data/simjdtraindata11', "data dir")
tf.app.flags.DEFINE_string("valid_dir", './data/simjdvalid_data', "data dir")
tf.app.flags.DEFINE_string("model_dir", './model_DeepAFM/DeepAFM', "model check point dir")
tf.app.flags.DEFINE_string("feature_dir", './data/deep_afm_fea.libsvm', "feature data dir")
tf.app.flags.DEFINE_integer("feature_size", 28+3, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 18+3, "Number of features")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Number of features")
tf.app.flags.DEFINE_float("split_ratio", 0.9, "Split ratio of train set")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '64,32', "deep layers")         # 1024,512,256,128,64
tf.app.flags.DEFINE_string("active_function", 'relu,relu,relu', "active function")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_integer("num_round", 20000, "Number of round")
tf.app.flags.DEFINE_integer("min_round", 200, "Number of min round")
tf.app.flags.DEFINE_integer("early_stop_round", 20, "Number of early stop round")

tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")

if FLAGS.dt_dir == "":  FLAGS.dt_dir = (date.today() + timedelta(1 - 1)).strftime('%Y%m%d')
FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

if FLAGS.train:
    data = read_libsvm(FLAGS.data_dir);    random.shuffle(data);    valid_data = read_libsvm(FLAGS.valid_dir)
    train_data = data[:(int)(len(data) * FLAGS.split_ratio)];    test_data = data[(int)(len(data) * FLAGS.split_ratio):]
    print('read finish');  print('train data size:', (len(train_data), len(train_data[0][0])));  print('test data size:', (len(test_data), len(test_data[0][0])))
    train_size = len(train_data); test_size = len(test_data)
    min_round = FLAGS.min_round;  num_round = FLAGS.num_round;  early_stop_round = FLAGS.early_stop_round;  batch_size = FLAGS.batch_size

deep_afm_params = {
    'field_size': FLAGS.field_size,
    'feature_size': FLAGS.feature_size,
    'embedding_size': FLAGS.embedding_size,
    'l2_reg': FLAGS.l2_reg,
    'learning_rate': FLAGS.learning_rate,
    'optimizer': FLAGS.optimizer,
    'layer_sizes': FLAGS.deep_layers,
    'layer_acts': FLAGS.active_function,
    'drop_out': FLAGS.dropout,
    'train': FLAGS.train
}

model = DeepAFM(**deep_afm_params)

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
        fetches = [model.optimizer, model.loss]#, model.attention_wgh, model.attention_out]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                feat_ids, feat_vals, label = slice_libsvm(train_data, j * batch_size, batch_size)
                #a = model.run_step(model.regularization, feat_ids, feat_vals, label)
                #for i in range(len(feat_vals)): feat_vals[i][18] /= 1000;feat_vals[i][19] /= 10;feat_vals[i][20] /= 10      # ********************
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
            preds = model.run_step(model.pred_prob, feat_ids, feat_vals, mode = 'predict')
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            feat_ids, feat_vals, label = slice_libsvm(test_data, j * 10000, 10000)
            preds = model.run_step(model.pred_prob, feat_ids, feat_vals, mode = 'predict')
            #auc = model.run_step(model.auc, feat_ids, feat_vals, label)
            test_preds.extend(preds)
        val_feat_ids, val_feat_vals, val_label = slice_libsvm(valid_data)
        val_pred = model.run_step(model.pred_prob, val_feat_ids, val_feat_vals, mode = 'predict')
        val_preds = []; val_preds.extend(val_pred)
        train_true = [];    test_true = []
        for e in train_data:
            train_true.append(e[2])
        for e in test_data:
            test_true.append(e[2])
        train_score = roc_auc_score(train_true, train_preds)
        test_score = roc_auc_score(test_true, test_preds)
        valide_score = roc_auc_score(val_label, val_preds)
        trprecision, trrecall, tracc = calScore(train_true, train_preds)
        teprecision, terecall, teacc = calScore(test_true, test_preds)
        valrecision, valrecall, valacc = calScore(val_label, val_preds)
        print('[%d]\tloss: %f\ttrain-auc: %f\teval-auc: %f\tval-auc: %f\tvalrecision: %f\tvalrecall: %f\tvalacc: %f'
              % (i, np.mean(ls), train_score, test_score, valide_score, valrecision, valrecall, valacc))
        #print('[%d]\tloss: %f\ttrain-auc: %f\teval-auc: %f\tprecision: %f\trecall: %f\ttrain-acc: %f\ttest-acc: %f'
        #      % (i, np.mean(ls), train_score, test_score, teprecision, terecall, tracc, teacc))
        his_score = valide_score
        history_score.append(his_score)
        if his_score > max_score:
            model.save_model(FLAGS.model_dir)
            max_score = his_score
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[-1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (np.argmax(history_score), np.max(history_score)))
                model.save_model(FLAGS.model_dir)
                break

def predict_nn(feature, gen_fea = False):
    # b = model.sess.run(model.b); w1 = model.sess.run(model.w1)
    fea_ids = []; fea_vals = []
    fea_id = [int(e.split(':')[0]) - 1 for e in feature]
    fea_val = [float(e.split(':')[1]) for e in feature]
    fea_ids.append(fea_id); fea_vals.append(fea_val)
    if gen_fea:
        fea = model.run_step(model.fea_out, fea_ids, fea_vals, mode='predict')
        return fea
    else:
        pred_prob = model.run_step(model.pred_prob, fea_ids, fea_vals, mode = 'predict')
        return pred_prob

def predictProcess(feature):
    layers = 3
    fetches = [model.FM_W, model.FM_V, model.FM_B, model.deepVars, model.attention_W, model.attention_b, model.attention_p, model.prediction_W, model.pred_prob]
    fea_ids = [];    fea_vals = []; x = [0] * FLAGS.feature_size
    fea_id = [int(e.split(':')[0]) - 1 for e in feature]
    fea_val = [float(e.split(':')[1]) for e in feature]
    fea_ids.append(fea_id);    fea_vals.append(fea_val)
    for e in feature: x[int(e.split(':')[0]) - 1] = float(e.split(':')[1])
    res = model.run_step(fetches, fea_ids, fea_vals, mode='predict')

    fmW = res[0]; fmV = res[1]; fmB = res[2]; deepVars = res[3]; attention_W = res[4]; attention_b = res[5]; attention_p = res[6]; prediction_W = res[7]; pred_prob = res[-1]

    fm_linear_terms = 0; fm_embedding = []; fm_sum_square = []; fm_square_sum = []; fm_interaction_terms = 0
    deep_inputs_nn = []
    for i in range(len(fea_id)):                    # w * x + b
        fm_linear_terms += fmW[fea_id[i]] * fea_val[i]
        fm_embedding.append(fmV[fea_id[i]])
    fm_linear_terms += fmB
    for i in range(len(fm_embedding)):              # embedding * feat_vals
        for j in range(len(fm_embedding[0])):
            fm_embedding[i][j] *= fea_val[i]

    att_fm_embedding, wgh = matmul(fm_embedding, attention_W, attention_b, attention_p)         # att_fm_embedding == fm_embedding

    for j in range(len(fm_embedding[0])):           # sum_square, sqare_sum
        sum_square_tmp = 0; sqare_sum_tmp = 0
        for i in range(len(fm_embedding)):
            sum_square_tmp += fm_embedding[i][j]
            sqare_sum_tmp += fm_embedding[i][j] * fm_embedding[i][j]
        fm_sum_square.append(sum_square_tmp * sum_square_tmp)
        fm_square_sum.append(sqare_sum_tmp)
    for i in range(len(fm_sum_square)):
        fm_interaction_terms += fm_sum_square[i] - fm_square_sum[i]
    fm_interaction_terms *= 0.5
    for i in range(len(fm_embedding)):              # deep part
        for j in range(len(fm_embedding[0])):
            deep_inputs_nn.append(fm_embedding[i][j])
    hidden_nn = deep_inputs_nn
    for i in range(layers):
        hidden_nn = wx_b(deepVars['deepW_%d' % i], hidden_nn, deepVars['deepB_%d' % i], 'relu')
    deepOut_nn = wx_b(deepVars['outW'], hidden_nn, deepVars['outB'])
    prob_out = 1 / (1 + math.exp(-(fm_linear_terms + fm_interaction_terms + deepOut_nn)))
    return prob_out

def wx_b(w, x, b, act = None):          # f(w * x + b)
    res = []
    for w_col in range(len(w[0])):
        sum = 0
        for w_row in range(len(w)):
            sum += w[w_row][w_col] * x[w_row]
        if act == 'relu':
            res.append(max(sum + b[w_col], 0))
        else:
            res.append(sum + b[w_col])
    return res

def export_features():
    train_data = read_libsvm(FLAGS.data_dir)
    feat_ids, feat_vals, label = slice_libsvm(train_data)
    fea = model.run_step(model.fea_out, feat_ids, feat_vals, label)
    with open(FLAGS.feature_dir, 'w') as f:
        for i in range (len(fea)):
            f.write(str(label[i]) + ' ')    # label
            for j in range(len(fea[i])):        # features
                f.write(str(j+1) + ':' + str(round(fea[i][j], 3)) + ' ')
            f.write('\n')
    print('export feature done, file : %s' % FLAGS.feature_dir); exit()

def matmul(w1, w2, b, p):
    res = []
    for row in range(len(w1)):          # w1 * w2
        row_res = []
        for col in range(len(w2[0])):
            sum = 0.0
            for i in range(len(w1[0])):
                sum += w1[row][i] * w2[i][col]
            row_res.append(sum)
        res.append(row_res)
    wgh = []; wgh_exp_sum = 0.0
    for i in range(len(res)):
        wgh_sum = 0.0
        for j in range(len(res[0])):
            res[i][j] += b[0][j]                # w1 * w2 + b
            res[i][j] = max(res[i][j], 0)       # relu( w1 * w2 + b)
            res[i][j] *= p[j]                   # relu( w1 * w2 + b) .* p
            wgh_sum += res[i][j]
        wgh_exp_sum += math.exp(wgh_sum)
        wgh.append(math.exp(wgh_sum))
    for i in range(len(wgh)):       # softmax( f(w1 * w2 + b) * p )
        wgh[i] /= wgh_exp_sum
    for i in range(len(w1)):
        for j in range(len(w1[0])):
            w1[i][j] *= wgh[i]
    return w1, wgh


if __name__ == "__main__":
    feature = ['2:10', '4:0.95388', '5:0.777509', '6:0', '9:1', '10:2', '11:5', '12:4', '13:4', '16:1', '19:1',
           '20:1.95', '21:3.5', '22:-1.0', '23:-1', '26:1', '27:1', '28:1', '29:0.3', '30:0.5', '31:0.8']
    #predict_nn(feature, True)    ;   exit()
    #predictProcess(feature)
    if FLAGS.train:
        print(deep_afm_params)
        train()
    else:
        export_features()
        predict_nn(feature)
