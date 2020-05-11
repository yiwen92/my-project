# -*- coding:utf-8 -*-
#!/usr/bin/env python

from utils import DTYPE, init_var_map, get_optimizer, activate, max_pool_4d
import pickle as pkl
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.layers.python.layers import initializers

# 定义基类模型
dtype = DTYPE
class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None
        self.saver = None
        self.feat_ids = None
        self.feat_vals = None
        self.initializer = initializers.xavier_initializer()

    # run model
    def run(self, fetches, X=None, y=None, mode='train'):
            # 通过feed_dict传入数据
            feed_dict = {}
            if type(self.X) is list:
                for i in range(len(X)):
                    feed_dict[self.X[i]] = X[i]
            else:
                feed_dict[self.X] = X
            if y is not None:
                feed_dict[self.y] = y
            if self.layer_keeps is not None:
                if mode == 'train':
                    feed_dict[self.layer_keeps] = self.keep_prob_train
                elif mode == 'test':
                    feed_dict[self.layer_keeps] = self.keep_prob_test
            #通过session.run去执行op
            return self.sess.run(fetches, feed_dict)

    def run_step(self, fetches, feat_ids, feat_vals, label = None, mode = 'train'):
        feed_dict = {}
        feed_dict[self.feat_ids] = feat_ids
        feed_dict[self.feat_vals] = feat_vals
        if mode == 'train':
            feed_dict[self.y] = label
        return self.sess.run(fetches, feed_dict)

    def runstep(self, fetches, features, label = None, mode = 'train'):
        feed_dict = {}
        feed_dict[self.X] = features
        if mode == 'train':
            feed_dict[self.y] = label
        return self.sess.run(fetches, feed_dict)


    # 模型参数持久化
    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)

    # 保存模型
    def save_model(self, ckptPath):
        checkpoint_path = os.path.join(ckptPath)
        self.saver.save(self.sess, checkpoint_path)
        print("model saved at", checkpoint_path)

# LR Model
class LR(Model):
    def __init__(self, field_size, feature_size, optimizer='gd', learning_rate=1e-2):
        Model.__init__(self)

        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')

        self.LR_W = tf.get_variable(name='lrW', shape=[feature_size], initializer=self.initializer, dtype=dtype)
        self.LR_B = tf.get_variable(name='lrB', shape=[1], initializer=self.initializer, dtype=dtype)

        with tf.variable_scope("LR-part"):
            feat_wgts = tf.nn.embedding_lookup(self.LR_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.LR_B    # w * x + b
        self.pred_prob = tf.sigmoid(self.linear_terms)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.linear_terms, labels=self.y))
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)
        # 保存模型的参数
        self.saver = tf.train.Saver(tf.global_variables())
        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)


# FM Model
class FM(Model):
    def __init__(self, field_size, feature_size, embedding_size, l2_reg, learning_rate, optimizer):
        Model.__init__(self)

        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')

        self.FM_W = tf.get_variable(name='fmW', shape=[feature_size], initializer=self.initializer, dtype=dtype)
        self.FM_V = tf.get_variable(name='fmV', shape=[feature_size, embedding_size], initializer=self.initializer, dtype=dtype)
        self.FM_B = tf.get_variable(name='fmB', shape=[1], initializer=self.initializer, dtype=dtype)

        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.FM_B   # FM <w, x> + b
        with tf.variable_scope("Second-order"):
            embedding = tf.nn.embedding_lookup(self.FM_V, self.feat_ids)
            feat_vals = tf.reshape(self.feat_vals, shape=[-1, field_size, 1])
            embeddings = tf.multiply(embedding, feat_vals)
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)
            self.interaction_terms = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)   # FM sum_sum<vi, vj>(xi, xj)
        self.out_sum = self.linear_terms + self.interaction_terms
        self.pred_prob = tf.sigmoid(self.out_sum)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y))
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)
        self.saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

# FFM Model
class FFM(Model):
    def __init__(self, field2feature, field2fields, field_size, feature_size, embedding_size, l2_reg, learning_rate, optimizer):
        Model.__init__(self)

        field2fields_size = len(set(field2fields))
        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')

        self.FM_W = tf.get_variable(name='fmW', shape=[feature_size], initializer=self.initializer, dtype=dtype)
        self.FM_V = tf.get_variable(name='fmV', shape=[field_size, feature_size, field2fields_size, embedding_size], initializer=self.initializer, dtype=dtype)
        self.FM_B = tf.get_variable(name='fmB', shape=[1], initializer=self.initializer, dtype=dtype)

        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.FM_B   # FM <w, x> + b
        with tf.variable_scope('field_aware_interaction_layer'):
            self.field_aware_interaction_terms = tf.constant(0, dtype=dtype)
            for i in range(field_size):
                for j in range(i+1, field_size):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(self.FM_V[i, field2feature[i], field2fields[j]], self.FM_V[j, field2feature[j], field2fields[i]])),     # <w_xifj, w_xjfi>
                        tf.multiply(self.feat_vals[:,i], self.feat_vals[:,j])           # x_i * x_j
                    )
        self.out_sum = self.linear_terms + self.field_aware_interaction_terms
        self.pred_prob = tf.sigmoid(self.out_sum)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y))
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)
        self.saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

# deep ffm Model
class DeepFFM(Model):
    def __init__(self, field2feature, field2fields, field_size, feature_size, embedding_size, l2_reg, learning_rate,
                 layer_sizes, optimizer, layer_acts, drop_out, train = True):
        Model.__init__(self)

        field2fields_size = len(set(field2fields))
        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')

        self.FM_W = tf.get_variable(name='fmW', shape=[feature_size], initializer=self.initializer, dtype=dtype)
        self.FM_V = tf.get_variable(name='fmV', shape=[field_size, feature_size, field2fields_size, embedding_size], initializer=self.initializer, dtype=dtype)
        self.FM_B = tf.get_variable(name='fmB', shape=[1], initializer=self.initializer, dtype=dtype)
        self.embedding = tf.get_variable(name='featureEmbedding', shape=[feature_size, embedding_size], initializer=self.initializer, dtype=dtype)
        init_deep_layer_vars = []
        input_dim = field_size * embedding_size
        for i in range(len(layers)):
            output_dim = layers[i]
            init_deep_layer_vars.append(('deepW_%d' % i, [input_dim, output_dim], 'xavier', dtype))
            init_deep_layer_vars.append(('deepB_%d' % i, [output_dim], 'zero', dtype))
            input_dim = layers[i]
        init_deep_layer_vars.append(('outW', [layers[-1], 1], 'xavier', dtype))
        init_deep_layer_vars.append(('outB', [1], 'zero', dtype))
        self.deepVars = init_var_map(init_deep_layer_vars)

        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.FM_B   # FM <w, x> + b
        with tf.variable_scope('field_aware_interaction_layer'):
            self.field_aware_interaction_terms = tf.constant(0, dtype=dtype)
            for i in range(field_size):
                for j in range(i+1, field_size):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(self.FM_V[i, field2feature[i], field2fields[j]], self.FM_V[j, field2feature[j], field2fields[i]])),     # <w_xifj, w_xjfi>
                        tf.multiply(self.feat_vals[:,i], self.feat_vals[:,j])           # x_i * x_j
                    )
        with tf.variable_scope("Deep-part"):
            embedding = tf.nn.embedding_lookup(self.embedding, self.feat_ids)
            deep_inputs = tf.reshape(embedding, shape=[-1, field_size * embedding_size])
            hidden = deep_inputs
            for i in range(len(layers)):
                if train:
                    hidden = tf.nn.dropout(  # h_i = W_i * x + b_i
                        activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i]),
                        dropout[i])
                else:
                    hidden = activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i])
            self.hidden = hidden
            self.deepOut = tf.matmul(hidden, self.deepVars['outW']) + self.deepVars['outB']
        self.deepOut = tf.reshape(self.deepOut, shape=[-1])
        with tf.variable_scope("DeepFM-out"):
            self.out_sum = self.linear_terms + self.field_aware_interaction_terms + self.deepOut
            self.pred_prob = tf.sigmoid(self.out_sum)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y))
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)
        self.saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

# deep fm Model
class DeepFM(Model):
    def __init__(self, field_size, feature_size, embedding_size, l2_reg, learning_rate, optimizer, layer_sizes, layer_acts, drop_out, train = True):
        # field_size : field number  # feature_size : feature number
        Model.__init__(self)

        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')
        self.keep_prob = tf.placeholder(dtype=dtype)

        self.FM_W = tf.get_variable(name='fmW', shape=[feature_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_V = tf.get_variable(name='fmV', shape=[feature_size, embedding_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_B = tf.get_variable(name='fmB', shape=[1], initializer=self.initializer, dtype=dtype)  # tf.constant_initializer(0.0), dtype=dtype)
        init_deep_layer_vars = []
        input_dim = field_size * embedding_size
        for i in range(len(layers)):
            output_dim = layers[i]
            init_deep_layer_vars.append(('deepW_%d' % i, [input_dim, output_dim], 'xavier', dtype))
            init_deep_layer_vars.append(('deepB_%d' % i, [output_dim], 'zero', dtype))
            input_dim = layers[i]
        init_deep_layer_vars.append(('outW', [layers[-1], 1], 'xavier', dtype))
        init_deep_layer_vars.append(('outB', [1], 'zero', dtype))
        self.deepVars = init_var_map(init_deep_layer_vars)

        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.FM_B   # FM <w, x>
        with tf.variable_scope("Second-order"):
            embedding = tf.nn.embedding_lookup(self.FM_V, self.feat_ids)
            feat_vals = tf.reshape(self.feat_vals, shape=[-1, field_size, 1])
            embeddings = tf.multiply(embedding, feat_vals)
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)
            self.interaction_terms = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)   # FM sum_sum<vi, vj>(xi, xj)
        with tf.variable_scope("Deep-part"):
            deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])   ;self.deep_inputs = deep_inputs
            hidden = deep_inputs
            for i in range(len(layers)):
                if train:
                    hidden = tf.nn.dropout(  # h_i = W_i * x + b_i
                        activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i]),
                        dropout[i])
                else:
                    hidden = activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i])
            self.hidden = hidden
            self.deepOut = tf.matmul(hidden, self.deepVars['outW']) + self.deepVars['outB']
        self.deepOut = tf.reshape(self.deepOut, shape=[-1])  # tf.reshape() 防止 python 自带的广播机制算出奇怪的值
        with tf.variable_scope("DeepFM-out"):
            self.out_sum = self.linear_terms + self.interaction_terms + self.deepOut
            self.pred_prob = tf.sigmoid(self.out_sum)

        self.auc = tf.metrics.auc(self.y, self.pred_prob)
        # ------bulid loss------
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y))
        #self.loss = tf.reduce_mean(tf.square(self.y - self.pred_prob))

        # ------bulid optimizer------
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)

        # 保存模型的参数
        self.saver = tf.train.Saver(tf.global_variables())

        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 图中所有variable初始化
        #tf.global_variables_initializer().run(session=self.sess)

# deep afm Model
class DeepAFM(Model):
    def __init__(self, field_size, feature_size, embedding_size, l2_reg, learning_rate, optimizer, layer_sizes, layer_acts, drop_out, train = True):
        # field_size : field number  # feature_size : feature number
        Model.__init__(self)

        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')
        self.keep_prob = tf.placeholder(dtype=dtype)

        self.FM_W = tf.get_variable(name='fmW', shape=[feature_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_V = tf.get_variable(name='fmV', shape=[feature_size, embedding_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_B = tf.get_variable(name='fmB', shape=[1], initializer=self.initializer, dtype=dtype)  # tf.constant_initializer(0.0), dtype=dtype)
        init_deep_layer_vars = []
        input_dim = field_size * embedding_size
        for i in range(len(layers)):
            output_dim = layers[i]
            init_deep_layer_vars.append(('deepW_%d' % i, [input_dim, output_dim], 'xavier', dtype))
            init_deep_layer_vars.append(('deepB_%d' % i, [output_dim], 'zero', dtype))
            input_dim = layers[i]
        init_deep_layer_vars.append(('outW', [layers[-1], 1], 'xavier', dtype))
        init_deep_layer_vars.append(('outB', [1], 'zero', dtype))
        self.deepVars = init_var_map(init_deep_layer_vars)

        self.attention_W = tf.get_variable(name='atention_W', shape=[embedding_size, embedding_size], initializer=self.initializer, dtype=dtype)
        self.attention_b = tf.get_variable(name='atention_b', shape=[1, embedding_size], initializer=self.initializer, dtype=dtype)
        self.attention_p = tf.get_variable(name='atention_p', shape=[embedding_size], initializer=self.initializer, dtype=dtype)
        self.prediction_W = tf.get_variable(name='prediction_W', shape=[embedding_size, 1], initializer=self.initializer, dtype=dtype)

        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.FM_B   # FM <w, x>
        with tf.variable_scope("Second-order"):
            embedding = tf.nn.embedding_lookup(self.FM_V, self.feat_ids)
            feat_vals = tf.reshape(self.feat_vals, shape=[-1, field_size, 1])
            embeddings = tf.multiply(embedding, feat_vals)
            #sum_square = tf.square(tf.reduce_sum(embeddings, 1))
            #square_sum = tf.reduce_sum(tf.square(embeddings), 1)
            #self.interaction_terms = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)   # FM sum_sum<vi, vj>(xi, xj)
        with tf.variable_scope("Attention-part"):
            self.attention_embed = tf.reshape(embeddings, shape=[-1, embedding_size])
            self.attention_embed_wx = tf.matmul(self.attention_embed, self.attention_W)
            self.attention_mul = tf.reshape(self.attention_embed_wx, shape=[-1, field_size, embedding_size])
            self.wx_b = tf.nn.relu(self.attention_mul + self.attention_b)
            self.atten_project = tf.multiply(self.attention_p, self.wx_b)
            self.attention_wgh = tf.reshape(tf.reduce_sum(self.atten_project, 2, keep_dims=True), shape=[-1, field_size])
            self.attention_out = tf.reshape(tf.nn.softmax(self.attention_wgh), shape=[-1, field_size, 1])
            embeddings = tf.multiply(embeddings, self.attention_out)      # embeddings * weight_attention
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))    ;self.embeddings = embeddings
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        self.interaction_terms = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)
        with tf.variable_scope("Deep-part"):
            deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])
            hidden = deep_inputs
            for i in range(len(layers)):    # h_i = W_i * x + b_i
                hidden = activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i])
                if train:
                    hidden = tf.nn.dropout(hidden, dropout[i])      # drop out
            self.fea_out = hidden
            self.deepOut = tf.matmul(hidden, self.deepVars['outW']) + self.deepVars['outB']
        self.deepOut = tf.reshape(self.deepOut, shape=[-1])  # tf.reshape() 防止 python 自带的广播机制算出奇怪的值
        with tf.variable_scope("DeepFM-out"):
            self.out_sum = self.linear_terms + self.interaction_terms + self.deepOut
            self.pred_prob = tf.sigmoid(self.out_sum)

        self.auc = tf.metrics.auc(self.y, self.pred_prob)
        # ------bulid loss------
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        self.regularization = tf.constant(0, dtype=dtype)
        for e in self.deepVars:
            self.regularization += regularizer(self.deepVars[e])
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y)) + self.regularization
        #self.loss = tf.reduce_mean(tf.square(self.y - self.pred_prob))

        # ------bulid optimizer------
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)

        # 保存模型的参数
        self.saver = tf.train.Saver(tf.global_variables())

        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 图中所有variable初始化
        #tf.global_variables_initializer().run(session=self.sess)

class NFM(Model):
    def __init__(self, field_size, feature_size, embedding_size, l2_reg, learning_rate, optimizer, layer_sizes, layer_acts, drop_out, train = True):
        # field_size : field number  # feature_size : feature number
        Model.__init__(self)

        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')
        self.keep_prob = tf.placeholder(dtype=dtype)

        self.FM_W = tf.get_variable(name='fmW', shape=[feature_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_V = tf.get_variable(name='fmV', shape=[feature_size, embedding_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_B = tf.get_variable(name='fmB', shape=[1], initializer=self.initializer, dtype=dtype)  # tf.constant_initializer(0.0), dtype=dtype)
        init_deep_layer_vars = []
        #input_dim = field_size * embedding_size        # deep_fm
        input_dim = embedding_size          # NFM
        for i in range(len(layers)):
            output_dim = layers[i]
            init_deep_layer_vars.append(('deepW_%d' % i, [input_dim, output_dim], 'xavier', dtype))
            init_deep_layer_vars.append(('deepB_%d' % i, [output_dim], 'zero', dtype))
            input_dim = layers[i]
        init_deep_layer_vars.append(('outW', [layers[-1], 1], 'xavier', dtype))
        init_deep_layer_vars.append(('outB', [1], 'zero', dtype))
        self.deepVars = init_var_map(init_deep_layer_vars)

        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.FM_B   # FM <w, x>
        with tf.variable_scope("Second-order"):
            embedding = tf.nn.embedding_lookup(self.FM_V, self.feat_ids)
            feat_vals = tf.reshape(self.feat_vals, shape=[-1, field_size, 1])
            embeddings = tf.multiply(embedding, feat_vals)
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)
            self.interaction_terms = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)   # FM sum_sum<vi, vj>(xi, xj)
        with tf.variable_scope("Deep-part"):
            #deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])   ;self.deep_inputs = deep_inputs    # deep_fm
            deep_inputs = 0.5 * tf.subtract(sum_square, square_sum)         # NFM
            hidden = deep_inputs
            for i in range(len(layers)):
                if train:
                    hidden = tf.nn.dropout(  # h_i = W_i * x + b_i
                        activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i]),
                        dropout[i])
                else:
                    hidden = activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i])
            self.hidden = hidden
            self.deepOut = tf.matmul(hidden, self.deepVars['outW']) + self.deepVars['outB']
        self.deepOut = tf.reshape(self.deepOut, shape=[-1])  # tf.reshape() 防止 python 自带的广播机制算出奇怪的值
        with tf.variable_scope("DeepFM-out"):
            self.out_sum = self.linear_terms + self.interaction_terms + self.deepOut
            self.pred_prob = tf.sigmoid(self.out_sum)

        self.auc = tf.metrics.auc(self.y, self.pred_prob)
        # ------bulid loss------
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y))
        #self.loss = tf.reduce_mean(tf.square(self.y - self.pred_prob))

        # ------bulid optimizer------
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)

        # 保存模型的参数
        self.saver = tf.train.Saver(tf.global_variables())

        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 图中所有variable初始化
        #tf.global_variables_initializer().run(session=self.sess)

class AFM(Model):
    def __init__(self, field_size, feature_size, embedding_size, l2_reg, learning_rate, optimizer, layer_sizes, layer_acts, drop_out, train = True):
        # field_size : field number  # feature_size : feature number
        Model.__init__(self)

        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        self.feat_ids = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='input_id')
        self.feat_vals = tf.placeholder(dtype=dtype, shape=[None, field_size], name='input_value')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')
        self.keep_prob = tf.placeholder(dtype=dtype)

        self.FM_W = tf.get_variable(name='fmW', shape=[feature_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_V = tf.get_variable(name='fmV', shape=[feature_size, embedding_size], initializer=self.initializer, dtype=dtype)#tf.glorot_normal_initializer(), dtype=dtype)
        self.FM_B = tf.get_variable(name='fmB', shape=[1], initializer=self.initializer, dtype=dtype)  # tf.constant_initializer(0.0), dtype=dtype)
        init_deep_layer_vars = []
        #input_dim = field_size * embedding_size        # deep_fm
        input_dim = embedding_size          # NFM
        for i in range(len(layers)):
            output_dim = layers[i]
            init_deep_layer_vars.append(('deepW_%d' % i, [input_dim, output_dim], 'xavier', dtype))
            init_deep_layer_vars.append(('deepB_%d' % i, [output_dim], 'zero', dtype))
            input_dim = layers[i]
        init_deep_layer_vars.append(('outW', [layers[-1], 1], 'xavier', dtype))
        init_deep_layer_vars.append(('outB', [1], 'zero', dtype))
        self.deepVars = init_var_map(init_deep_layer_vars)

        self.attention_W = tf.get_variable(name='atention_W', shape=[embedding_size, embedding_size], initializer=self.initializer, dtype=dtype)
        self.attention_b = tf.get_variable(name='atention_b', shape=[1, embedding_size], initializer=self.initializer, dtype=dtype)
        self.attention_p = tf.get_variable(name='atention_p', shape=[embedding_size], initializer=self.initializer, dtype=dtype)
        self.prediction_W = tf.get_variable(name='prediction_W', shape=[embedding_size, 1], initializer=self.initializer, dtype=dtype)

        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.FM_W, self.feat_ids)
            self.linear_terms = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals), 1) + self.FM_B   # FM <w, x>
        with tf.variable_scope("Attention-net"):
            embedding = tf.nn.embedding_lookup(self.FM_V, self.feat_ids)        # batch_size * field_size * embedding_size
            feat_vals = tf.reshape(self.feat_vals, shape=[-1, field_size, 1])   # batch_size * field_size * 1
            embeddings = tf.multiply(embedding, feat_vals)                      # batch_size * field_size * embedding_size
            element_wise_product_list = []
            for i in range(field_size):
                for j in range(i+1, field_size):
                    element_wise_product_list.append(tf.multiply(embeddings[:,i,:], embeddings[:,j,:]))     # <v_i, v_j> * x_i * x_j
            self.element_wise_product_tmp = tf.stack(element_wise_product_list)  # (field_size * (field_size-1) / 2) * batch_size * embedding_size
            self.element_wise_product = tf.transpose(self.element_wise_product_tmp, perm=[1, 0, 2],)  # batch_size * (field_size * (field_size-1) / 2) * embedding_size
            self.interactions = tf.reduce_sum(self.element_wise_product, 2)

        # _________ MLP Layer / attention part _____________
        num_interactions = int(field_size * (field_size - 1) / 2)

        self.element_wise_product_embed = tf.reshape(self.element_wise_product, shape=[-1, embedding_size]) # [batch_size * (field_size * (field_size-1) / 2)] * embedding_size
        self.element_wise_product_embed_wx = tf.matmul(self.element_wise_product_embed, self.attention_W)
        self.attention_mul = tf.reshape(self.element_wise_product_embed_wx , shape=[-1, num_interactions, embedding_size])
        self.wx_b = tf.nn.relu(self.attention_mul + self.attention_b)
        self.atten_project = tf.multiply(self.attention_p, self.wx_b)
        self.attention_relu = tf.reduce_sum(self.atten_project, 2, keep_dims=True)      # batch_size * (field_size * (field_size-1) / 2) * 1

        self.attention_out = tf.nn.softmax(self.attention_relu)
        #self.attention_out = tf.nn.dropout(self.attention_out, 0.5)
        attention_weight = tf.multiply(self.attention_out, self.element_wise_product)
        self.AFM = tf.reduce_sum(attention_weight, 1, name="afm")  # batch_size * embedding_size
        self.prediction = tf.matmul(self.AFM, self.prediction_W)  # batch_size * 1
        self.prediction = tf.reshape(self.prediction, shape=[-1])  # tf.reshape() 防止 python 自带的广播机制算出奇怪的值

        self.out_sum = self.linear_terms + self.prediction
        self.pred_prob = tf.sigmoid(self.out_sum)

        #self.auc = tf.metrics.auc(self.y, self.pred_prob)
        # ------bulid loss------
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y))
        #self.loss = tf.reduce_mean(tf.square(self.y - self.pred_prob))

        # ------bulid optimizer------
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)

        # 保存模型的参数
        self.saver = tf.train.Saver(tf.global_variables())

        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 图中所有variable初始化
        #tf.global_variables_initializer().run(session=self.sess)

# FNN Model
class FNN(Model):
    def __init__(self, fields, feature_size, embedding_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 optimizer='gd', learning_rate=1e-2, train=True):
        Model.__init__(self)

        init_vars = []
        num_field = len(fields)
        field_offset = []
        for i in range(num_field):
            field_offset.append(sum(fields[:i+1]))

        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        for i in range(num_field):
            init_vars.append(('embed_%d' % i, [fields[i], embedding_size], 'xavier', dtype))
        node_in = num_field * embedding_size
        for i in range(len(layers)):
            init_vars.append(('w%d' % i, [node_in, layers[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layers[i]], 'zero', dtype))
            node_in = layers[i]
        init_vars.append(('outW', [layers[-1], 1], 'xavier', dtype))
        init_vars.append(('outB', [1], 'zero', dtype))
        self.X = tf.placeholder(dtype=dtype, shape=[None, feature_size], name='input')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')
        self.vars = init_var_map(init_vars)

        with tf.variable_scope("FM-embedding"):
            w0 = [self.vars['embed_%d' % i] for i in range(num_field)]
            concat_hidden = []
            start = 0
            for i in range(num_field):
                concat_hidden.append(tf.matmul(self.X[:,start:field_offset[i]], w0[i]))
                start = field_offset[i]
            xw = tf.concat(concat_hidden, 1)

        hidden = xw
        with tf.variable_scope("MLP"):
            for i in range(len(layers)):
                if train:
                    hidden = tf.nn.dropout(
                        activate(tf.matmul(hidden, self.vars['w%d' % i]) + self.vars['b%d' % i], layer_active_func[i]),
                        dropout[i])
                else:
                    hidden = activate(tf.matmul(hidden, self.vars['w%d' % i]) + self.vars['b%d' % i], layer_active_func[i])
        self.deepOut = tf.matmul(hidden, self.vars['outW']) + self.vars['outB']
        self.deepOut = tf.reshape(self.deepOut, shape=[-1])  # tf.reshape() 防止 python 自带的广播机制算出奇怪的值
        self.pred_prob = tf.sigmoid(self.deepOut)

        # ------bulid loss------
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.deepOut, labels=self.y))

        # ------bulid optimizer------
        self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)

        # 保存模型的参数
        self.saver = tf.train.Saver(tf.global_variables())

        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 图中所有variable初始化
        # tf.global_variables_initializer().run(session=self.sess)

# CCPM Model
class CCPM(Model):
    def __init__(self, field_sizes=None, embed_size=10, filter_sizes=None, layer_acts=None, drop_out=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        init_vars.append(('f1', [embed_size, filter_sizes[0], 1, 2], 'xavier', dtype))
        init_vars.append(('f2', [embed_size, filter_sizes[1], 2, 2], 'xavier', dtype))
        init_vars.append(('w1', [2 * 3 * embed_size, 1], 'xavier', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embed_size, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    int(num_inputs / 2)),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                activate(
                    tf.reshape(l, [-1, embed_size * 3 * 2]),
                    layer_acts[0]),
                self.layer_keeps[0])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.matmul(l, w1) + b1

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

# PNN1 Model
class PNN1(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        node_in = num_inputs * embed_size + num_pairs
        # node_in = num_inputs * (embed_size + num_inputs)
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])

            row = []
            col = []
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    row.append(i)
                    col.append(j)
            # batch * pair * k
            p = tf.transpose(
                # pair * batch * k
                tf.gather(
                    # num * batch * k
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    row),
                [1, 0, 2])
            # batch * pair * k
            q = tf.transpose(
                tf.gather(
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    col),
                [1, 0, 2])
            p = tf.reshape(p, [-1, num_pairs, embed_size])
            q = tf.reshape(q, [-1, num_pairs, embed_size])
            ip = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])

            # simple but redundant
            # batch * n * 1 * k, batch * 1 * n * k
            # ip = tf.reshape(
            #     tf.reduce_sum(
            #         tf.expand_dims(xw3d, 2) *
            #         tf.expand_dims(xw3d, 1),
            #         3),
            #     [-1, num_inputs**2])
            l = tf.concat([xw, ip], 1)

            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
                self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                tf.global_variables_initializer().run(session=self.sess)

# PNN2 Model
class PNN2(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None,
                 layer_norm=True):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        node_in = num_inputs * embed_size + num_pairs
        init_vars.append(('kernel', [embed_size, num_pairs, embed_size], 'xavier', dtype))
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero',  dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])

            row = []
            col = []
            for i in range(num_inputs - 1):
                for j in range(i + 1, num_inputs):
                    row.append(i)
                    col.append(j)
            # batch * pair * k
            p = tf.transpose(
                # pair * batch * k
                tf.gather(
                    # num * batch * k
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    row),
                [1, 0, 2])
            # batch * pair * k
            q = tf.transpose(
                tf.gather(
                    tf.transpose(
                        xw3d, [1, 0, 2]),
                    col),
                [1, 0, 2])
            # b * p * k
            p = tf.reshape(p, [-1, num_pairs, embed_size])
            # b * p * k
            q = tf.reshape(q, [-1, num_pairs, embed_size])
            # k * p * k
            k = self.vars['kernel']

            # batch * 1 * pair * k
            p = tf.expand_dims(p, 1)
            # batch * pair
            kp = tf.reduce_sum(
                # batch * pair * k
                tf.multiply(
                    # batch * pair * k
                    tf.transpose(
                        # batch * k * pair
                        tf.reduce_sum(
                            # batch * k * pair * k
                            tf.multiply(
                                p, k),
                            -1),
                        [0, 2, 1]),
                    q),
                -1)

            #
            # if layer_norm:
            #     # x_mean, x_var = tf.nn.moments(xw, [1], keep_dims=True)
            #     # xw = (xw - x_mean) / tf.sqrt(x_var)
            #     # x_g = tf.Variable(tf.ones([num_inputs * embed_size]), name='x_g')
            #     # x_b = tf.Variable(tf.zeros([num_inputs * embed_size]), name='x_b')
            #     # x_g = tf.Print(x_g, [x_g[:10], x_b])
            #     # xw = xw * x_g + x_b
            #     p_mean, p_var = tf.nn.moments(op, [1], keep_dims=True)
            #     op = (op - p_mean) / tf.sqrt(p_var)
            #     p_g = tf.Variable(tf.ones([embed_size**2]), name='p_g')
            #     p_b = tf.Variable(tf.zeros([embed_size**2]), name='p_b')
            #     # p_g = tf.Print(p_g, [p_g[:10], p_b])
            #     op = op * p_g + p_b

            l = tf.concat([xw, kp], 1)
            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)#tf.concat(w0, 0))
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)