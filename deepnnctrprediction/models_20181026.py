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
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2, l2_weight=0,
                 random_seed=None):
        Model.__init__(self)
        # 声明参数
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            # 用稀疏的placeholder
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            # init参数
            self.vars = init_var_map(init_vars, init_path)

            w = self.vars['w']
            b = self.vars['b']
            # sigmoid(wx+b)
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)
            # GPU设定
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # 初始化图里的参数
            tf.global_variables_initializer().run(session=self.sess)

# FM Model
class FM(Model):
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        Model.__init__(self)
        # 一次、二次交叉、偏置项
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('v', [input_dim, factor_order], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']

            # [(x1+x2+x3)^2 - (x1^2+x2^2+x3^2)]/2
            # 先计算所有的交叉项，再减去平方项(自己和自己相乘)
            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v))
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 1),
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b + p, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            #GPU设定
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # 图中所有variable初始化
            tf.global_variables_initializer().run(session=self.sess)

# FFM Model
class FFM(Model):
    def __init__(self, feature2field = None, learning_rate=1e-2, n = None, f = None, k = 40, opt_algo = 'gd'):
        # n : feature number  # f : field number  # k : latent vector dimension
        Model.__init__(self)

        self.X = tf.placeholder('float64', [None, n])
        self.y = tf.placeholder('float64', [None, ])
        self.keep_prob = tf.placeholder('float64')

        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer(), dtype='float64')
            w1 = tf.get_variable('w1', shape=[n, 1], initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2), dtype='float64')
            # shape of [None, 1]
            self.linear_terms = tf.reshape(tf.add(tf.matmul(self.X, w1), b), [-1])
        with tf.variable_scope('field_aware_interaction_layer'):
            v = tf.get_variable('v', shape=[n, f, k], dtype='float64', initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.field_aware_interaction_terms = tf.constant(0, dtype='float64')
            # build dict to find f, key of feature,value of field
            for i in range(n):
                for j in range(i+1,n):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(v[i,feature2field[j]], v[j,feature2field[i]])),
                        tf.multiply(self.X[:,i], self.X[:,j])
                    )
        # shape of [None, 1]
        #self.y_out = tf.reshape(tf.add(self.linear_terms, self.field_aware_interaction_terms), [-1])
        #self.field_aware_interaction_terms = tf.reshape(self.field_aware_interaction_terms, [-1])
        logits = tf.reshape(self.linear_terms + self.field_aware_interaction_terms , [-1])
        #logits = self.field_aware_interaction_terms#tf.add(self.linear_terms, self.field_aware_interaction_terms)
        self.y_prob = tf.sigmoid(logits)
        #self.y_out = tf.add(self.linear_terms, self.field_aware_interaction_terms)

        #self.y_out_prob = tf.nn.softmax(self.y_out)

        # loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits);#tf.nn.sigmoid_cross_entropy_with_logits
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)
        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 图中所有variable初始化
        tf.global_variables_initializer().run(session=self.sess)

# deep ffm Model
class DeepFFM(Model):
    def __init__(self, feature2field , layer_sizes, layer_acts, drop_out, learning_rate=1e-2, l2_reg = 0,
                 feature_size = None, field_size = None, k = 40, opt_algo = 'adam', train = True):
        # feature_size : feature number  # field_size : field number  # k : latent vector dimension
        Model.__init__(self)

        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        self.X = tf.placeholder(dtype=dtype, shape=[None, feature_size], name='input')
        self.y = tf.placeholder(dtype=dtype, shape=[None, ], name='label')
        self.keep_prob = tf.placeholder(dtype=dtype)

        with tf.variable_scope('linear_layer'):     # w .* x + b
            self.b = tf.get_variable(name='bias', initializer=tf.constant(0.5), dtype=dtype)    # tf.zeros_initializer()
            self.w1 = tf.get_variable(name='w1', shape=[feature_size], initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2), dtype=dtype)
            self.linear_terms = tf.reduce_sum(tf.multiply(self.w1, self.X), 1) + self.b
        with tf.variable_scope('field_aware_interaction_layer'):    # sum(<vi_fj, vj_fi> * x_i * x_j)
            self.nfk = tf.get_variable('nfk', shape=[feature_size, field_size, k], dtype=dtype, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            self.field_aware_interaction_terms = tf.constant(0, dtype=dtype)
            for i in range(feature_size):
                for j in range(i + 1, feature_size):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(self.nfk[i, feature2field[j]], self.nfk[j, feature2field[i]])),
                        tf.multiply(self.X[:, i], self.X[:, j])
                    )
        init_deep_layer_vars = []
        input_dim = feature_size
        for i in range(len(layers)):
            output_dim = layers[i]
            init_deep_layer_vars.append(('deepW_%d' % i, [input_dim, output_dim], 'xavier', dtype))
            init_deep_layer_vars.append(('deepB_%d' % i, [output_dim], 'zero', dtype))
            input_dim = layers[i]
        init_deep_layer_vars.append(('outW', [layers[-1], 1], 'xavier', dtype))
        init_deep_layer_vars.append(('outB', [1], 'zero', dtype))
        self.deepVars = init_var_map(init_deep_layer_vars)
        with tf.variable_scope("Deep-part"):
            hidden = self.X
            for i in range(len(layers)):
                if train:
                    hidden = tf.nn.dropout(                 # h_i = W_i * x + b_i
                        activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i]),
                        dropout[i])
                else:
                    hidden = activate(tf.matmul(hidden, self.deepVars['deepW_%d' % i]) + self.deepVars['deepB_%d' % i], layer_active_func[i])
            #self.a=hidden;self.aa=self.deepVars['outW'];self.aaa=tf.matmul(hidden, self.deepVars['outW']);self.aaaa=self.deepVars['outB']
            self.deepOut = tf.matmul(hidden, self.deepVars['outW']) + self.deepVars['outB']
        self.deepOut = tf.reshape(self.deepOut, shape=[-1])             # tf.reshape() 防止 python 自带的广播机制算出奇怪的值
        with tf.variable_scope("DeepFM-out"):
            self.out_sum = self.linear_terms + self.field_aware_interaction_terms + self.deepOut
            self.pred_prob = tf.sigmoid(self.out_sum)

        # ------bulid loss------
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_sum, labels=self.y)) + \
                        l2_reg * tf.nn.l2_loss(self.w1) + \
                        l2_reg * tf.nn.l2_loss(self.nfk)

        # ------bulid optimizer------
        self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

        # 保存模型的参数
        self.saver = tf.train.Saver(tf.global_variables())

        # GPU设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 图中所有variable初始化
        #tf.global_variables_initializer().run(session=self.sess)

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


# FNN Model
class FNN(Model):
    def __init__(self, fields, feature_size, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)

        init_vars = []
        num_field = len(set(fields))
        field_size = [0] * num_field
        for e in fields:
            field_size[e] += 1
        field_offset = [field_size[0]]
        for i in range(2, len(field_size) + 1):
            field_offset.append(sum(field_size[:i]))

        layers = [int(e) for e in layer_sizes.split(',')]
        dropout = [float(e) for e in drop_out.split(',')]
        layer_active_func = [e for e in layer_acts.split(',')]

        for i in range(num_field):
            init_vars.append(('embed_%d' % i, [field_size[i], embed_size], 'xavier', dtype))
        node_in = num_field * embed_size
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
                hidden = tf.nn.dropout(
                    activate(tf.matmul(hidden, self.vars['w%d' % i]) + self.vars['b%d' % i], layer_active_func[i]),
                    dropout[i])
        self.deepOut = tf.matmul(hidden, self.vars['outW']) + self.vars['outB']
        self.deepOut = tf.reshape(self.deepOut, shape=[-1])  # tf.reshape() 防止 python 自带的广播机制算出奇怪的值
        self.pred_prob = tf.sigmoid(self.deepOut)

        # ------bulid loss------
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.deepOut, labels=self.y))

        # ------bulid optimizer------
        self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

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