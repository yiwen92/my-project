#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from config import VOCAB_SIZE, EMBEDDING_DIM, SEMANTIC_DIM

class TRNNConfig(object):
    """RNN配置参数"""
    # 模型参数
    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'lstm'            # lstm 或 gru
    keep_prob = 0.8         # dropout保留比例

config=TRNNConfig()

# Create the rnn neural network
def rnn_net(input_x, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('RnnNet', reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, EMBEDDING_DIM])
        # TF Estimator input is a dict, in case of multiple inputs
        embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

        # 多层rnn网络
        cells = [dropout() for _ in range(config.num_layers)]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
        last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        fc = tf.layers.dense(last, config.hidden_dim, name='fc1')
        fc = tf.contrib.layers.dropout(fc, config.keep_prob)
        fc = tf.nn.relu(fc)

        # 分类器
        logits = tf.layers.dense(fc, SEMANTIC_DIM, name='fc2')

    return logits

def lstm_cell():  # lstm核
    return tf.contrib.rnn.BasicLSTMCell(config.hidden_dim, state_is_tuple=True)

def gru_cell():  # gru核
    return tf.contrib.rnn.GRUCell(config.hidden_dim)

def dropout():  # 为每一个rnn核后面加一个dropout层
    if (config.rnn == 'lstm'):
        cell = lstm_cell()
    else:
        cell = gru_cell()
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)