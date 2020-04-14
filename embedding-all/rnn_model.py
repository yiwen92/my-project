#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from config import VOCAB_SIZE, EMBEDDING_DIM, SEMANTIC_DIM
from attention_mechanism import attention
import copy

class RNNConfig(object):
    """RNN配置参数"""
    # 模型参数
    rnn_num_layers= 2           # rnn 层数
    rnn_size = 3            # rnn 编码维度
    attention_size = 4         # attention 维度
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'lstm'            # lstm 或 gru
    dropout = 0.8         # dropout保留比例

# Create the rnn neural network object
class RNN(object):
    pass

def rnn_net(input_x, is_training=True, scope='RnnNet', config=RNNConfig()):
    """
    :param input_x: int32 Tensor in shape [bsz, len], the input token IDs.
    :param is_training:
    :param scope:
    :param config:
    :return:
    """
    debug_info = {}
    # Define a scope for reusing the variables
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        used = tf.sign(tf.abs(input_x))
        length = tf.reduce_sum(used, reduction_indices=1)
        lengths = tf.cast(length, tf.int32)
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, EMBEDDING_DIM])
        # TF Estimator input is a dict, in case of multiple inputs
        embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

        # 多层rnn网络
        rnn_cell = create_rnn_cell(config)

        # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量。
        # encoder_outputs用于attention，batch_size*encoder_inputs_length*rnn_size
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32, sequence_length=lengths)
        #encoder_outputs = tf.layers.dropout(encoder_outputs, config.dropout, training=is_training)
        #last = attention(encoder_outputs, lengths, config.attention_size)        # Attention mechanism
        # 取最后一个时序输出作为结果
        with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
            inputs = encoder_outputs
            if isinstance(encoder_outputs, tuple):
                print("========= tuple")
                # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
                inputs = tf.concat(encoder_outputs, 2)

            hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

            # Trainable parameters
            w_omega = tf.get_variable('w_omega', [hidden_size, config.attention_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_omega = tf.get_variable('b_omega', [config.attention_size], dtype=tf.float32)
            u_omega = tf.get_variable('u_omega', [config.attention_size], dtype=tf.float32)

            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        last = output #encoder_outputs[:, -1, :]
        debug_info['encoder_outputs'] = encoder_outputs; debug_info['last'] = last; debug_info['lengths'] = lengths
        debug_info['output'] = output; debug_info['alphas'] = alphas; debug_info['inputs'] = inputs

        fc = tf.layers.dense(last, config.hidden_dim, name='fc1')
        fc = tf.layers.dropout(fc, config.dropout, training=is_training)
        fc = tf.nn.relu(fc)

        # 分类器
        logits = tf.layers.dense(fc, SEMANTIC_DIM, name='fc2')

    return logits, debug_info

def create_rnn_cell(config):
    def single_rnn_cell():
        # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
        # 的列表中最终模型会发生错误
        if config.rnn == 'lstm':
            single_cell = tf.nn.rnn_cell.LSTMCell(config.rnn_size)
        else:
            single_cell = tf.nn.rnn_cell.GRUCell(config.rnn_size)
        #添加dropout
        cell = tf.nn.rnn_cell.DropoutWrapper(single_cell)
        return cell
    #列表中每个元素都是调用single_rnn_cell函数
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(config.rnn_num_layers)])
    return rnn_cell


