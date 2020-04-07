# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

class IDCNNConfig(object):
    """IDCNN配置参数"""
    layers = [
        {
            'dilation': 1
        },
        {
            'dilation': 1
        },
        {
            'dilation': 2
        },
    ]
    vocab_size = 5000  # 词汇表达小
    seq_length = 100  # 序列长度
    embedding_dim = 64  # 词向量维度
    learning_rate = 1e-2  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10000  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    dropout_keep_prob = 0.5  # dropout保留比例

class IDCNN(object):
    """文本分类，IDCNN模型"""
    def __init__(self, config):
        self.filter_width = 3
        self.num_filter = 10
        self.embedding_dim = config.embedding_dim
        self.initializer = initializers.xavier_initializer()
        self.repeat_times = 4
        self.layers = config.layers
        self.num_tags = config.num_classes
        self.seq_length = config.seq_length
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.idcnn()

    def idcnn(self):
        """IDCNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("idcnn"):
            # IDCNN layer
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding_inputs, self.keep_prob)
            # ldcnn layer
            model_outputs = self.IDCNN_layer(model_inputs)
            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)

            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            self.y_pred_prob = tf.nn.softmax(self.logits)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # IDCNN layer
    def IDCNN_layer(self, model_inputs, name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)
        reuse = False
        if self.keep_prob == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape = [1, self.filter_width, self.embedding_dim, self.num_filter]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim, self.num_filter],
                initializer=self.initializer)

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs, filter_weights, strides=[1, 1, 1, 1], padding="SAME", name="init_layer")
            finalOutFromLayers = []
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i, reuse=True if (reuse or j > 0) else False):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter, self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput, w, rate=dilation, padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            self.cnn_output_width = self.repeat_times * self.seq_length * self.num_filter
            finalOut = tf.reshape(finalOut, [-1, self.cnn_output_width])    ; self.aaa = finalOut
            return finalOut

    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags], dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.num_tags]))
                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)
            return tf.reshape(pred, [-1, self.num_tags])