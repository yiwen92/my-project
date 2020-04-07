#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle, heapq, json
import tensorflow as tf
import tensorflow.contrib.keras as kr
from cnn_model import TCNNConfig, TextCNN
from rnn_model import TRNNConfig, TextRNN
from data.loader import native_content

model_type = 'cnn'

if model_type == 'cnn':
    from run_cnn import map_path, save_path
else:
    from run_rnn import map_path, save_path

class Predictor():
    def __init__(self):
        with open(map_path, "rb") as f:
            self.word_to_id, self.cat_to_id, self.seq_length, self.num_classes = pickle.load(f)
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        if model_type == 'cnn':
            self.config = TCNNConfig()
            self.config.num_classes = self.num_classes
            self.config.vocab_size = len(self.word_to_id)
            self.model = TextCNN(self.config)
        else:
            self.config = TRNNConfig()
            self.config.num_classes = self.num_classes
            self.config.vocab_size = len(self.word_to_id)
            self.model = TextRNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
    def predict(self, txt, k = 1):
        data_id = [[self.word_to_id[x] for x in list(native_content(txt)) if x in self.word_to_id]]
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.seq_length)
        y_pred_prob = self.session.run(self.model.y_pred_prob, feed_dict={self.model.input_x: x_pad, self.model.keep_prob: 1.0})
        y_pred_prob_list = y_pred_prob[0].tolist()
        k_index = list(map(y_pred_prob_list.index, heapq.nlargest(k, y_pred_prob_list)))
        k_value = heapq.nlargest(k, y_pred_prob_list)
        res = []
        for i in range(len(k_index)):
            res.append({self.id_to_cat[k_index[i]]: round(k_value[i], 3)})
        return json.dumps({'sentence': txt, 'predict result': res}, ensure_ascii=False)

if __name__ == '__main__':
    p = Predictor()
    res = p.predict('怎么找工作', 3)
    print(res)
    a=1