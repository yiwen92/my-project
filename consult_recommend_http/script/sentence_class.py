#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle, heapq, json, sys, logging
import tensorflow as tf
import tensorflow.contrib.keras as kr
from cnn_model import TCNNConfig, TextCNN
reload(sys)
sys.setdefaultencoding('utf8')

class Predictor():
    def __init__(self):
        self.map_path = './model/ids.map'
        self.save_path = './model/best_validation'
        with open(self.map_path, "rb") as f:
            self.word_to_id, self.cat_to_id, self.seq_length, self.num_classes = pickle.load(f)
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        self.config = TCNNConfig()
        self.config.num_classes = self.num_classes
        self.config.vocab_size = len(self.word_to_id)
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=self.save_path)  # 读取保存的模型
    def predict(self, txt, k = 1):
        data_id = [[self.word_to_id[x] for x in list(txt.decode('utf-8')) if x in self.word_to_id]]     #; print 'data_id : ', data_id
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.seq_length)
        y_pred_prob = self.session.run(self.model.y_pred_prob, feed_dict={self.model.input_x: x_pad, self.model.keep_prob: 1.0})
        y_pred_prob_list = y_pred_prob[0].tolist()
        k_index = list(map(y_pred_prob_list.index, heapq.nlargest(k, y_pred_prob_list)))
        k_value = heapq.nlargest(k, y_pred_prob_list)
        res = []
        for i in range(len(k_index)):
            res.append({self.id_to_cat[k_index[i]]: round(k_value[i], 3)})
        return json.dumps({'sentence': txt, 'predict_result': res}, ensure_ascii=False)
    def label_sentence(self, txt, k = 1):
        try:
            data_id = [[self.word_to_id[x] for x in list(txt.decode('utf-8')) if x in self.word_to_id]]     #; print 'data_id : ', data_id
            x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.seq_length)
            y_pred_prob = self.session.run(self.model.y_pred_prob, feed_dict={self.model.input_x: x_pad, self.model.keep_prob: 1.0})
            y_pred_prob_list = y_pred_prob[0].tolist()
            k_index = list(map(y_pred_prob_list.index, heapq.nlargest(k, y_pred_prob_list)))
            k_value = heapq.nlargest(k, y_pred_prob_list)
            res = []
            for i in range(len(k_index)):
                res.append((self.id_to_cat[k_index[i]], round(k_value[i], 3)))
            return res[0]
        except Exception, e:
            logging.warn('label_sentence_error=%s' % (str(repr(e))))
            return None

if __name__ == '__main__':
    p = Predictor()
    res = p.label_sentence('怎么找工作', 3)
    #res = p.predict('怎么找工作', 3)
    print json.dumps(res, ensure_ascii=False)
    a=1
