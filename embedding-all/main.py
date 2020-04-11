import json, jieba
import numpy as np
import tensorflow as tf
from utils import re_en
from config import conf, SEQ_LEN, MAX_NUM_NEG
from embedding import Encoder

def token2list(token):
    data = []
    if re_en.fullmatch(token):
        data.append(token)
    else:
        for e in list(token):
            data.append(e)
    return data

class train_sample():
    def __init__(self, entity, pos_entity, neg_entitys):
        self.entity = entity
        self.pos_entity = pos_entity
        self.neg_entitys = neg_entitys

class entity_similar:
    def __init__(self, model_type='rnn'):
        self.word2id = json.load(open(conf.vocab, encoding="utf8"))
        self.encoder = Encoder(model_type)

    def get_train_corp(self):
        text = [e.strip().split("\t")[33].replace("\\n", "").lower() for e in open("data/jddata_1000", encoding="utf8").readlines()]
        with open("data/train.txt", "w", encoding="utf8") as fin:
            fin.write("\n".join(text))

    def train(self):
        train_samples = json.load(open(conf.train_samples, encoding="utf8"))
        samples = []
        for k, (p, n) in train_samples.items():
            kid = self.seq2ids(k)
            pid = self.seq2ids(p)
            nid = [self.seq2ids(e) for e in n]
            ts = train_sample(kid, pid, nid)
            samples.append(ts)
        X = np.array([e.entity for e in samples])
        Y = np.array([[e.pos_entity] + e.neg_entitys for e in samples])
        batch_size = 1
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.a_in = tf.placeholder(tf.int32, (batch_size, SEQ_LEN), name='a')
            self.b_in = tf.placeholder(tf.int32, (batch_size, 1 + MAX_NUM_NEG, SEQ_LEN), name='b')
            is_training = tf.placeholder_with_default(True, shape=())
            self.word_embed, self.intent_embed = self.encoder.create_tf_embed(self.a_in, self.b_in, is_training)
            self.sim_op, sim_emb = self.encoder.tf_sim(self.word_embed, self.intent_embed)
            loss = self.encoder.tf_loss(self.sim_op, sim_emb)
            train_op = tf.train.AdamOptimizer().minimize(loss)
            # train tensorflow graph
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            indices = [i for i, e in enumerate(X)] #np.random.permutation(len(X))
            batches_per_epoch = len(X) // batch_size
            for i in range(batches_per_epoch):
                end_idx = (i + 1) * batch_size
                start_idx = i * batch_size
                batch_a = X[indices[start_idx:end_idx]]
                batch_pos_b = Y[indices[start_idx:end_idx]]
                Asess_out = self.session.run(
                    {'loss': loss, 'train_op': train_op, 'a_in': self.a_in, 'b_in': self.b_in, 'word_embed': self.word_embed, 'intent_embed': self.intent_embed, \
                     'debug_rnn': self.encoder.debug_rnn, 'debug_sim': self.encoder.debug_sim},
                    feed_dict={self.a_in: batch_a,
                               self.b_in: batch_pos_b,
                               is_training: True}
                    )
                a=1

            a_in, b_in = X[0], Y[0]
            feed_dict = {self.a_in: [a_in], self.b_in: [b_in]}
            fetch = self.session.run([self.encoder.a], feed_dict)
            a=1


    def seq2ids(self, text, max_length=SEQ_LEN):
        """将文本转换为id表示"""
        data = token2list(text)
        data_id = [self.word2id.get(x, 'UNKNOWN') for x in data]
        if len(data_id) > max_length: x_pad = data_id[: max_length]
        else: x_pad = data_id + [0] * (max_length - len(data_id))
        return x_pad

if __name__ == "__main__":
    es = entity_similar()
    #es.get_train_corp()
    es.train()
    pass