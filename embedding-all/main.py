import json, os
import numpy as np
import tensorflow as tf
from utils import re_en, cal_vec_sim
from config import conf, SEQ_LEN, MAX_NUM_NEG
from embedding import Encoder
from data_utils import batch_iter
from sklearn.metrics.pairwise import cosine_similarity

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
        self.save_dir = conf.models_path + model_type
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.save_name = conf.models_path + model_type + "/" + model_type + "_model"
        # 模型输入
        self.a_in = tf.placeholder(tf.int32, [None, SEQ_LEN], name='a')         # [batch_size, SEQ_LEN]
        self.b_in = tf.placeholder(tf.int32, [None, None, SEQ_LEN], name='b')   # [batch_size, 1 + MAX_NUM_NEG, SEQ_LEN]
        self.is_training = tf.placeholder_with_default(True, shape=())
        # 创建session
        self.session = tf.Session()
        self.word_embed, self.intent_embed = self.encoder.create_tf_embed(self.a_in, self.b_in, self.is_training)  # 语义编码

    def gen_train_samples(self, file_path):
        train_samples = json.load(open(file_path, encoding="utf8"))
        samples = []
        for k, (p, n) in train_samples.items():
            kid = self.seq2ids(k)
            pid = self.seq2ids(p)
            nid = [self.seq2ids(e) for e in n]
            ts = train_sample(kid, pid, nid)
            samples.append(ts)
        X = np.array([e.entity for e in samples])
        Y = np.array([[e.pos_entity] + e.neg_entitys for e in samples])
        return X, Y

    def train(self):
        # 载入训练集
        x_train, y_train = self.gen_train_samples(conf.train_samples)
        x_valid, y_valid = self.gen_train_samples(conf.valid_samples)
        # 模型损失计算
        # 配置 Saver
        saver = tf.train.Saver()
        sim_op, sim_emb = self.encoder.tf_sim(self.word_embed, self.intent_embed)
        self.loss = self.encoder.tf_loss(sim_op, sim_emb)
        train_op = tf.train.AdamOptimizer().minimize(self.loss)
        print('Training and evaluating...')
        self.session.run(tf.global_variables_initializer())
        total_batch = 0  # 总批次
        min_loss_val = 1e8  # 最佳验证集损失
        print_per_batch = 1
        for epoch in range(conf.num_epochs):
            batch_train = batch_iter(x_train, y_train, conf.batch_size)
            for x_batch, y_batch in batch_train:
                Asess_out = self.session.run(
                    {'loss': self.loss, 'train_op': train_op, 'a_in': self.a_in, 'b_in': self.b_in,
                     'word_embed': self.word_embed, 'intent_embed': self.intent_embed, \
                     'debug_rnn': self.encoder.debug_rnn, 'debug_sim': self.encoder.debug_sim},
                    feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.is_training: True}
                )
                if total_batch % print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    loss_val = self.evaluate(x_valid, y_valid)
                    if loss_val < min_loss_val:
                        # 保存最好结果
                        min_loss_val = loss_val
                        saver.save(sess=self.session, save_path=self.save_name)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    print("total_batch: %d\tloss_train: %.3f\tloss_valid: %.3f\timproved: %s" % \
                          (total_batch, Asess_out['loss'], loss_val, improved_str))
                total_batch += 1
        aa=1

    def predict(self, entity, entity_list):
        tf.train.Saver().restore(self.session, self.save_name)
        x_batch = np.array([self.seq2ids(entity)])
        y_batch = np.array([[self.seq2ids(e) for e in entity_list]])
        fetch = self.session.run({'emb_a': self.encoder.emb_a, 'emb_b': self.encoder.emb_b, 'sim_ab': self.encoder.sim_ab}, \
                                 feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.is_training: False})
        emb_a = [fetch['emb_a'][0]]
        emb_b = fetch['emb_b'][0]
        a = cosine_similarity(emb_a, emb_b)
        a=1

    def evaluate(self, x_, y_):
        """评估在某一数据上的准确率和损失"""
        data_len = len(x_)
        batch_eval = batch_iter(x_, y_, 128)
        total_loss = 0.0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            loss = self.session.run(self.loss, feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.is_training: False})
            total_loss += loss * batch_len
        return total_loss / data_len

    def seq2ids(self, text, max_length=SEQ_LEN):
        """将文本转换为id表示"""
        data = token2list(text)
        data_id = [self.word2id.get(x, 0) for x in data]
        if len(data_id) > max_length: x_pad = data_id[: max_length]
        else: x_pad = data_id + [0] * (max_length - len(data_id))
        return x_pad

def test():
    x_in = tf.placeholder(tf.int32, [None, None, 3], name='b')
    alternate = tf.map_fn(lambda x: tf.shape(x)[1], x_in)
    for i in range(x_in.get_shape().as_list()[1]):
        s=x_in[:, i, :]
        a=1
    i0 = tf.constant(0)
    m0 = tf.ones([2, 2])
    c = lambda i: i < 10
    b = lambda i: [i]
    s=tf.while_loop(
        c, b, loop_vars=[x_in],
        shape_invariants=[tf.TensorShape([None, None, 3])])
    a=1

if __name__ == "__main__":
    #test()
    es = entity_similar()
    #es.get_train_corp()
    #es.train()
    es.predict('java', ['测试','android', '销售'])
    pass