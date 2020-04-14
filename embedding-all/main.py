import os
import numpy as np
import tensorflow as tf
from config import conf, SEQ_LEN, FLAGS, MAX_NUM_NEG
from embedding import Encoder, tf_sim, tf_loss
from data_utils import batch_iter, gen_train_samples, seq2ids
from sklearn.metrics.pairwise import cosine_similarity

class entity_similar:
    def __init__(self, model_type='rnn', ckpt_num=0):
        self.encoder = Encoder(model_type)
        self.save_dir = conf.models_path + model_type
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.save_name = conf.models_path + model_type + "/" + model_type + "_model"
        #self.save_name = FLAGS.model_dir + "/model.ckpt-" + str(ckpt_num)
        # 模型输入
        self.a_in = tf.placeholder(tf.int32, [None, SEQ_LEN], name='a')         # [batch_size, SEQ_LEN]
        self.b_in = tf.placeholder(tf.int32, [None, 1 + MAX_NUM_NEG, SEQ_LEN], name='b')   # [batch_size, 1 + MAX_NUM_NEG, SEQ_LEN]
        self.is_training = tf.placeholder_with_default(True, shape=())
        # 创建session
        self.session = tf.Session()
        self.word_embed, self.intent_embed = self.encoder.create_tf_embed(self.a_in, self.b_in, self.is_training)  # 语义编码

    def train(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        # 载入训练集
        x_train, y_train = gen_train_samples(conf.train_samples)
        x_valid, y_valid = gen_train_samples(conf.valid_samples)
        # 模型损失计算
        # 配置 Saver
        saver = tf.train.Saver()
        sim_op, sim_emb = tf_sim(self.word_embed, self.intent_embed)
        self.loss = tf_loss(sim_op, sim_emb)
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
                     'debug_infoa': self.encoder.debug_info_a, 'debug_infob': self.encoder.debug_info_b, \
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
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.train.Saver().restore(self.session, self.save_name)
        x_batch = np.array([seq2ids(entity)])
        y_batch = np.array([[seq2ids(e) for e in entity_list]])
        #for _ in range(3):
        fetch = self.session.run({'emb_a': self.encoder.emb_a, 'emb_b': self.encoder.emb_b, 'sim_ab': self.encoder.sim_ab}, \
                                 feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.is_training: False})
        print(fetch['sim_ab'][0])
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

if __name__ == "__main__":
    es = entity_similar(ckpt_num=17700)
    #es.get_train_corp()
    es.train()
    #es.predict('java', ['java', '测试','android', '销售'])
    pass