import os
import numpy as np
import tensorflow as tf
from config import conf, SEQ_LEN, FLAGS, MAX_NUM_NEG
from embedding import Encoder, tf_sim, tf_loss, cross_entropy_loss
from data_utils import batch_iter, gen_train_samples, seq2ids
from sklearn.metrics.pairwise import cosine_similarity
from seg_utils import Tokenizer
from utils import sigmoid

class entity_similar:
    def __init__(self, model_type='rnn', ckpt_num=0):
        self.encoder = Encoder(model_type)
        self.save_dir = FLAGS.serving_model_dir + model_type
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.save_name = FLAGS.serving_model_dir + model_type + "/" + model_type + "_model"
        self.estimator_save_name = FLAGS.model_dir + "/model.ckpt-" + str(ckpt_num)
        # 模型输入
        self.a_in = tf.placeholder(tf.int32, [None, SEQ_LEN], name='a')         # [batch_size, SEQ_LEN]
        self.b_in = tf.placeholder(tf.int32, [None, None, SEQ_LEN], name='b')   # [batch_size, 1 + MAX_NUM_NEG, SEQ_LEN]
        self.label = tf.placeholder(tf.int32, [None, None], name='b')   # [batch_size, 1 + MAX_NUM_NEG]
        self.is_training = tf.placeholder_with_default(True, shape=())
        # 创建session
        self.session = tf.Session()
        self.word_embed, self.intent_embed = self.encoder.create_tf_embed(self.a_in, self.b_in, self.is_training)  # 语义编码

    def train(self):
        dubug_info = {}
        tf.logging.set_verbosity(tf.logging.INFO)
        # 载入训练集
        x_train, y_train, label_train = gen_train_samples(FLAGS.train_samples)
        x_valid, y_valid, label_valid = gen_train_samples(FLAGS.valid_samples)
        # 模型损失计算
        # 配置 Saver
        saver = tf.train.Saver()
        sim_op, sim_emb = tf_sim(self.word_embed, self.intent_embed)
        #self.loss = tf_loss(sim_op, sim_emb)
        self.loss = cross_entropy_loss(sim_op, self.label, dubug_info)
        train_op = tf.train.AdamOptimizer().minimize(self.loss)
        print('Training and evaluating...')
        self.session.run(tf.global_variables_initializer())
        total_batch = 0  # 总批次
        min_loss_val = 1e8  # 最佳验证集损失
        print_per_batch = 1
        for epoch in range(conf.num_epochs):
            batch_train = batch_iter(x_train, y_train, label_train, FLAGS.batch_size)
            for x_batch, y_batch, label_batch in batch_train:
                Asess_out = self.session.run(
                    {'loss': self.loss, 'train_op': train_op, 'a_in': self.a_in, 'b_in': self.b_in,
                     'word_embed': self.word_embed, 'intent_embed': self.intent_embed, \
                     'debug_infoa': self.encoder.debug_info_a, 'debug_infob': self.encoder.debug_info_b, \
                     'debug_sim': self.encoder.debug_sim, "dubug_info": dubug_info},
                    feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.label: label_batch, self.is_training: True}
                )
                if total_batch % print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    loss_val = self.evaluate(x_valid, y_valid, label_valid)
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
        tf.train.Saver().restore(self.session, self.estimator_save_name)
        x_batch = np.array([seq2ids(entity)])
        y_batch = np.array([[seq2ids(e) for e in entity_list]])
        #for _ in range(3):
        fetch = self.session.run({'emb_a': self.encoder.emb_a, 'emb_b': self.encoder.emb_b, 'sim_ab': self.encoder.sim_ab}, \
                                 feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.is_training: False})
        print(fetch['sim_ab'][0])
        emb_a = [fetch['emb_a'][0]]
        emb_b = fetch['emb_b'][0]
        a = cosine_similarity(emb_a, emb_b)
        res = {entity + "-" +e: fetch['sim_ab'][0][i] for i, e in enumerate(entity_list)}
        sorted_res = sorted(res.items(), key=lambda d: d[1], reverse=True)
        a=1

    def evaluate(self, x_, y_, label_):
        """评估在某一数据上的准确率和损失"""
        data_len = len(x_)
        batch_eval = batch_iter(x_, y_, label_, 128)
        total_loss = 0.0
        for x_batch, y_batch, label_batch in batch_eval:
            batch_len = len(x_batch)
            loss = self.session.run(self.loss, feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.label: label_batch, self.is_training: False})
            total_loss += loss * batch_len
        return total_loss / data_len

class entitySimilar:
    def __init__(self, model_type='rnn', ckpt_num=0):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.tokenizer = Tokenizer()
        self.encoder = Encoder(model_type)
        self.estimator_save_name = FLAGS.model_dir + "/model.ckpt-" + str(ckpt_num)
        # 模型输入
        self.a_in = tf.placeholder(tf.int32, [None, SEQ_LEN], name='a')  # [batch_size, SEQ_LEN]
        self.b_in = tf.placeholder(tf.int32, [None, None, SEQ_LEN], name='b')  # [batch_size, 1 + MAX_NUM_NEG, SEQ_LEN]
        self.is_training = tf.placeholder_with_default(False, shape=())
        # 创建session
        self.session = tf.Session()
        self.word_embed, self.intent_embed = self.encoder.create_tf_embed(self.a_in, self.b_in, self.is_training)  # 语义编码
        tf.train.Saver().restore(self.session, self.estimator_save_name)

    def run_step(self, entity, entity_list):
        x_batch = np.array([seq2ids(entity)])
        y_batch = np.array([[seq2ids(e) for e in entity_list]])
        fetch = self.session.run(
            {'emb_a': self.encoder.emb_a, 'emb_b': self.encoder.emb_b, 'sim_ab': self.encoder.sim_ab}, \
            feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.is_training: False})
        res = {entity + "-" + e: fetch['sim_ab'][0][i] for i, e in enumerate(entity_list)}
        sorted_res = sorted(res.items(), key=lambda d: d[1], reverse=True)
        return sorted_res

    def analyze(self, word, text):
        tmp = self.tokenizer.select_important_tokens(text)
        sim_res = self.run_step(word, tmp)
        prob_res = [(k, sigmoid(v)) for k, v in sim_res]
        pass

if __name__ == "__main__":
    text = "1、熟悉Python、C、C++、Matlab、java编程语言中的一种或多种；2、熟练使用Caffe、Tensorflow、PyTorch等开源深度学习框架中的一种或多种；\
3、扎实的机器学习、深度学习，NLP理论基础，熟悉深度学习在自然语言处理以及领域知识图谱构建中的应用，实现切实有效的深度学习模型；\
4、研究语义建模、自然语言理解、机器阅读理解、文本分类、搜索问答等NLP关键前沿技术的研发工作；\
5、熟悉文本处理相关研发工作, 包括但不限于数据挖掘、情感分析、自然语言NLP处理、信息抽取、智能问答、知识图谱、语义理解、意图识别, 机器翻译, 机器写作等； \
6、有扎实的数学基础，熟悉并能灵活运用高数、线性代数、概率论等数学知识；\
7、良好的英文阅读能力和销售动力；\
8、具备良好的逻辑思维能力与沟通表达能力。hr专员"
    #eS = entitySimilar(ckpt_num=14000); eS.analyze("nlp", text)
    es = entity_similar(ckpt_num=8500)
    #es.get_train_corp()
    es.train()
    #es.predict('java', ['java', '测试','android', '销售', '自然语言处理'])
    pass