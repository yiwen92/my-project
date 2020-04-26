import logging, traceback
from config import FLAGS, SEQ_LEN
import tensorflow as tf
from seg_utils import Tokenizer
from embedding import Encoder
from utils import sigmoid
import numpy as np
from data_utils import seq2ids

class queryDocSim:
    def __init__(self, model_type='rnn', ckpt_num=14000):
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
        tf.train.Saver().restore(self.session, self.estimator_save_name)        # 加载模型

    def run_step(self, entity, entity_list):
        x_batch = np.array([seq2ids(entity)])
        y_batch = np.array([[seq2ids(e) for e in entity_list]])
        fetch = self.session.run(
            {'emb_a': self.encoder.emb_a, 'emb_b': self.encoder.emb_b, 'sim_ab': self.encoder.sim_ab}, \
            feed_dict={self.a_in: x_batch, self.b_in: y_batch, self.is_training: False})
        res = {entity + "-" + e: fetch['sim_ab'][0][i] for i, e in enumerate(entity_list)}
        sorted_res = sorted(res.items(), key=lambda d: d[1], reverse=True)
        return sorted_res

    def sim(self, query, doc, topk=5):
        sim_sum = 0.0
        tmp = self.tokenizer.select_important_tokens(doc)
        sim_res = self.run_step(query, tmp)
        prob_res = [(k, sigmoid(v)) for k, v in sim_res]
        for k, v in prob_res[:topk]:
            sim_sum += v
        sim_avg = round(sim_sum / topk, 3)
        return sim_avg, prob_res

    def cal_sim(self, req_dict):
        similarity, prob_res = 0.0, []
        try:
            query = req_dict['request']['p']['query']
            doc = req_dict['request']['p']['doc']
            similarity, prob_res = self.sim(query, doc)
        except Exception as e:
            logging.warning("run_error: %s" % traceback.format_exc())
        return similarity, prob_res

if __name__ == "__main__":
    text = "1、熟悉Python、C、C++、Matlab、java编程语言中的一种或多种；2、熟练使用Caffe、Tensorflow、PyTorch等开源深度学习框架中的一种或多种；\
3、扎实的机器学习、深度学习，NLP理论基础，熟悉深度学习在自然语言处理以及领域知识图谱构建中的应用，实现切实有效的深度学习模型；\
4、研究语义建模、自然语言理解、机器阅读理解、文本分类、搜索问答等NLP关键前沿技术的研发工作；\
5、熟悉文本处理相关研发工作, 包括但不限于数据挖掘、情感分析、自然语言NLP处理、信息抽取、智能问答、知识图谱、语义理解、意图识别, 机器翻译, 机器写作等； \
6、有扎实的数学基础，熟悉并能灵活运用高数、线性代数、概率论等数学知识；\
7、良好的英文阅读能力和销售动力；\
8、具备良好的逻辑思维能力与沟通表达能力。hr专员"
    eS = queryDocSim(ckpt_num=25000); eS.sim("法务经理", text)