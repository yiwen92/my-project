import xlnet, logging, kenlm
from seg_utils import Tokenizer, PUNCTUATION_LIST, STOP_WORDS
from config import FLAGS, conf
import tensorflow as tf
from data_utils import preprocess_text, SEP_ID, CLS_ID
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FLAGS = tf.app.flags.FLAGS

def init_log():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='log',
                        filemode='a')

class language_model:
    def __init__(self):
        self.lm = kenlm.Model(conf.lm)

    def cal_weight_lm(self, senten2term):
        total_score = self.lm.perplexity(' '.join(senten2term))
        weight, weight_sum = [], 0.0
        for i in range(len(senten2term)):
            tmp = [senten2term[j] for j in range(len(senten2term)) if i != j]
            score = self.lm.perplexity((' '.join(tmp)))
            val = total_score / score
            if senten2term[i] in PUNCTUATION_LIST or senten2term[i] in STOP_WORDS: val = 0.0
            weight.append((senten2term[i], val))
            weight_sum += val
        token_weight = [(k, round(v/weight_sum, 3)) for k, v in weight]
        return token_weight

class query_weight:
    def __init__(self, ckpt_num=0, is_training=False):
        init_log()  ; batch_size = 1
        logging.info("Init query weight model ...")
        self.sp = Tokenizer()
        self.lm = language_model()
        tf.logging.set_verbosity(tf.logging.INFO)
        tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
        self.input_ids = tf.placeholder(dtype=tf.int64, shape=[batch_size, FLAGS.seq_len], name="input_ids")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[batch_size, FLAGS.seq_len], name="segment_ids")
        self.input_mask = tf.placeholder(dtype=tf_float, shape=[batch_size, FLAGS.seq_len], name="input_mask")
        self.label_ids = tf.placeholder(dtype=tf.int64, shape=[batch_size], name="label_ids")
        inp = tf.transpose(self.input_ids, [1, 0])
        seg_id = tf.transpose(self.segment_ids, [1, 0])
        inp_mask = tf.transpose(self.input_mask, [1, 0])
        self.sess = tf.Session()
        xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
        run_config = xlnet.create_run_config(is_training, True, FLAGS)

        xlnet_model = xlnet.XLNetModel(xlnet_config=xlnet_config, run_config=run_config, input_ids=inp, seg_ids=seg_id, input_mask=inp_mask)
        self.output, self.attn_prob, self.attention_out = xlnet_model.output_encode, xlnet_model.attn_prob, xlnet_model.attention_out

        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))
        xlnet_model.saver.restore(self.sess, FLAGS.init_checkpoint + "/model.ckpt-" + str(ckpt_num))
        #### load pretrained models
        # scaffold_fn = model_utils.init_from_checkpoint(FLAGS)
        logging.info("Init query weight model finished ...")

    def run_step(self, text):
        cur_sent = preprocess_text(text.strip(), lower=FLAGS.uncased)
        tokens, ids = self.sp.encode_ids(cur_sent)
        sent_len, diff_len = len(ids), FLAGS.seq_len - len(ids)
        input_ids = ids + [SEP_ID] * (diff_len-1) + [CLS_ID]         #  cat_data = np.concatenate([inp, a_data, sep_array, b_data, sep_array, cls_array])
        input_tokens = tokens + ["<sep>"] * (diff_len-1) + ["<cls>"]
        input_mask = [1] + [0] * (sent_len-1) + [1] * diff_len
        segment_ids = [0] * sent_len + [2] * diff_len    # seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] + [1] * b_data.shape[0] + [1] + [2])

        logging.info("text: %s, seg_text: %s" % (text, " ".join([str(x) for x in tokens])))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        feed_dict = {self.input_ids: [input_ids], self.segment_ids: [segment_ids], self.input_mask: [input_mask]}
        fetch = self.sess.run([self.output, self.attn_prob, self.attention_out], feed_dict)
        out_encode, atten_prob =  fetch[0], fetch[1]
        weight0 = self.cal_weight(out_encode, input_tokens)
        weight_attn = self.weight_attenprob(atten_prob, input_tokens)
        weight_lm = self.lm.cal_weight_lm(tokens[1:])
        weight_idf = self.sp.cal_weight_idf(tokens[1:])
        weight = self.merge_weight([(weight_attn, 0.6),(weight_idf, 0.2), (weight_lm, 0.2)])
        a=1

    def merge_weight(self, weight_tuple):
        weight, weight_sum = [], 0.0
        for j in range(len(weight_tuple[0][0])):
            tmp = 0.0
            for i in range(len(weight_tuple)):
                (word, val), coef = weight_tuple[i][0][j], weight_tuple[i][1]
                tmp += val * coef
            weight.append((weight_tuple[0][0][j][0], tmp))
            weight_sum += tmp
        token_weight = [(k, round(v / weight_sum, 3)) for k, v in weight]
        return token_weight

    def weight_attenprob(self, attention_probs, input_tokens):
        weights, weights_sum = [], 0.0
        special_words = ['▁', '<sep>', '<cls>']
        (row, col, batch, dim) = attention_probs.shape
        for i in range(col):
            tmp = 0.0
            for j in range(row):
                if i == j: continue
                prob = attention_probs[j][i][0][0]
                tmp += prob
            if input_tokens[i] in PUNCTUATION_LIST or input_tokens[i] in STOP_WORDS: tmp = 0.0
            weights.append(tmp)
            weights_sum += tmp
        for i in range(len(weights)):
            weights[i] = round(weights[i] / weights_sum, 3)
        token_weight = [(input_tokens[i], weights[i]) for i in range(len(weights)) if input_tokens[i] not in special_words]
        return token_weight

    def cal_weight(self, encode_vects, input_tokens):
        weight, weight_sum = [], 0.0
        vects = encode_vects[0]
        vect = np.sum(encode_vects, axis=1)[0]
        special_words = ['▁' ,'<sep>', '<cls>']
        for i in range(len(vects)):
            if input_tokens[i] in special_words: continue
            val = cal_sim(vect, vects[i])
            if input_tokens[i] in PUNCTUATION_LIST or input_tokens[i] in STOP_WORDS: val = 0.0
            weight.append(val)
            weight_sum += val
        for i in range(len(weight)):
            weight[i] /= weight_sum
        #token_weight = [(input_tokens[i], weight[i-1]) if input_tokens[i] not in special_words else (input_tokens[i], 0.0) for i in range(len(input_tokens))]
        token_weight = [(input_tokens[i], round(weight[i - 1], 3)) for i in range(len(input_tokens)) if input_tokens[i] not in special_words]
        return token_weight

def cal_sim(vec1, vec2):
    """
    vec_sum, vec1_sum, vec2_sum = 0.0, 0.0, 0.0
    for i in range(len(vec1)):
        vec_sum += vec1[i] * vec2[i]
        vec1_sum += vec1[i] * vec1[i]
        vec2_sum += vec2[i] * vec2[i]
    sim = vec_sum / (math.sqrt(vec1_sum) * math.sqrt(vec2_sum))
    return sim
    """
    return cosine_similarity([vec1], [vec2])[0][0]

if __name__ == "__main__":
    qw = query_weight(801000);
    qw.run_step("web前端 javascript");