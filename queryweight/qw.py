import xlnet, logging, kenlm, re, time, math, json, traceback, sys
from seg_utils import Tokenizer, PUNCTUATION_LIST, SPECIAL_WORDS_CUSTOM
from config import FLAGS, conf
import tensorflow as tf
from data_utils import preprocess_text, SEP_ID, CLS_ID
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import xgboost as xgb
from utils import get_feature, STOP_WORDS, PLACE_NAMES, FUNC_DICT, INDUS_DICT, parse_xgb_dict, predict_proba
from xgboost import DMatrix
from scipy import sparse

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
        token_weight, total_score = [], self.lm.perplexity(' '.join(senten2term))
        for i in range(len(senten2term)):
            tmp = [senten2term[j] for j in range(len(senten2term)) if i != j]
            val = self.lm.perplexity((' '.join(tmp))) / total_score
            token_weight.append((senten2term[i], val))
        return token_weight

special_words = ['▁' ,'<sep>', '<cls>']

class query_weight:
    def __init__(self, ckpt_num=156000, is_training=False):
        #init_log()
        self.logs = {}
        batch_size = 1
        logging.info("Init query weight model ...")
        self.sp = Tokenizer()
        self.lm = language_model()
        self.xgb_model = xgb.Booster(model_file=conf.rank_model)
        #self.xgb_dict = parse_xgb_dict(conf.rank_model + '.txt')
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

    def on_weight_begin(self):
         logging.debug('on_weight_begin')
         self.t_begin = time.time()

    def on_weight_end(self):
         logging.debug('on_weight_end')
         phead = '[on_weight_end] | log_info=%s | cost=%.3fs'
         logging.info(phead % (json.dumps(self.logs, ensure_ascii=False), (time.time()-self.t_begin)))

    def run(self, req_dict):
        self.on_weight_begin()
        result = {}
        try:
            query = req_dict["request"]["p"]["query"]
            token_weights = self.run_step(query)
            for t, w in token_weights:
                result[t] = w
        except Exception as e:
            logging.warning("run_error: %s" % traceback.format_exc())
        self.on_weight_end()
        return result

    def run_step(self, text):
        cur_sent = preprocess_text(text.strip(), lower=FLAGS.uncased)
        tokens, ids = self.sp.encode_ids(cur_sent)
        sent_len, diff_len = len(ids) - 1, FLAGS.seq_len - len(ids)

        input_ids = ids + [SEP_ID] * (diff_len-1) + [CLS_ID]         #  cat_data = np.concatenate([inp, a_data, sep_array, b_data, sep_array, cls_array])
        input_tokens = tokens + ["<sep>"] * (diff_len-1) + ["<cls>"]
        input_mask = [1] + [0] * sent_len + [1] * diff_len
        segment_ids = [0] * (sent_len + 1) + [2] * diff_len    # seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] + [1] * b_data.shape[0] + [1] + [2])
        input_ids, input_tokens, input_mask, segment_ids = input_ids[:FLAGS.seq_len], input_tokens[:FLAGS.seq_len], input_mask[:FLAGS.seq_len], segment_ids[:FLAGS.seq_len]
        '''
       logging.info("text: %s, seg_text: %s" % (text, " ".join([str(x) for x in tokens])))
       logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
       logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
       logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
       '''
        self.logs['xlnet_input_info'] = {'text':text,'seg_text':" ".join([str(x) for x in tokens]),'input_ids':" ".join([str(x) for x in input_ids]), \
           'input_mask':" ".join([str(x) for x in input_mask]),'segment_ids':" ".join([str(x) for x in segment_ids])}
        #logging.info(json.dumps(il, ensure_ascii=False))

        feed_dict = {self.input_ids: [input_ids], self.segment_ids: [segment_ids], self.input_mask: [input_mask]}
        fetch = self.sess.run([self.output, self.attn_prob, self.attention_out], feed_dict)
        out_encode, atten_prob =  fetch[0], fetch[1]
        #weight0 = normalization(self.cal_weight(out_encode, input_tokens))
        weight_attn = normalization(self.weight_attenprob(atten_prob, tokens))
        weight_idf = normalization(self.sp.cal_weight_idf(tokens[1:]))
        weight_lm = normalization(self.lm.cal_weight_lm(tokens[1:]))
        weight_rule = self.merge_weight([(weight_attn, 0.5),(weight_idf, 0.5), (weight_lm, 0.5)])
        self.weight_attn, self.weight_idf, self.weight_lm = weight_attn, weight_idf, weight_lm
        sen2terms = [e for e in tokens[1:]]
        weightrank = self.rank_weight(sen2terms, weight_attn, weight_idf, weight_lm)
        weight_rank = normalization(weightrank)
        weight = self.merge_weight([(weight_rank, 0.7), (weight_rule, 0.0)])        # 0.6-0.4
        self.logs['weight_results'] = {'weight_rank':' '.join([str(k)+':'+str(v) for k, v in weight_rank]),'weight_rule':' '.join([str(k)+':'+str(v) for k, v in weight_rule]), \
              'weight': ' '.join([str(k) + ':' + str(v) for k, v in weight])}
        #logging.info(json.dumps(wl, ensure_ascii=False))
        return weight

    def rank_weight(self, sen2terms, weight_attn, weight_idf, weight_lm):
        tmp, score_sum = [], 1e-8
        for term in sen2terms:
            feature_vector, _ = get_feature(term, sen2terms, weight_attn, weight_idf, weight_lm)
            feature = np.array(feature_vector)
            feature_csr = sparse.csr_matrix(feature)
            input = DMatrix(feature_csr)
            score = self.xgb_model.predict(input)[0]
            prob = 1.0 / (1 + math.exp(-1 * score))        # xgboost 自带的预测函数
            #prob = predict_proba(self.xgb_dict, [feature_vector])   # 自定义的预测函数
            tmp.append((term, prob))
            score_sum += prob
        res = [(k, round(v / score_sum, 3)) for k, v in tmp]
        return res

    def merge_weight(self, weight_tuple):
        weight, weight_sum = [], 1e-8
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
        weights = []
        (row, col, batch, dim) = attention_probs.shape
        for j in range(col):
            tmp = 0.0
            for i in range(row):
                if i == j: continue
                tmp += attention_probs[i][j][0][0]
            weights.append(tmp)
        token_weight = [(input_tokens[i], weights[i]) for i in range(min(len(input_tokens), len(weights))) if input_tokens[i] not in special_words]
        token_weights = token_weight + [(input_tokens[i], 0.0) for i in range(len(token_weight) + 1, len(input_tokens))]
        return token_weights

    def cal_weight(self, encode_vects, input_tokens):
        vects, vect = encode_vects[0], np.sum(encode_vects, axis=1)[0]
        token_weights = [(input_tokens[i], cal_sim(vect, vects[i])) for i in range(len(vects)) if input_tokens[i] not in special_words]
        #token_weight = [(input_tokens[i], weight[i-1]) if input_tokens[i] not in special_words else (input_tokens[i], 0.0) for i in range(len(input_tokens))]
        return token_weights

def cal_sim(vec1, vec2):
    """
    vec_sum, vec1_sum, vec2_sum = 0.0, 0.0, 0.0
    for i in range(len(vec1)):
        vec_sum += vec1[i] * vec2[i];   vec1_sum += vec1[i] * vec1[i];  vec2_sum += vec2[i] * vec2[i]
    sim = vec_sum / (math.sqrt(vec1_sum) * math.sqrt(vec2_sum))
    return sim
    """
    return cosine_similarity([vec1], [vec2])[0][0]

def post_process(token_weights):
    results = []
    for token , weight in token_weights:
        if token.isdigit() and len(token) == 1: weight = weight * 0.2       # 单个数字降权处理
        if token in PLACE_NAMES: weight *= 0.3              # 地名降权
        if token in ['男','女','windows','linux','工程师','开发','程序','计算机','资深','国际','师','电话','前端',
                     '硕士','员','本科','助理','经理','游戏','算法','招聘']: weight *= 0.3
        if token in FUNC_DICT or token in INDUS_DICT or token in SPECIAL_WORDS_CUSTOM: weight *= 1.3     # 实体词升权
        results.append((token, weight))
    return results

def normalization(token_weights):
    results, weight_sum = [], 1e-8
    tmp = [(token, 0.0) if token in PUNCTUATION_LIST or token in STOP_WORDS else (token, weight) for token, weight in token_weights]
    tmp = post_process(tmp)
    for token, weight in tmp: weight_sum += weight
    results = [(token, round(weight / weight_sum, 3)) for token, weight in tmp]
    return results

def test(path):
    qw = query_weight()  ; qw_res = []
    #matchObj = re.compile(r'(.+)\t ([0-9]+)', re.M | re.I)
    matchObj = re.compile(r'(.+)\t([0-9]+)', re.M | re.I)
    total_num = len(open(path, encoding="utf8").readlines())
    for i, line in enumerate(tqdm(open(path, encoding="utf8"), total=total_num)):
        match_res = matchObj.match(line)
        #if not match_res: continue
        #query, freq = match_res.group(1), int(match_res.group(2))       #; query = "javascript开发工程师"
        query = line.strip().replace("\t", "")
        res = qw.run_step(query)
        qw_res.append(str(i+1) + "\t" + "\t".join([t + ":" + str(w) for t, w in res]) + "\n")
        #if i > 10: break
    with open("sort_search_data.res2", "w", encoding="utf8") as fin:
        fin.write("".join(qw_res))
    exit()

if __name__ == "__main__":
    try: query = sys.argv[1]
    except: query = "基金经理硕士3年" #"自然语言处理算法工程师招聘"
    req_dict = {"header": {}, "request": {"c": "", "m": "query_correct", "p": {"query": query}}}
    #test("get_jdcv_data/query.true")      # "corpus/sort_search_data" "get_jdcv_data/query.freq.csv" "get_jdcv_data/query.true"
    qw = query_weight()
    t0 = time.time()   ;   res = qw.run(req_dict); print("res: %s\tcost time: %f" % (json.dumps(res, ensure_ascii=False), time.time() - t0))
    pass