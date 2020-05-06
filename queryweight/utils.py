import re, json, math, os, logging, codecs, jieba
from collections import defaultdict
from tqdm import tqdm
from config import conf

re_ch = re.compile(u"([\u4e00-\u9fa5])",re.S)
re_year = re.compile(u'([0-9]*年)', re.M | re.I)
PUNCTUATION_LIST = ".。,，,、?？:：;；{}[]【】“‘’”《》/!！%……（）<>@#$~^￥%&*\"\'=+-_——「」"
NONE_STOPWORD = ["it"]
CUSTOM_STOPWORD = ["人","年","大"]
STOP_WORDS = [e.strip() for e in open(conf.stop_words, encoding="utf8").readlines() if e.strip() not in NONE_STOPWORD] + CUSTOM_STOPWORD

def load_place(path):
    res = []
    txt = [e.strip().split(",")[-1] for e in open(path, encoding="utf8").readlines()[1:]]
    for w in txt:
        if w.endswith("市") or w.endswith("省"): res.append(w[:-1])
        res.append(w)
    return res

PLACE_NAMES = load_place(conf.place_names)

def contain_chinese_word(sentence):
    if re_ch.findall(sentence): return True
    return False

def invalid_entity(word):
    for e in PLACE_NAMES:
        if e in word and word != e: return True     # 过滤实体中包含地址的词
    if re_year.findall(word): return True             # 过滤实体中包含时间的词
    return False

def gen_entity_dict():
    for file_name in ['__func__.txt', '__ind__.txt']:
        res = []
        text = open('dict/' + file_name, encoding='utf8').readlines()
        sub_word = ""
        for i, line in enumerate(text):
            #line = "前端开发1年"
            cur_word = line.strip().replace(" ", "")
            if cur_word in ['开发工程']: continue
            if sub_word and contain_chinese_word(cur_word) and sub_word in cur_word and cur_word.index(sub_word) == 0: continue
            elif invalid_entity(cur_word): continue
            else:
                res.append(cur_word + ' 10000\n')
                sub_word = cur_word
        with open('dict/' + file_name.replace("__", ""), "w", encoding="utf8") as fin:
            fin.write("".join(res))

def load_word_freq_dict(path, th=0):      # 加载词典
    matchObj = re.compile(r'(.+) ([0-9]+)', re.M | re.I)
    word_freq = {}
    if not os.path.exists(path):
        logging.warning("file not exists:" + path)
        return word_freq
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): continue
            matchRes = matchObj.match(line)
            word, freq = matchRes.group(1), int(matchRes.group(2))
            if freq < th: continue
            word_freq[word] = freq
    return word_freq
FUNC_DICT = load_word_freq_dict(conf.func_file)
INDUS_DICT = load_word_freq_dict(conf.indus_file)

def term_type(word_index, sen2terms):
    """
    0-中文: 前端开发，1-英文：web，2-数字：2000，3-符号：k，4-停用词：的，5-其它：app123
    """
    type_encode = [0] * 6
    ty_dict = {'ch': 0, 'en': 1, 'digit': 2, 'punct': 3, 'stopword': 4, 'other': 5}
    def is_ch(w):
        if re_ch.findall(w): return True
        return False
    def is_digit(w):
        if w.isdigit(): return True
        return False
    def is_en(w):
        if w.isalpha(): return True
        return False
    if word_index < 0 or word_index >= len(sen2terms):
        type_encode[ty_dict['other']] = 1
        return type_encode
    term = sen2terms[word_index]
    if is_ch(term): type_encode[ty_dict['ch']] = 1
    elif is_en(term): type_encode[ty_dict['en']] = 1
    elif is_digit(term): type_encode[ty_dict['digit']] = 1
    elif term in PUNCTUATION_LIST: type_encode[ty_dict['punct']] = 1
    elif term in STOP_WORDS: type_encode[ty_dict['stopword']] = 1
    else: type_encode[ty_dict['other']] = 1
    return type_encode

def entity_type(word_index, sen2terms):
    """ 0-行业词，1-职能词, 3-其它 """
    entiey_encode = [0] * 3
    ty_dict = {'indus': 0, 'func': 1, 'other': 2}
    if word_index < 0 or word_index >= len(sen2terms):
        entiey_encode[ty_dict['other']] = 1
        return entiey_encode
    term = sen2terms[word_index]
    if term in INDUS_DICT: entiey_encode[ty_dict['indus']] = 1
    elif term in FUNC_DICT: entiey_encode[ty_dict['func']] = 1
    else: entiey_encode[ty_dict['other']] = 1
    return entiey_encode

def term_position(word_index, sen2terms):
    position_encode = [0] * 3
    position_dict = {'begin': 0, 'end': 1, 'other': 2}
    if word_index == 0: position_encode[position_dict['begin']] = 1
    elif word_index == len(sen2terms) - 1: position_encode[position_dict['end']] = 1
    else: position_encode[position_dict['other']] = 1
    return position_encode

def get_feature(word, sen2terms, weight_attn, weight_idf, weight_lm):
    word_index = sen2terms.index(word)
    TermLength, TermOffset = [0], [0]
    TermPosition = term_position(word_index, sen2terms)     # 当前term的位置：[头部，尾部，其它]
    TermType = term_type(word_index, sen2terms)             # 当前term的类型：[中文，英文，数字，符号，停用词，其它, 职能词, 行业词]
    TermLength = [len(list(word))]                          # 当前term的字符级别长度
    TermOffset = [round(word_index / len(sen2terms), 3)]              # 当前term的偏移百分比
    PreTermPosition = term_position(word_index - 1, sen2terms)    # 前一个term的位置
    PreTermType = term_type(word_index - 1, sen2terms)        # 前一个term的类型
    BehindTermPosition = term_position(word_index + 1, sen2terms)  # 后一个term的位置
    BehindTermType = term_type(word_index + 1, sen2terms)  # 后一个term的类型
    TermEntityType = entity_type(word_index, sen2terms)     # 当前词的实体类型：[行业词，职能词，其它]
    PreTermEntityType = entity_type(word_index - 1, sen2terms)  # 前面词的实体类型
    BehindTermEntityType = entity_type(word_index + 1, sen2terms)  # 后面词的实体类型

    features = [
        ('词的位置', TermPosition, 'term_position', len(TermPosition)),
        ('词的类型', TermType, 'term_type', len(TermType)),
        ('长度', TermLength, 'term_length', 1),
        ('偏移百分比', TermOffset, 'term_offset', 1),
        ('前面词的位置', PreTermPosition, 'pre_term_position', len(PreTermPosition)),
        ('前面词的类型', PreTermType, 'pre_term_type', len(PreTermType)),
        ('后面词的位置', BehindTermPosition, 'behind_term_position', len(BehindTermPosition)),
        ('后面词的类型', BehindTermType, 'behind_term_type', len(BehindTermType)),
        ('词的attenion权重', [weight_attn[word_index][1]], 'term_attention_weight', 1),
        ('词的idf权重', [weight_idf[word_index][1]], 'term_idf_weight', 1),
        ('词的lm权重', [weight_lm[word_index][1]], 'term_lm_weight', 1),
        ('词的实体类型', TermEntityType, 'term_entity_type', len(TermEntityType)),
        ('前面词的实体类型', PreTermEntityType, 'pre_term_entity_type', len(PreTermEntityType)),
        ('词的实体类型', BehindTermEntityType, 'behind_term_entity_type', len(BehindTermEntityType)),
        ]
    feature_vector = []; fidindex = 0; fmap = []
    for fid in features:
        feature_vector.extend(fid[1])
        for i in range(len(fid[1])):
            fmap.append("\t".join([str(fidindex), fid[2]+":"+str(i), "q"]))
            fidindex += 1
    return feature_vector, fmap

def filter_ids(in_path, out_path):
    """
    user_id: 用户ID, jd_id: 职位ID, resume_id: 简历ID
    status: 状态: 1-投递成功, 2-被查看 ,3-筛选通过, 4-面试通知, 5-简历不合适, 6-简历被转发, 99-取消应聘
    type: 0-未知, 1-主动投递, 2-高意向, 3-逸橙推荐, 4-内推
    """
    status_static, type_static, res = defaultdict(int), defaultdict(int), []
    total_num = len(open(in_path, encoding="utf8").readlines())
    for i, line in enumerate(tqdm(open(in_path, encoding="utf8"), total=total_num)):
        if i == 0: res.append(line)
        user_id, jd_id, resume_id, status, type = line.strip().split("\t")
        status_static[status] += 1; type_static[type] += 1
        if status not in ['3', '4'] or type not in ['1', '2', '3', '4']: continue
        res.append(line)
    print("status_static: %s\ttype_static: %s" % (json.dumps(status_static), json.dumps(type_static)))
    print("read file :%s\twrite file: %s" % (in_path, out_path))
    with open(out_path, "w", encoding="utf8") as fin:
        fin.write("".join(res))

def cal_ndcg(label_list, topk):
    label_list = [int(e) for e in label_list]
    dcg, idcg = 0.0, 0.0
    pred = label_list[:topk]
    label = sorted(label_list, key=lambda d: d, reverse=True)[: topk]
    diff = topk - len(label_list)
    if diff > 0:        # 分数补全
        pred = pred + [0] * diff
        label = label + [0] * diff
    for i in range(len(pred)):
        dcg += (pow(2, pred[i]) - 1) / math.log2(i + 2)
        idcg += (pow(2, label[i]) - 1) / math.log2(i + 2)
    ndcg = dcg / (idcg + 1e-8)
    return dcg, idcg, ndcg

def parse_xgb_dict(xgb_dump_path):
    xgb_tree_path_dict = {};    tree_num = -1
    with open(xgb_dump_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.split('[')[0] == 'booster':
                tree_num += 1
                root = True
                if tree_num not in xgb_tree_path_dict:
                    xgb_tree_path_dict[tree_num] = {'decision_nodes': {}, 'root': -1}
            else:
                node_id = line.strip().split(':')[0]
                if root:
                    xgb_tree_path_dict[tree_num]['root'] = node_id
                    root = False
                arr = line.split('[')
                if len(arr) == 1:  # leaf node
                    leaf_value = line.split('=')[-1]
                    if node_id not in xgb_tree_path_dict[tree_num]['decision_nodes']:
                        xgb_tree_path_dict[tree_num]['decision_nodes'][node_id] = [leaf_value]
                else:   # tree node
                    tmp = arr[1].split(']')
                    fid = tmp[0]
                    feat_id, split_thr = fid.split('<')
                    jump_nodes = tmp[1].strip().split(',')
                    yes_node = jump_nodes[0].split('=')[-1]
                    no_node = jump_nodes[1].split('=')[-1]
                    missing_node = jump_nodes[2].split('=')[-1]
                    if node_id not in xgb_tree_path_dict[tree_num]['decision_nodes']:
                        xgb_tree_path_dict[tree_num]['decision_nodes'][node_id] = [int(feat_id.split('f')[-1]),
                                                                                   split_thr, yes_node, no_node,
                                                                                   missing_node]
        return xgb_tree_path_dict

def predict_proba(xgb_tree_path_dict,input_X):
    features = input_X[0]
    boosting_value = 0.0  # logit value
    hit_feats = []
    path_ids = []
    leaf_enc = []; leaf_value = []
    for tree_num in xgb_tree_path_dict:
        sub_tree_path = []
        sub_hit_nodes = {}
        tree_info = xgb_tree_path_dict[tree_num]
        decision_nodes = tree_info['decision_nodes']
        root_node = tree_info['root']
        cur_decision = decision_nodes[root_node]
        node_id = root_node
        while True:
            if len(cur_decision) == 1: # leaf node
                boosting_value += float(cur_decision[0])
                leaf_enc.append(int(node_id))
                break
            else:
                feat_id = cur_decision[0]
                sub_tree_path.append(feat_id)
                if feat_id not in sub_hit_nodes:
                    sub_hit_nodes[feat_id] = 0
                sub_hit_nodes[feat_id] += 1
                split_thr = float(cur_decision[1])
                yes_node = cur_decision[2]
                no_node = cur_decision[3]
                missing_node = cur_decision[4]
                if features[feat_id] < split_thr:
                    cur_decision = decision_nodes[yes_node] ; node_id = yes_node
                else:
                    cur_decision = decision_nodes[no_node] ; node_id = no_node
        path_ids.append(sub_tree_path)
        hit_feats.append(sub_hit_nodes)
    prob = 1.0 /  ( 1 + math.exp( -1 * boosting_value) )
    return prob

if __name__ == "__main__":
    #filter_ids("get_jdcv_data/jdcvids", "get_jdcv_data/sampleids")
    #cal_ndcg([5,6,3,2,4,1,0], 6)    #[3,2,3,0,1,2,3,0]
    gen_entity_dict()