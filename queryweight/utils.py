import re, json, math, os, logging, codecs, jieba
from collections import defaultdict
from tqdm import tqdm
from config import conf

re_ch = re.compile(u"([\u4e00-\u9fa5])",re.S)
PUNCTUATION_LIST = ".。,，,、?？:：;；{}[]【】“‘’”《》/!！%……（）<>@#$~^￥%&*\"\'=+-_——「」"
NONE_STOPWORD = ["it"]
CUSTOM_STOPWORD = ["人","年","大"]
STOP_WORDS = [e.strip() for e in open(conf.stop_words, encoding="utf8").readlines() if e.strip() not in NONE_STOPWORD] + CUSTOM_STOPWORD

def contain_chinese_word(sentence):
    if re_ch.findall(sentence): return True
    return False

def gen_entity_dict():
    for file_name in ['__func__.txt', '__ind__.txt']:
        res = []
        text = open('dict/' + file_name, encoding='utf8').readlines()
        sub_word = ""
        for i, line in enumerate(text):
            cur_word = line.strip().replace(" ", "")
            if sub_word and contain_chinese_word(cur_word) and sub_word in cur_word and cur_word.index(sub_word) == 0:
                pass
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
    TermType = term_type(word_index, sen2terms)             # 当前term的类型：[中文，英文，数字，符号，停用词，其它]
    TermLength = [len(list(word))]                          # 当前term的字符级别长度
    TermOffset = [round(word_index / len(sen2terms), 3)]              # 当前term的偏移百分比
    PreTermPosition = term_position(word_index - 1, sen2terms)    # 前一个term的位置：[头部，尾部，其它]
    PreTermType = term_type(word_index - 1, sen2terms)        # 前一个term的类型：[中文，英文，数字，符号，停用词，其它]
    BehindTermPosition = term_position(word_index + 1, sen2terms)  # 后一个term的位置：[头部，尾部，其它]
    BehindTermType = term_type(word_index + 1, sen2terms)  # 后一个term的类型：[中文，英文，数字，符号，停用词，其它]

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
        ]
    feature_vector = []; fidindex = 0; fmap = []
    for fid in features:
        feature_vector.extend(fid[1])
        for i in range(len(fid[1])):
            fmap.append("\t".join([str(fidindex), fid[2]+":"+str(i)]))
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
    dcg, idcg = 1e-8, 1e-8
    pred = label_list[:topk]
    label = sorted(label_list, key=lambda d: d, reverse=True)[: topk]
    diff = topk - len(label_list)
    if diff > 0:        # 分数补全
        pred = pred + [0] * diff
        label = label + [0] * diff
    for i in range(len(pred)):
        dcg += (pow(2, pred[i]) - 1) / math.log2(i + 2)
        idcg += (pow(2, label[i]) - 1) / math.log2(i + 2)
    ndcg = dcg / idcg
    return dcg, idcg, ndcg

if __name__ == "__main__":
    #filter_ids("get_jdcv_data/jdcvids", "get_jdcv_data/sampleids")
    #cal_ndcg([5,6,3,2,4,1,0], 6)    #[3,2,3,0,1,2,3,0]
    gen_entity_dict()