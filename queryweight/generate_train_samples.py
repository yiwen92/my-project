#!/usr/bin/env

import json, traceback, re, math
#from jdapi import JDFromGearman
#from cvapi import get_cv_dicts
from tqdm import tqdm
from seg_utils import Tokenizer, PUNCTUATION_LIST
from collections import Counter, defaultdict
from qw import query_weight
from utils import get_feature, STOP_WORDS

#jd_obj=JDFromGearman("icdc_position_basic_online", ["192.168.8.70:4730"])

def getter_info(base_dict, base_key, default=None):
    if base_key in base_dict:
        ret = base_dict[base_key]
        return ret
    else:
        return default

def get_jd_info(jd_id):
    res = {}
    try:
        jd_dict = jd_obj.getJson(jd_id)
        name = getter_info(jd_dict, 'name', '') ;    #print('jd_dict\n', json.dumps(jd_dict, ensure_ascii=False), '\nname: ', name); #exit()
        requirement = getter_info(jd_dict, 'requirement', '')
        description = getter_info(jd_dict, 'description', '')
        res['name'], res['requirement'], res['description'] = name, requirement, description
    except Exception as e:
        print('get_jd_info: traceback=%s' % str(traceback.format_exc()))
    return res

def get_cv_info(cv_dict, sep="|-|"):
    res, tmp = "", []
    try:
        #print('cv_dict\n', json.dumps(cv_dict, ensure_ascii=False)); exit()
        work = getter_info(cv_dict, 'work', {})
        if not work: return res
        for k, v in work.items():
            responsibilities = getter_info(v, 'responsibilities', '')
            #res['work' + '_' + 'responsibilities' + '_' + k] = responsibilities
            if responsibilities in ['']: continue
            tmp.append(responsibilities)
        res = sep.join(tmp)
        #print(json.dumps(res, ensure_ascii=False)); exit()
    except Exception as e:
        print('get_cv_info: traceback=%s\tcv_id=%s' % (str(traceback.format_exc()), cv_id)); exit()
    return res

def get_cv_infos(cv_id_list):
    res = []
    cv_dicts = get_cv_dicts(cv_id_list)
    for cv_id, cv_dict in cv_dicts.items():
        cv_info = get_cv_info(cv_dict)
        if cv_info:
            res.append(json.dumps(cv_info, ensure_ascii=False))
    return res

def sample_jdcvid(path):
    """
    user_id: 用户ID, jd_id: 职位ID, resume_id: 简历ID
    status: 状态: 1-投递成功, 2-被查看 ,3-筛选通过, 4-面试通知, 5-简历不合适, 6-简历被转发, 99-取消应聘
    type: 0-未知, 1-主动投递, 2-高意向, 3-逸橙推荐, 4-内推
    """
    print("sample jd-cv id pairs")
    tmp = {}    ; status_static, type_static, len_static = defaultdict(int), defaultdict(int), defaultdict(int)
    total_num = len(open(path, encoding="utf8").readlines())
    for i, line in enumerate(tqdm(open(path, encoding="utf8"), total=total_num)):
        if i == 0: continue
        user_id, jd_id, resume_id, status, type = line.strip().split("\t")
        status_static[status] += 1; type_static[type] += 1
        if status not in ['3', '4'] or type not in ['1', '2', '3', '4']: continue
        if jd_id not in tmp: tmp[jd_id] = []
        tmp[jd_id].append(resume_id)
    jd_cvs = {k: v for k, v in tmp.items() if len(v) > 20}
    for k, v in tmp.items(): len_static[len(v)] += 1
    print("jd-cvs pairs: %d" % (len(jd_cvs)))
    print("status_static: %s\ttype_static: %s\tcv_len_static: %s" % (json.dumps(status_static), json.dumps(type_static), json.dumps(len_static)))
    return jd_cvs

def gen_text_pairs(jdcvidpath, textpath):
    jd_cv_ids = sample_jdcvid(jdcvidpath)
    print("generate jd-cv text pairs")
    res = []
    for jd_id, cv_ids in tqdm(jd_cv_ids.items(), total=len(jd_cv_ids)):
        line_info = {}
        jd_info = get_jd_info(jd_id)
        line_info['jd'] = json.dumps(jd_info, ensure_ascii=False)
        cv_infos = get_cv_infos(cv_ids)    #; print(json.dumps(cv_info, ensure_ascii=False, indent=2)); exit()
        line_info['cv'] = json.dumps(cv_infos, ensure_ascii=False)
        res.append(json.dumps(line_info, ensure_ascii=False))
        #print(json.dumps(line_info, ensure_ascii=False)); exit()
    print("writing text file %s" % (textpath))
    with open(textpath, "w", encoding="utf8") as fin:
        fin.write("\n".join(res))

def label_data(path, out_path):
    print("generage query weighting label data")
    t = Tokenizer() ; res = []
    total_num = len(open(path, encoding="utf8").readlines())
    for i, line in enumerate(tqdm(open(path, encoding="utf8"), total=total_num)):
        line_info = json.loads(line)
        cv_info, jd_info = json.loads(line_info['cv']), json.loads(line_info['jd'])
        senten2term, word_seg = t.tokenize(jd_info['name'])
        weight_cv = cv_weight(cv_info, senten2term, t)
        weight_jd = jd_weight(jd_info, senten2term, t)
        tmp = "\t".join([(weight_cv[i][0] + ":" + str(round(0.6 * weight_jd[i][1] + 0.4 * weight_cv[i][1], 3))) for i in range(len(weight_cv))]) + "\n"
        res.append(tmp)
    print("writing label data %s" % (out_path))
    with open(out_path, "w", encoding="utf8") as fin:
        fin.write("".join(res))

def jd_weight(jd_info, senten2term, t):
    tmp, weight_sum = [], 1e-8
    word_idf = t.cal_weight_idf(senten2term)
    for i in range(len(senten2term)):
        term, val = senten2term[i], 0.0
        if jd_info['requirement']:
            requirement_seg, _ = t.tokenize(jd_info['requirement'])
            requirement_word_freq = Counter(requirement_seg)
            val += 0.6 * requirement_word_freq.get(term, 0) * word_idf[i][1]
        if jd_info['description']:
            description_seg, _ = t.tokenize(jd_info['description'])
            description_word_freq = Counter(description_seg)
            val += 0.6 * description_word_freq.get(term, 0) * word_idf[i][1]
        tmp.append((term, val))
        weight_sum += val
    token_weight = [(k, round(v / weight_sum, 3)) for k, v in tmp]
    return token_weight

def cv_weight(cv_info, senten2term, t):
    word_tf, word_idf, word_tfidf, sum_tf_idf = defaultdict(int), defaultdict(int), [], 1e-8
    for cv in cv_info:
        cv = re.sub(u"[ ]{1,}", "", cv)
        cv = cv.replace("\\n", "")
        cv = cv.replace("|-|", "")
        cv = cv.rstrip("\"")
        cv = cv.lstrip("\"")
        cv = cv.replace("工作描述：", "")
        cv_seg, _ = t.tokenize(cv)
        cv_word_freq = Counter(cv_seg)
        for term in senten2term:
            if term in PUNCTUATION_LIST or term in STOP_WORDS: continue
            if term in cv_word_freq:
                word_tf[term] += cv_word_freq[term]
                word_idf[term] += 1
    for e in senten2term:
        tf_idf = word_tf.get(e, 0) * math.log10(len(cv_info) / (word_idf.get(e, 0) + 1) + 1)
        word_tfidf.append((e, tf_idf))
        sum_tf_idf += tf_idf
    word_tfidf = [(k, round(v / sum_tf_idf, 3)) for k, v in word_tfidf]
    return word_tfidf

def gen_train_data(in_path, out_path):
    matchObj = re.compile(r'(.+):([0-9\.]+)', re.M | re.I)
    res, qid = [], 0
    qw = query_weight(1000000)
    text = [e.strip().split("\t") for e in open(in_path, encoding="utf8").readlines() if e.strip()]
    for i, ele in enumerate(tqdm(text, total=len(text))):
        line_ele, sen2terms = [], []
        for e in ele:
            matchRes = matchObj.match(e)
            term, weight = matchRes.group(1), matchRes.group(2)
            line_ele.append((term, weight))
            sen2terms.append(term)
        qw.run_step(" ".join(sen2terms))
        weight_attn, weight_idf, weight_lm = qw.weight_attn, qw.weight_idf, qw.weight_lm
        sorted_line_ele = sorted(line_ele, key=lambda d: d[1], reverse=True)
        for i in range(len(sorted_line_ele)):
            feature_vector, fmap = get_feature(sorted_line_ele[i][0], sen2terms, weight_attn, weight_idf, weight_lm)
            res.append(" ".join([str(len(sorted_line_ele) - i - 1), "qid:" + str(qid)] + [str(i+1) + ":" + str(e) for i, e in enumerate(feature_vector)]))
        qid += 1
    print("train data length: %d" % (len(res)))
    with open(out_path + "train.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[:int(len(res) * 0.9)]))
    with open(out_path + "test.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.9):]))
    with open(out_path + "valid.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.9):]))
    with open(out_path + "feature.fmap", "w", encoding="utf8") as fin:
        fin.write("\n".join(fmap))

if __name__ == '__main__':
    #   /basic_data/tob/operation/search_data_jc
    suf = '3000000'
    #jdcvidpath = 'get_jdcv_data/jdcvids_'+suf    ;   text_data_path = 'get_jdcv_data/jdcvtext_'+suf ;   label_data_path = "get_jdcv_data/label.data_"+suf
    jdcvidpath = 'get_jdcv_data/sampleids'; text_data_path = 'get_jdcv_data/jdcvtext'; label_data_path = "get_jdcv_data/label.data"
    feature_path = 'get_jdcv_data/'
    #print(json.dumps(get_jd_info(163), ensure_ascii=False, indent=2))
    #print(json.dumps(get_cv_infos([11375431, 10430437]), ensure_ascii=False, indent=2))
    #sample_jdcvid(jdcvidpath)       # 产生 jd-cvs id列表
    #gen_text_pairs(jdcvidpath, text_data_path)      # 得到id对应的文本数据
    #label_data(text_data_path, label_data_path)     # 对query标注权重
    gen_train_data(label_data_path, feature_path)
    pass
