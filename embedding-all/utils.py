import re, os, logging, codecs, math, copy
from config import conf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

re_ch = re.compile(u"([\u4e00-\u9fa5])",re.S)
re_en = re.compile(u"([a-zA-Z]+|[0-9]+k[\+]*)",re.S)
re_salary = re.compile(u"([0-9]+k[\+]*)",re.S)
re_year = re.compile(u'([0-9]*年)', re.M | re.I)
PUNCTUATION_LIST = ".。,，,、?？:：;；{}[]【】“‘’”《》/!！%……（）()<>@#$~^￥%&*\"\'=+-_——「」"
NONE_STOPWORD = []
CUSTOM_STOPWORD = ["人","年","大","工作","能力","负责","生产","员工","福利","加班","交通","双休","五险","底薪","一金","提成","诚聘","高薪","代表", \
                   "住宿","急聘","包住","无责","月薪","电话","周末","编号","接受","奖金","方向","公司","免费","带薪","包吃","晋升","职位","经验","管理", \
                   "急招","提供","精英","应届","食宿","直招","责任","中心","平台","人员","朝九晚","补贴","高级","社保","诚招","集团","六险","办公室", \
                   "过万","绩效","有限","优先","年底","银行","薪资","餐补","待遇","总部","基础","年薪","轻松","就近","双薪","团队","补助","品牌","校区", \
                   "企业","上市公司","专业","兼职","节日","保底","岗位","店员","无需","分配","熟悉","很多","更好","岗位职责","基于","功能","发布","父母", \
                   "两餐"]
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

def Q2B(uchar):     # 全角转半角
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

def stringQ2B(ustring):     # 把字符串全角转半角
    return "".join([Q2B(uchar) for uchar in ustring])

def uniform(ustring):       # 格式化字符串，完成全角转半角，大写转小写的工作
    return stringQ2B(ustring).lower()

def clean_line(line):
    line = uniform(line)
    line = re.sub("[&$￥～�|＠？＞＝＜；!｜｛＼］［／－＋＊*＆％＃＂！🌐．﹒海金]{1,}|[.#-]{2,}|[+]{3,}|[0-9]*%", "", line)
    line = re.sub("[【】]{1,}", "", line)
    return line

def cal_vec_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def token2list(token):
    data = []
    if re_en.fullmatch(token):
        data.append(token)
    else:
        for e in list(token):
            data.append(e)
    return data

def sigmoid(value):
    prob = 1.0 / (1 + math.exp(-1 * value))
    return prob

def textRank(sc, sims, thr_sim=0.7, sc_pow=0.7):
    assert (len(sc) == len(sims))

    def retainMax(arr):
        m = np.max(arr) * thr_sim
        return [0 if x < m else x for x in arr]

    sc_ = sc
    sims = (sims / np.sum(sims, axis=1, keepdims=True)).astype(np.float)
    sims = np.apply_along_axis(retainMax, 1, sims)
    sc = np.array(sc) / np.sum(sc)

    for _ in range(1000):
        ss = copy.deepcopy(sc)
        for i in range(len(sc)):
            s = 0
            for j in range(len(sc)):
                if i == j: continue
                s += sc[j] * sims[j][i]
            sc[i] = 0.15 + 0.85 * s
        if np.sum(np.abs(sc - ss)) < 0.001: break

        # for i in range(len(sc)):sc[i] *= math.pow(sc_[i], sc_pow)
    print(sc)
    return sc

if __name__ == "__main__":
    a = clean_line("（一）【任职资格】：1、大专及以上学历")
    #filter_ids("get_jdcv_data/jdcvids", "get_jdcv_data/sampleids")
    #cal_ndcg([5,6,3,2,4,1,0], 6)    #[3,2,3,0,1,2,3,0]
    #gen_entity_dict()