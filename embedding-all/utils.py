import re, json, math, os, logging, codecs, jieba
from collections import defaultdict
from tqdm import tqdm
from config import conf

re_ch = re.compile(u"([\u4e00-\u9fa5])",re.S)
re_en = re.compile(u"([a-zA-Z]+|[0-9]+k[\+]*)",re.S)
re_year = re.compile(u'([0-9]*年)', re.M | re.I)
PUNCTUATION_LIST = ".。,，,、?？:：;；{}[]【】“‘’”《》/!！%……（）<>@#$~^￥%&*\"\'=+-_——「」"
NONE_STOPWORD = []
CUSTOM_STOPWORD = ["人","年","大","工作","能力","负责","负责","生产"]
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

if __name__ == "__main__":
    a = clean_line("（一）【任职资格】：1、大专及以上学历")
    #filter_ids("get_jdcv_data/jdcvids", "get_jdcv_data/sampleids")
    #cal_ndcg([5,6,3,2,4,1,0], 6)    #[3,2,3,0,1,2,3,0]
    #gen_entity_dict()