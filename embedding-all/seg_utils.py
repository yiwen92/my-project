import logging, json, requests, jieba, sys, os, copy, re
from config import conf
from utils import FUNC_DICT, INDUS_DICT, PUNCTUATION_LIST, contain_chinese_word, STOP_WORDS, re_en
from huqie import Huqie

ENTITY_DICT = [line.strip() for line in open(conf.new_entity_file, encoding="utf8").readlines()]
SPECIAL_WORDS_CUSTOM = ['c++','cocos2d-x','.net','--','node.js','c/s','c#','unity3d','cocos2d','u-boot','u3d','2d','3d','html5','j2ee','o2o', \
                        '外贸销售','h5','as3','matlab']
SPECIAL_WORDS_FUNC = list(set([e.lower().strip().split()[0] for e in open(conf.func_file, encoding="utf8").readlines() if not contain_chinese_word(e.strip().split()[0])]))
SPECIAL_WORDS_INDU = list(set([e.lower().strip().split()[0] for e in open(conf.indus_file, encoding="utf8").readlines() if not contain_chinese_word(e.strip().split()[0])]))
SPECIAL_WORDS = SPECIAL_WORDS_CUSTOM
re_salary = re.compile(u"([0-9]+k[\+]*)",re.S)

def cut(text):
    res = []
    try:
        url = 'http://192.168.12.18:51990/huqie'
        body = {"txt":str(text)}
        query = json.dumps(body)
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        response = requests.post(url, data=query, headers=headers)
        response = json.loads(response.text)
        res = response['1']['txt']
    except Exception as e:
        logging.warn('getSegTxt_error=%s' % (str(repr(e))))
    return res

def jieba_cut(text):
    res = list(jieba.cut(text))
    return res

def nlu_cut(text):
    res = nlu.cut(text)
    return res

def check_contain(word, find_words):
    for ele in find_words:
        if word[0] in ele[0] and word[1] >= ele[1] and word[2] <= ele[2]:
            return True
    return False

def dict_split(text_origin):
    text = copy.deepcopy(text_origin).lower()
    find_words = []
    for w in SPECIAL_WORDS:
        index = text_origin.find(w)
        if index < 0: continue
        if not check_contain((w, index, index + len(w)), find_words):
            text = text.replace(w, ' '+w+' ')
            find_words.append((w, index, index + len(w)))
    return text

def select_important_tokens(text):
    seg_text = cut(text)
    res = [e for e in seg_text if is_valid_tokens(e)]
    return list(set(res))

class Tokenizer():
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
        for w in ENTITY_DICT: self.model.add_word(w, freq=1000000)
        if FUNC_DICT:
            for w, f in FUNC_DICT.items():
                self.model.add_word(w, freq=f)
        if INDUS_DICT:
            for w, f in INDUS_DICT.items():
                self.model.add_word(w, freq=f)
        #self.model.set_dictionary(conf.indus_file)

    def tokenize(self, sentence):
        re_seg_sentence = dict_split(sentence)        #; a = list(self.model.tokenize(sentence))
        seg_sentence = list(self.model.tokenize(re_seg_sentence))
        res = [e[0] for e in seg_sentence if e[0] not in ['', ' ']]
        return res

    def select_important_tokens(self, text):
        tokens = list(self.tokenize(text))
        res = [e for e in tokens if is_valid_tokens(e)]
        return list(set(res))

def is_valid_tokens(word):
    if word in ENTITY_DICT and word not in STOP_WORDS: return True
    return False

if __name__ == '__main__':
    #gen_entity_dict()
    try: que = sys.argv[1]
    except: que = "古c熟悉java各种开发软件以及很多电子.net商务知识，熟悉tensorflow、matlab软件" #"advc#montage+深圳c++c/s5k"
    qie = Huqie()
    a = qie.qie(que).split()
    #nlu_seg = nlu_cut(que)
    #jieba_seg = jieba_cut("分布式文件系统")
    #a0=list(jieba.cut_for_search(que)); a1=list(jieba.tokenize(que)); a2=list(jieba.cut(que))
    #print(json.dumps(cut(que), ensure_ascii=False))   # 分词服务
    t = Tokenizer()
    s = t.tokenize(que)
    aa = select_important_tokens(que)
    pass
