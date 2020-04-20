import logging, json, requests, jieba, sys, os, copy, re
from config import conf
from utils import FUNC_DICT, INDUS_DICT, PUNCTUATION_LIST, contain_chinese_word, STOP_WORDS, re_en
from collections import Counter
from tqdm import tqdm
from huqie import Huqie

SPECIAL_WORDS_CUSTOM = ['c++','cocos2d-x','.net','--','node.js','c/s','c#','unity3d','cocos2d','u-boot','u3d','2d','3d','html5','j2ee','o2o', \
                        '外贸销售','h5','as3']
SPECIAL_WORDS_FUNC = list(set([e.lower().strip().split()[0] for e in open(conf.func_file, encoding="utf8").readlines() if not contain_chinese_word(e.strip().split()[0])]))
SPECIAL_WORDS_INDU = list(set([e.lower().strip().split()[0] for e in open(conf.indus_file, encoding="utf8").readlines() if not contain_chinese_word(e.strip().split()[0])]))
SPECIAL_WORDS = SPECIAL_WORDS_CUSTOM + SPECIAL_WORDS_FUNC + SPECIAL_WORDS_INDU
re_salary = re.compile(u"([0-9]+k[\+]*)",re.S)

re_seg = re.compile(u"([，,])", re.S)
def regular_cut(text):
    return re_seg.split(text)
a=re_seg.split("董英姿,前端")

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

def en_split(text_origin):
    text = copy.deepcopy(text_origin).lower()
    res, find_words = [], []
    for w in SPECIAL_WORDS:
        index = text_origin.find(w)
        if index < 0: continue
        if not check_contain((w, index, index + len(w)), find_words):
            text = text.replace(w, ' '+w+' ')
            find_words.append((w, index, index + len(w)))
    seg_text = text.strip().split()
    for w in seg_text:
        if w in SPECIAL_WORDS:
            res.append(w)
        else:
            for e in re_en.split(w):
                if e in ['', ' ']: continue
                res.append(e)
    return res

a=en_split("as3")

def valid_idf(token):
    if token.isdigit() and len(token) == 1: return False    # 单个数字的idf无效
    return True

ENTITY_DICT = [line.strip() for line in open(conf.new_entity_file, encoding="utf8").readlines()]
def select_important_tokens(text):
    seg_text = cut(text)
    res = [e for e in seg_text if is_valid_tokens(e)]
    return list(set(res))

class Tokenizer():
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
#        self.vocab = json.load(open(conf.vocab, encoding="utf8"))
#        self.idf = {k: v for k, v in json.load(open(conf.idf, encoding="utf8")).items() if valid_idf(k)}
#        self.id2word = {v: k for k, v in self.vocab.items()}
        #self.model.set_dictionary(conf.corp_file)
        for w in ENTITY_DICT:
            self.model.add_word(w, freq=1000000)
        if FUNC_DICT:
            for w, f in FUNC_DICT.items():
                self.model.add_word(w, freq=f)
        if INDUS_DICT:
            for w, f in INDUS_DICT.items():
                self.model.add_word(w, freq=f)
        #self.model.set_dictionary(conf.indus_file)

    def cut(self, text):
        #a0, a1=list(self.model.tokenize(text)),list(self.model.cut(text))
        cut_res, index_words = [], {}
        try:
            cut_words = list(self.model.cut_for_search(text))
            for e in cut_words:
                if len(cut_words) !=1 and e == text and e not in PUNCTUATION_LIST: continue
                word_index = text.find(e)
                if word_index not in index_words: index_words[word_index] = []
                index_words[word_index].append(e)
            sorted_word_index = sorted(index_words.items(), key=lambda d: d[0], reverse=False)
            for index, word_list in sorted_word_index:
                if len(word_list) == 1: cur_word = word_list[0]
                else: cur_word = sorted(word_list, key=lambda d: len(d[1]))[-1]
                if index == 0:
                    cut_res.append((cur_word, 0, len(cur_word)))
                    continue
                if set(list(cur_word)).intersection(set(list(cut_res[- 1][0]))): continue
                cut_res.append((cur_word, index, index + len(cur_word)))
        except:
            cut_res = list(self.model.tokenize(text))
        if ''.join([e[0] for e in cut_res]) != text: cut_res = list(self.model.tokenize(text))
        token_list = [e[0] for e in cut_res]
        return token_list, cut_res

    def custom_cut(self, text):
        '''
        res = []; #a=list(self.model.tokenize(text))
        for words, start_index, end_index in self.model.tokenize(text):
            for e in self.cut(words):
                res.append((e[0], e[1] + start_index, e[2] + start_index))
        return res
        '''
        return list(self.model.tokenize(text))

    def tokenize(self, sentence):
        senten2term, word_seg, word_index = [], [],  0      ; #a=en_split(sentence)
        for word in en_split(sentence):
            word = word.strip().lower()
            if word in ['', ' ']: continue
            if re_en.fullmatch(word):   # 英文处理
                if re_salary.fullmatch(word): rword = word            # 薪水term处理
                else: rword = word                                  # 不处理
                word_seg.append((rword, word_index, word_index+len(rword)))
                word_index +=len(rword)
            else:                       # 非英文处理
                if word in SPECIAL_WORDS:
                    model_seg = [(word, 0, len(word))]
                else:
                    model_seg = self.custom_cut(word) #list(self.model.tokenize(word))
                word_seg.extend([(e[0], e[1]+word_index, e[2]+word_index) for e in model_seg])
                word_index = word_seg[-1][2]
        senten2term = [e[0] for e in word_seg]
        return senten2term, word_seg

    def encode_ids(self, sentence):
        senten2term, word_seg = self.tokenize(sentence)
        tokens = ["▁"] + senten2term
        ids = [self.vocab.get(e, 0) for e in tokens]         # ["▁"]
        return tokens, ids

    def IdToPiece(self, Id):
        return self.id2word.get(Id, "<unk>")

    def id_to_piece(self, Id):
        return self.id2word.get(Id, "<unk>")

    def cal_weight_idf(self, senten2term):
        token_weight = [(w, self.idf.get(w, 0.0)) for w in senten2term]
        return token_weight

    def select_important_tokens(self, text):
        #tokens, _ = self.cut(text)
        tokens = list(self.model.cut(text))
        res = [e for e in tokens if is_valid_tokens(e)]
        return list(set(res))

def is_valid_tokens(word):
        '''
        if word in FUNC_DICT or word in INDUS_DICT or word in SPECIAL_WORDS:
            if word not in STOP_WORDS:
                if len(word) == 1 and word in ['c']: return True
                else: return True
        '''
        if word in ENTITY_DICT and word not in STOP_WORDS: return True
        return False

def gen_entity_dict():
    token = Tokenizer()
    '''
    title_freq = Counter([line.split("\t")[0] for line in open("data/jdtitledesc", encoding="utf8").readlines()])
    top_title_freq = title_freq.most_common()
    with open("data/total_title", "w", encoding="utf8") as fin:
        for t, f in top_title_freq:
            fin.write(t + "\n")
    '''
    title_words = []
    text = [line.strip() for line in open("data/total_title", encoding="utf8").readlines()]
    for line in tqdm(text, total=len(text)):
        title_words.extend(token.cut(line)[0])
    title_freq = Counter(title_words).most_common()
    with open("data/title_entitys", "w", encoding="utf8") as fin:
        for t, f in title_freq:
            fin.write(t + "\n")
    exit()

if __name__ == '__main__':
    #gen_entity_dict()
    try: que = sys.argv[1]
    except: que = "古c熟悉java各种开发软件以及很多电子.net商务知识" #"advc#montage+深圳c++c/s5k"  新加坡航空公司
    qie = Huqie()
    a = qie.qie(que).split()
    #nlu_seg = nlu_cut(que)
    #jieba_seg = jieba_cut("分布式文件系统")
    #a0=list(jieba.cut_for_search(que)); a1=list(jieba.tokenize(que)); a2=list(jieba.cut(que))
    #print(json.dumps(cut(que), ensure_ascii=False))   # 分词服务
    t = Tokenizer()
    #tokens, _ = t.cut(que)
    #a = t.select_important_tokens(que)
    aa = select_important_tokens(que)
    pass
