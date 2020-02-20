import logging, json, requests, jieba, sys, os, copy, re
from config import conf

PUNCTUATION_LIST = ".。,，,、?？:：;；{}[]【】“‘’”《》/!！%……（）<>@#$~^￥%&*\"\'=+-_——「」"
SPECIAL_WORDS = ['c++','cocos2d-x','.net','--','node.js','c/s','c#','unity3d','2d','3d','cocos2d']
STOP_WORDS = [e.strip() for e in open(conf.stop_words, encoding="utf8").readlines()]
re_en = re.compile(u"([a-zA-Z]+|[0-9]+k[\+]*)",re.S)
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

def en_split(text):
    text = text.lower()
    res = []
    for w in SPECIAL_WORDS:
        if text.find(w) < 0: continue
        text = text.replace(w, ' '+w+' ')
    seg_text = text.strip().split()
    for w in seg_text:
        if w in SPECIAL_WORDS:
            res.append(w)
        else:
            for e in re_en.split(w):
                if e in ['', ' ']: continue
                res.append(e)
    return res

class Tokenizer():
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
        self.vocab = json.load(open(conf.vocab, encoding="utf8"))
        self.idf = json.load(open(conf.idf, encoding="utf8"))
        self.id2word = {v: k for k, v in self.vocab.items()}

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
        return cut_res

    def custom_cut(self, text):
        res = []; #a=list(self.model.tokenize(text))
        for words, start_index, end_index in self.model.tokenize(text):
            for e in self.cut(words):
                res.append((e[0], e[1] + start_index, e[2] + start_index))
        return res

    def tokenize(self, sentence):
        senten2term, word_seg, word_index = [], [],  0
        #a=en_split(sentence)
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

if __name__ == '__main__':
    try: que = sys.argv[1]
    except: que = "软件工程师武汉3年本科.net" #"advc#montage+深圳c++c/s5k"  新加坡航空公司
    #nlu_seg = nlu_cut(que)
    #jieba_seg = jieba_cut("分布式文件系统")
    a0=list(jieba.cut_for_search(que)); a1=list(jieba.tokenize(que)); a2=list(jieba.cut(que))
    #print(json.dumps(cut(que), ensure_ascii=False))   # 分词服务
    t = Tokenizer(); #a3=t.custom_cut(que); a = t.tokenize(que)
    tokens, ids = t.encode_ids(que)
    t.cal_weight_idf(tokens[1:])
    pass

