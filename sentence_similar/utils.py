import io, cloudpickle, re
from future.utils import PY2

def pycloud_pickle(file_name, obj):
    # type: (Text, Any) -> None
    """Pickle an object to a file using cloudpickle."""
    with io.open(file_name, 'wb') as f:
        cloudpickle.dump(obj, f)

def pycloud_unpickle(file_name):
    # type: (Text) -> Any
    """Unpickle an object from file using cloudpickle."""

    with io.open(file_name, 'rb') as f:  # pragma: no test
        if PY2:
            return cloudpickle.load(f)
        else:
            return cloudpickle.load(f, encoding="latin-1")

re_ch = re.compile(u"([\u4e00-\u9fa5])", re.S)
patt_hard = [',', '，', ':', '：', '!', '！', '?', '？', '(', '（', ')', '）', '*', '%', '@', '$', '\\', '/', ';', '；', '、',
             '&', '&&', '。', '[', ']', '【', '】', '《', '》', '“', '”', '{', '}', ';']
patt_hard = ['\\' + x for x in patt_hard]
patt_soft = ['-', '+', ' #']  # ,'.']
patt_soft = ['\\' + x for x in patt_soft]
re_hard = re.compile('(' + '|'.join(patt_hard) + ')', re.S)
re_soft = re.compile('(' + '|'.join(patt_soft) + ')', re.S)
user_patt_word = dict([(w, '1') for w in
                       ['tcp/ip', 'b/s', 'div+css', 'jquery+css', 'c++', 'cocos2d-x', '.net', '--', 'node.js', 'c/s',
                        'c#']])

def char_cut(sent):
    sent = str(sent)
    # hard pattern
#    sp = re_ch.sub(" \g<1> ", sent.decode('utf8'))         # python2
    sp = re_ch.sub(" \g<1> ", sent)         # python3
    # we filter by user_patt_word
#    sp = sp.encode('utf8')         # python2
    sp = re_hard.sub(" \g<1> ", sp).split()         # python3
    # now we should deal with cases like '-' ,'+'
    final_words = []
    for word in sp:
        # if word.strip() == '': continue
        word = word.strip().lower()
        if word is not None:
            if word in user_patt_word:
                final_words.append(word)
            else:
                # sp_word = re.sub('(' + '|'.join(patt_soft) +')'," \g<1> ",word)   # python2
                sp_word = re_soft.sub(" \g<1> ", word)          # python3

#                final_words.append(sp_word.encode('utf8'))          # python2
                final_words.append(sp_word)         # python3
    return final_words

def tokenize(text):
    # type: (Text) -> List[Token]
    import jieba

    tokenized = jieba.tokenize(text)
    tokens = [(word, start, end) for (word, start, end) in tokenized]
    return tokens

if __name__ == '__main__':
    a = tokenize('熟悉java开发')
    print(char_cut('熟悉java开发'))