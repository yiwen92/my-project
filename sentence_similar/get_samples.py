#!/usr/bin/env
#coding:utf8

import re, json, random, sys, jieba
from utils import char_cut

re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)                     # 匹配 CDATA
re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)     # Script
re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)        # tyle
re_br=re.compile('<br\s*?/?>')                                              # 处理换行
re_h=re.compile('</?\w+[^>]*>')                                             # ML标签
re_comment=re.compile('<!--[^>]*-->')                                       # HTML注释
blank_line=re.compile('\n+')                                                 # 去掉多余的空行

def filterHtmlTag(htmlstr):
    s = re_cdata.sub('', htmlstr)  # 去掉 CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    s = blank_line.sub('\n', s) #去掉多余的空行
    return s

p = re.compile(r"<p>(.+?)</p>")
l = re.compile(r"[\"\d+\",\"(.+)\"]")

capacity_txt = [e.strip().split(',') for e in open('./data/capacity_point.csv', 'r', encoding='utf8').readlines()[1:]]
capacity_dict = {}
for e1 in capacity_txt:
    for e in e1:
        if e == '0': continue
        capacity_dict[e] = 0
ca = set(capacity_dict.keys())
intents = set(json.loads(open('./data/intents.txt', 'r', encoding='utf8').readlines()[0]))
diff = intents ^ ca

def min_edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    if m == 0: return n
    if n == 0: return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化dp和边界
    for i in range(1, m + 1): dp[i][0] = i
    for j in range(1, n + 1): dp[0][j] = j
    for i in range(1, m + 1):  # 计算dp
        for j in range(1, n + 1):
            a=word1[i - 1];b=word2[j - 1]
            if word1[i - 1] == word2[j - 1]:
                d = 0
            else:
                d = 1
            dp[i][j] = min(dp[i - 1][j - 1] + d, dp[i][j - 1] + 1, dp[i - 1][j] + 1)
    return dp[m][n]

def words_sim(word1, word2):
    w1 = char_cut(word1); w2 = set(char_cut(word2))
    intersection = set(w1).intersection(w2)
    union = set(w1).union(set(w2))
    if len(intersection) == 0:
        return None
    dice_dist = 2 * len(intersection) / len(union)
    #edit_distance = min_edit_distance(word1, word2)
    return dice_dist #/ (edit_distance + 1e-8)

def get_sample(src_file, des_file):
    qes2label = {}; label2id = {}; index = 0; label2qes = {}; label_cnt = {}; qesset = set(); idcnt = 1; threshold = 0.5
    txt = open(src_file, 'r', encoding='utf8').readlines()
    for e in txt:
        sys.stdout.write('Handle progress: ' + str(idcnt) + ' /  ' + str(len(txt)) + '\n'); sys.stdout.flush(); idcnt += 1
        split_text = e.split('\t')
        question = filterHtmlTag(split_text[2])
        labels = json.loads((split_text[4]))
        if len(labels) == 0: continue
        if question in qesset: continue
        sim_dict = {}
        for e in labels:
            max_sim = 0; sim_word = ''
            for k, v in capacity_dict.items():
                dist = words_sim(e[1], k)
                if dist and dist > max_sim:
                    max_sim = dist
                    sim_word = k
            if max_sim < threshold: continue
            if sim_word not in sim_dict: sim_dict[sim_word] = 0
            sim_dict[sim_word] += max_sim
        sorted_sim_dict = sorted(sim_dict.items(), key=lambda d:d[1], reverse=True)
        if sorted_sim_dict:
            label = sorted_sim_dict[0][0]
        else:
            continue
        if question not in qes2label:
            qes2label[question] = []
            qesset.add(question)

        if label not in label2id:
            label2id[label] = index
            index += 1
        if label not in label_cnt: label_cnt[label] = 0
        label_cnt[label] += 1
        qes2label[question].append(label)
        if label not in label2qes:
            label2qes[label] = []
        label2qes[label].append(question)
    sorted_label_cnt = sorted(label_cnt.items(), key=lambda d:d[1], reverse=True)

    for k, v in capacity_dict.items():
        if k in label_cnt:
            capacity_dict[k] = label_cnt[k]

    label_num = 0; sample_num = 0
    with open(des_file, 'w', encoding='utf8') as f:
        for k, v in label2qes.items():
            if k not in capacity_dict and len(v) < 1000: continue
            f.write('## intent:' + k + '\n'); label_num += 1
            v = list(set(v))
            for ele in v:
                #f.write('- ' + ' '.join(char_cut(ele)) + '\n')
                f.write('- ' + ele + '\n')
                sample_num += 1
            f.write('\n')
    print('label_num = %d, sample_num = %d' % (label_num, sample_num))
    a=1

def get_fasttext_sample(src_file, des_file):
    label_set = set(); qes_set = set()
    txt = open(src_file, 'r', encoding='utf8').readlines()
    with open(des_file, 'w', encoding='utf8') as f:
        for e in txt:
            split_text = e.split('\t')
            question = filterHtmlTag(split_text[2])
            labels = json.loads((split_text[4]))
            if len(labels) == 0 or question.strip() == '': continue
            qes_set.add(question)
            line_sample = []
            for e in labels:
                line_sample.append('__label__' + e[1]); label_set.add(e[1])
            for e in char_cut(question):
                line_sample.append(e)
            f.write(' '.join(line_sample) + '\n')
    print("label number: {}, question number: {}".format(len(label_set), len(qes_set)))
    b=1

def get_ft_data(src_file, train_file, test_file, val_file):
    res = []
    item_regex = re.compile(r'\s*[-\*+]\s*(.+)')
    txt = open(src_file, 'r', encoding='utf8').readlines()
    for line in txt:
        if '## intent:' in line:
            label = line.strip().split(':')[-1]
        else:
            match = re.match(item_regex, line)
            if match:
                item = match.group(1)
                #seg_item = ' '.join(list(jieba.cut(item)))
                seg_item = item #' '.join(char_cut(item))
                #res.append('__label__' + label + ' ' + seg_item + '\n')
                res.append(label + '\t' + seg_item + '\n')
    random.shuffle(res)
    with open(train_file, 'w', encoding='utf8') as f1:
        for e in res[:int(len(res) * 0.6)]: f1.write(e)
    with open(test_file, 'w', encoding='utf8') as f2:
        for e in res[int(len(res) * 0.6):int(len(res) * 0.8)]: f2.write(e)
    with open(val_file, 'w', encoding='utf8') as f3:
        for e in res[int(len(res) * 0.8):]: f3.write(e)

def get_sample_new(src_file, train_file, test_file, val_file):
    qes2label = {}; label2id = {}; index = 0; label2qes = {}; label_cnt = {}; qesset = set(); idcnt = 1; threshold = 0.5;  res = []
    txt = open(src_file, 'r', encoding='utf8').readlines()
    re_patten = [('活动策划', re.compile(u"活动|策划")), ('视频识别', re.compile(u"视频识别")), ('项目管理', re.compile(u"项目|管理")),
                 ('图像算法', re.compile(u"图像算法")), ('视频算法', re.compile(u"视频算法")), ('入职准备', re.compile(u"入职|入职准备")),
                 ('视频流转码', re.compile(u"视频流|转码")), ('用户运营', re.compile(u"用户|运营")), ('数据挖掘', re.compile(u"数据挖掘|挖掘")),
                 ('用户研究', re.compile(u"用户研究")), ('数据库索引', re.compile(u"数据库|索引")), ('社交', re.compile(u"社交")),
                 ('音频编解码', re.compile(u"音频|编解码")), ('数据分析', re.compile(u"数据|分析")), ('流媒体封装', re.compile(u"流媒体|封装")),
                 ('图像识别', re.compile(u"图像识别")), ('游戏', re.compile(u"游戏")), ('计算广告', re.compile(u"计算广告")),
                 ('高并发', re.compile(u"高并发|并发")), ('面试辅导', re.compile(u"面试|辅导")), ('技术', re.compile(u"技术")),
                 ('手机游戏', re.compile(u"手机|游戏")), ('需求评估', re.compile(u"需求评估")), ('全栈', re.compile(u"全栈")),
                 ('游戏制作人', re.compile(u"游戏制作人|制作人")), ('创意创新', re.compile(u"创意|创新")), ('协调能力', re.compile(u"协调能力|协调")),
                 ('数据运营', re.compile(u"数据运营")), ('排版美工', re.compile(u"排版|美工")), ('SQL调优', re.compile(u"SQL|调优")),
                 ('数值策划', re.compile(u"数值|策划")), ('求职应聘', re.compile(u"求职|应聘")), ('广告算法', re.compile(u"广告算法")),
                 ('选题策划', re.compile(u"选题|策划")), ('游戏运营', re.compile(u"游戏运营")), ('需求分析', re.compile(u"需求分析")),
                 ('文案编辑', re.compile(u"文案|编辑")), ('运营', re.compile(u"运营")), ('推荐算法', re.compile(u"推荐算法|推荐")),
                 ('宣传推广', re.compile(u"宣传|推广")), ('电子商务', re.compile(u"电子|商务")), ('沟通能力', re.compile(u"沟通能力|沟通")),
                 ('物料制作', re.compile(u"物料|制作")), ('交互设计', re.compile(u"交互|设计")), ('APP', re.compile(u"APP")),
                 ('爬虫', re.compile(u"爬虫")), ('渠道增长', re.compile(u"渠道增长")), ('资源谈判', re.compile(u"资源谈判|谈判")),
                 ('数据采集', re.compile(u"数据采集")), ('产品', re.compile(u"产品")), ('机器学习', re.compile(u"机器学习|深度学习|人工智能")),
                 ('视频编解码', re.compile(u"视频|编解码")), ('游戏策划', re.compile(u"游戏策划")),]
    for e in txt:
        sys.stdout.write('Handle progress: ' + str(idcnt) + ' /  ' + str(len(txt)) + '\n'); sys.stdout.flush(); idcnt += 1
        split_text = e.split('\t')
        question = filterHtmlTag(split_text[2])
        labels = json.loads((split_text[4]))
        '''
        for e in labels:
            if e[1] not in label2qes: label2qes[e[1]] = set()
            label2qes[e[1]].add(question)
            if e[1] not in label_cnt: label_cnt[e[1]] = 0
            label_cnt[e[1]] += 1
        '''
        for e1, e2 in re_patten:
            if e2.search(question):
                res.append(e1 + '\t' + question + '\n'); break
                aa=e2.search(question)
    a=1
    '''
    sorted_label2qes = sorted(label2qes.items(), key=lambda d:len(d[1]), reverse=True)
    sorted_label_cnt = sorted(label_cnt.items(), key=lambda d:d[1], reverse=True)
    for e in sorted_label2qes:
        if len(e[1]) < 1000: continue
        for e1 in e[1]: res.append(e[0] + '\t' + e1 + '\n')
    '''
    random.shuffle(res)
    with open(train_file, 'w', encoding='utf8') as f1:
        for e in res[:int(len(res) * 0.6)]: f1.write(e)
    with open(test_file, 'w', encoding='utf8') as f2:
        for e in res[int(len(res) * 0.6):int(len(res) * 0.8)]: f2.write(e)
    with open(val_file, 'w', encoding='utf8') as f3:
        for e in res[int(len(res) * 0.8):]: f3.write(e)
    a=1

if __name__ == '__main__':
    #min_edit_distance('求职', '求职应聘')
    #get_sample('./data/q1.res', './data/sen_class_corp666.md')
    #get_fasttext_sample('./data/q1.res', './data/fasttext.train')
    #get_ft_data('./data/sen_class_corp666.md', './data/sen_class.train', './data/sen_class.test', './data/sen_class.val')
    patt = re.compile(r"活动|策划"); aa=patt.search("活动着")
    get_sample_new('./data/q1.res', './data/sen_class.train', './data/sen_class.test', './data/sen_class.val')