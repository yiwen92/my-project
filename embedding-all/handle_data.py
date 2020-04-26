from utils import PUNCTUATION_LIST, STOP_WORDS, re_ch, re_salary
import random, os, re
from seg_utils import Tokenizer, STOP_WORDS, PUNCTUATION_LIST
from tqdm import tqdm
from collections import Counter
from config import conf
from collections import defaultdict

def is_valid_entity(entity):
    entity = entity.strip().lower()
    if entity in PUNCTUATION_LIST or entity in STOP_WORDS or entity.isdigit(): return False     # 过滤标点符号和停用词
    if len(entity) == 1 and re_ch.fullmatch(entity): return False       # 过滤单个汉字
    if re_salary.fullmatch(entity): return False        # 过滤无效的薪水
    if len(entity) == 1 and entity.isalpha() and entity not in ['c']: return False      # 过滤单个无效的单词
    return True

def gen_entity_dict():
    token = Tokenizer()
    '''
    title_freq = Counter([line.split("\t")[0] for line in open("data/jdtitledesc", encoding="utf8").readlines()])
    top_title_freq = title_freq.most_common()
    with open("data/total_title", "w", encoding="utf8") as fin:
        for t, f in top_title_freq:
            fin.write(t + "\t" + str(f) + "\n")
    '''
    match_obj = re.compile("(.+)\t([0-9]+)", re.M | re.I)
    titles, title_words = [], []
    stop_word_re = "[" + "|".join(STOP_WORDS) + "]{1,}"
    custom_word_re = "[急聘|诚聘|双休|代表|高薪|五险]{1,}"
    punction_re = "[" + "\|".join([e for e in PUNCTUATION_LIST]) + "]{1,}"
    salary_re = "50k"
    t=re.sub(custom_word_re + stop_word_re , "", "50k,{急聘客服专员（双休  五险一金）")
    text = [line.strip().lower() for line in open("data/total_title", encoding="utf8").readlines()]
    for line in tqdm(text, total=len(text)):
        match_res = match_obj.match(line)
        if not match_res: continue
        title, freq = match_res.group(1), int(match_res.group(2))
        if freq <= 2 or len(title) >= 10: continue
        #title = "50k,急聘客服专员（双休  五险一金）"
        title = re.split("[(|（ )/]", title)[0]
        title = re.sub(custom_word_re + stop_word_re, "", title)
        titles.append(title)
        title_words.extend(token.cut(title)[0])
    title_freq = Counter(title_words).most_common()
    with open("data/title_entitys", "w", encoding="utf8") as fin:
        for t, f in title_freq:
            fin.write(t + "\n")
    with open("data/valid_titles", "w", encoding="utf8") as fin:
        for t, f in Counter(titles).most_common():
            fin.write(t + "\n")
    exit()

def title_entitys2entity_dict():
    entitys = []
    title_entitys = [e.strip().lower() for e in open("data/title_entitys", encoding="utf8").readlines()]
    #random.shuffle(title_entitys)
    for ent in title_entitys:
        if is_valid_entity(ent):
            entitys.append(ent)
    with open(conf.entity_file, "w", encoding="utf8") as fin:
        fin.write("\n".join(entitys))
    a=1

def get_corpus(file_path="position_name_desc_re"):
    title_entitys = {}
    token = Tokenizer()
    for file_name in os.listdir(file_path):     # 遍历文件夹里的文件
        text = [line.strip().lower().replace("\\n", "").split('\t') for line in open(file_path + "/" + file_name, encoding="utf8").readlines()]
        for line in tqdm(text, total=len(text)):
            if len(line) <=1: continue
            import_tokens = token.select_important_tokens("".join(line[1:]))
            if line[0] not in title_entitys: title_entitys[line[0]] = []
            title_entitys[line[0]].extend(import_tokens)
        a=1

def clean_title_desc():
    valid_titles = {line.strip(): 1 for line in open("data/valid_titles", encoding="utf8").readlines()}
    matchObj = re.compile(r'(.+)&([0-9]+)', re.M | re.I)
    texts = [line.strip().split("&")[0].split("\t") for line in open(conf.jdtitledesc_file, encoding="utf8").readlines() if matchObj.match(line)]
    text = []
    for line in tqdm(texts, total=len(texts)):
        if line[0] not in valid_titles: continue
        text.append(line)
        a = 1
    exit()

def test():
    a = open("cvtitledesc/part-00000", encoding="utf8").readlines()
    #text = {e.strip(): len(e.strip()) for e in open("dict/new_entity", encoding="utf8").readlines()}
    text = defaultdict(int)
    for e in open("dict/func.txt", encoding="utf8").readlines(): text[e.strip().lower().split()[0]] = len(e.strip().lower().split()[0])
    for e in open("dict/ind.txt", encoding="utf8").readlines(): text[e.strip().lower().split()[0]] = len(e.strip().lower().split()[0])
    text = {k: v for k, v in text.items() if v > 2 and k not in ['互联网','公积金','职业道德','办公室','合伙人','解决方案','健身房']}
    sorted_text = sorted(text.items(), key=lambda d: d[1])
    with open("dict/func_ind.txt", "w", encoding="utf8") as fin:
        for k, v in sorted_text:
            fin.write(k + "\n")
    exit()
    '''
    indu_func = set()
    text = [line.lower().strip().split('\t') for line in open("hu_ner.txt", encoding="utf8").readlines()]
    for line in text:
        if len(line) >=2 and line[1] in ['indu', 'func']:
            indu_func.add(line[0])
    with open("dict/indu_func", "w", encoding="utf8") as fin:
        fin.write("\n".join(indu_func))
    '''
    new_entitys = set()
    indu_func = [line.lower().strip() for line in open("dict/indu_func", encoding="utf8").readlines()]
    func = [line.lower().strip().split()[0] for line in open("dict/func.txt", encoding="utf8").readlines()]
    indu = [line.lower().strip().split()[0] for line in open("dict/ind.txt", encoding="utf8").readlines()]
    new_entitys.update(set(indu_func)); new_entitys.update(set(func)); new_entitys.update(indu)
    with open("dict/new_entity", "w", encoding="utf8") as fin:
        fin.write("\n".join(new_entitys))
    exit()
    ent_freq = defaultdict(int)
    text = [line.replace("\\n", "").split('\t')[1:] for line in open("data/title2entitys_corpu", encoding="utf8").readlines()[:100]]
    for line in text:
        for ent in set(line):
            ent_freq[ent] +=1
    sorted_ent_freq = sorted(ent_freq.items(), key=lambda d: d[1], reverse=True)
    with open("data/entity_idf", "w", encoding="utf8") as fin:
        for k, v in sorted_ent_freq:
            fin.write(k + "\t" + str(v) + "\n")
    ents = Counter([e for line in text for e in line]).most_common()
    a=1

def gen_func_indu_entity():
    re_ch = re.compile(u"([\u4e00-\u9fa5])",re.S)   ; s = re_ch.match("小无")
    func_indu = set()
    # function 职能实体
    for line in open("dict/functions.csv").readlines()[1:]: func_indu.add(line.strip().lower().split(",")[3])
    for line in open("dict/function_competence.txt", encoding="utf8").readlines()[1:]: func_indu.add(line.strip().lower().split("\t")[1])
    for line in open("dict/function_department_mapping_dict.txt", encoding="utf8").readlines()[1:]: func_indu.add(line.strip().lower().split("\t")[3])
    for line in open("dict/function_frequency.txt", encoding="utf8").readlines()[1:]: func_indu.add(line.strip().lower().split("\t")[1])
    for line in open("dict/function_keyword", encoding="utf8").readlines(): func_indu.add(line.strip().lower())
    for line in open("dict/function_query.txt", encoding="utf8").readlines(): func_indu.add(line.strip().lower())
    for line in open("dict/function_title_hit_requirement.txt", encoding="utf8").readlines()[1:]: func_indu.add(line.strip().lower().split("\t")[1])
    # industry 行业实体
    for line in open("dict/industry_alias.txt", encoding="utf8").readlines(): func_indu.add(line.strip().lower().split("\t")[0])
    for line in open("dict/industry.csv", encoding="utf8").readlines()[1:]: func_indu.add(line.strip().lower().split(",")[1])
    func_indu_len = {e: len(e) for e in func_indu}
    sorted_func_indu_len = sorted(func_indu_len.items(), key=lambda d: d[1])
    bw = ['生产','配送','特惠','现任','兼任','存储','驱动','保卫','工程','设计','服务','包装','即将','商场','发行','真有','认证','播放','加工','担保','陈丽', \
        '邮箱','勘探','活动','食堂','王敏','融资','传达','策划','计划','餐饮','传播']
    with open("dict/fun_indu_entity.txt", "w", encoding="utf8") as fin:
        for k, v in sorted_func_indu_len:
            if k in bw or (len(k) == 2 and re_ch.match(k)): continue
            fin.write(k + "\n")
    a=1

def gen_train_corpus_cv(file_path="cvtitledesc"):
    title_entitys = {}
    file_list = os.listdir(file_path)
    for file_name in tqdm(file_list, total=len(file_list)):     # 遍历文件夹里的文件
        text = [line.strip().lower().split("\t") for line in open(file_path + "/" + file_name, encoding="utf8").readlines()]
        for k, v in text:
            title = re.sub("[ ]{1,}", "", k)
            entitys = set(v.split("|"))
            if title not in title_entitys: title_entitys[title] = set()
            title_entitys[title].update(entitys)
    with open("data/cv_title2entitys_corpu", "w", encoding="utf8") as fin:
        for k, v in title_entitys.items():
            fin.write("\t".join([k] + list(v)) + "\n")
    a=1

def gen_train_corpus_jd(file_path="jdtitledesc_2020_04_17_20_57_43"):
    func_indu_entity = {line.strip(): len(line.strip()) for line in open("dict/fun_indu_entity.txt", encoding="utf8").readlines()}
    title_entitys = {}
    file_list = os.listdir(file_path)
    valid_titles = {line.strip(): 1 for line in open("data/valid_titles", encoding="utf8").readlines()}
    for file_name in tqdm(file_list, total=len(file_list)):     # 遍历文件夹里的文件
        text = [line.strip().lower() for line in open(file_path + "/" + file_name, encoding="utf8").readlines()]
        if not text: continue
        for line in text:
            seg_line = line.split("\t")
            title, desc = seg_line[0], seg_line[1:]
            if title not in valid_titles: continue
            if title not in title_entitys: title_entitys[title] = set()
            for e in desc:
                if e in func_indu_entity:
                    title_entitys[title].add(e)
    with open("data/jd_title2entitys_corpu", "w", encoding="utf8") as fin:
        for k, v in title_entitys.items():
            fin.write("\t".join([k] + list(v)) + "\n")
    a=1

if __name__ == "__main__":
    #gen_entity_dict()  # 根据1300万title过滤产生title里面的实体词集合、有效的title集合
    #title_entitys2entity_dict()    # 根据title实体词集合过滤得到最终的实体词典
    #get_corpus()
    #clean_title_desc()
    #test()
    #gen_func_indu_entity()
    #gen_train_corpus_cv()      # 简历中的标题和实体关系
    gen_train_corpus_jd()       # 职位中的标题和实体关系
    pass