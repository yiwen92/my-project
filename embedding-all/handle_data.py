from utils import PUNCTUATION_LIST, STOP_WORDS, re_ch, re_salary
import random

def is_valid_entity(entity):
    entity = entity.strip().lower()
    if entity in PUNCTUATION_LIST or entity in STOP_WORDS or entity.isdigit(): return False     # 过滤标点符号和停用词
    if len(entity) == 1 and re_ch.fullmatch(entity): return False       # 过滤单个汉字
    if re_salary.fullmatch(entity): return False        # 过滤无效的薪水
    if len(entity) == 1 and entity.isalpha() and entity not in ['c']: return False      # 过滤单个无效的单词
    return True

def title_entitys2entity_dict():
    entitys = []
    title_entitys = [e.strip().lower() for e in open("dict/title_entitys", encoding="utf8").readlines()]
    #random.shuffle(title_entitys)
    for ent in title_entitys:
        if is_valid_entity(ent):
            entitys.append(ent)
    with open("dict/entitys.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(entitys))
    a=1

if __name__ == "__main__":
    title_entitys2entity_dict()