#coding:utf8
from utils import char_cut
import re

item_regex = re.compile(r'\s*[-\*+]\s*(.+)')

def estimate_model(model_path, test_file_path):
    bin_file = open(model_path, 'r', encoding='utf8'); #aa = bytes.decode(bin_file.read())
    bin_file_content = bin_file.readlines()
    for line in bin_file_content:
        s2 = bytes.decode(line)
        a=1

def test():
    res = {}
    txt = open('./data/sen_class_corp666.md', 'r', encoding='utf8').readlines()
    for line in txt:
        if 'intent:' in line: continue
        match = re.match(item_regex, line)
        if match:
            item = match.group(1)
            seg_item = char_cut(item)
            len_seg_item = len(seg_item)
            if len_seg_item not in res: res[len_seg_item] = 0
            res[len_seg_item] += 1
    sen_lens = []
    for k, v in res.items():
        sen_lens.append(k)
    sen_lens.sort()
    mean_len = sum(sen_lens) / len(sen_lens); min_len = sen_lens[0]; max_len = sen_lens[-1]

    a=1

if __name__ == '__main__':
    test()
    #estimate_model('./models/sen_class', './data/sen_class.test')