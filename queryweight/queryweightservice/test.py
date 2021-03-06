import re, json
from seg_utils import Tokenizer
from tqdm import tqdm
from qw import query_weight
from utils import cal_ndcg
from collections import Counter
from data_utils import preprocess_text

def gen_true_data(source_path, out_path):
    t = Tokenizer()
    res = []
    total_num = len(open(source_path, encoding="utf8").readlines())
    with open(source_path, encoding="utf8") as fin:
        for line in tqdm(fin, total=total_num):
            query = line.strip().split("\t")[0]
            senten2term, _ = t.tokenize(query)
            if len(senten2term) < 2: continue
            res.append("\t".join(senten2term) + "\n")
    with open(out_path, "w", encoding="utf8") as fin:
        fin.write("".join(res))

def test():
    qw = query_weight()
    pred_num, total_num = 0, 0
    text = [e.strip().split("\t") for e in open("get_jdcv_data/querytrue.txt", encoding="utf8").readlines()[1:159]]
    for (i, (query, label)) in enumerate(text):
        #query = "移动医疗"
        res = qw.run_step(query)
        pred = sorted(res, key=lambda d: d[1], reverse=True)[0]
        if pred[0] == label.split()[0]: pred_num += 1
        else: print(str(i+1) + "\t" + query + "\t" +" ".join([k+":"+str(v) for k, v in res]) + "\t" + pred[0] + "_" + label.split()[0])
        #if set([k for k, v in res]).difference(set(label.split())): print(str(i+1), '\t', " ".join([k for k, v in res]), '\t', label)
        total_num += 1
    print("acc: %f" % (round(pred_num / total_num, 3)))
    a=1

def cal_feedback_ndcg(file_name="get_jdcv_data/feedback.res"):
    query_label, query_ndcg, dcg_sum, ndcg_sum = {}, {}, 0.0, 0.0
    text = [e.strip().split("\t") for e in open(file_name, encoding="utf8").readlines()]
    for i in range(len(text)):
        ele = text[i]
        if i == 0: field2id = {e: i for i, e in enumerate(ele)}
        else:
            keyword, is_correct = ele[field2id['keyword']], ele[field2id['is_correct']]
            if keyword not in query_label: query_label[keyword] = []
            query_label[keyword].append(is_correct)
    for query, label in query_label.items():
        label_freq = sorted(Counter(label).items(), key=lambda d: d[1], reverse=True)
        if len(label_freq) == 1 or (len(label_freq) == 2 and label_freq[1][1] < 3):
            pass; #print(query, '\t', label); #continue
        dcg, idcg, ndcg = cal_ndcg(label, 20)
        query_ndcg[query] = [round(dcg, 3), round(idcg, 3), round(ndcg, 3)]
        dcg_sum += dcg; ndcg_sum += ndcg
    sorted_query_ndcg = sorted(query_ndcg.items(), key=lambda d: d[1][2]); print(json.dumps(query_ndcg, ensure_ascii=False))
    dcg_avg, ndcg_avg = dcg_sum / len(query_ndcg), ndcg_sum / len(query_ndcg)
    print("total query: %d\tvalid query: %d\ndcg_avg: %.3f\tndcg_avg%.3f" % (len(query_label), len(query_ndcg), dcg_avg, ndcg_avg))
    return dcg_avg, ndcg_avg, query_ndcg

def cal_ndcg_train_data(topk=1):
    ndcg_sum = 0.0
    matchObj = re.compile(r'(.+)\t([0-9]+)', re.M | re.I)   ;   qw = query_weight()
    text = [e.strip().split("\t") for e in open("get_jdcv_data/label.data", encoding="utf8").readlines() if e.strip()]
    for line in tqdm(text, total=len(text)):
        seg_line = [(preprocess_text(e.split(":")[0]), e.split(":")[1]) for e in line]
        sorted_seg_line = sorted(seg_line, key=lambda d: d[1], reverse=True)
        rel = {k: len(sorted_seg_line)-i-1 for i, (k, v) in enumerate(sorted_seg_line)}
        query = " ".join([e[0] for e in seg_line])
        dcg, idcg, ndcg = get_one_query_ndcg(qw, query, rel, topk)
        ndcg_sum += ndcg
    ndcg_avg = ndcg_sum / len(text)
    print("ndcg_avg@%d: %.3f" % (topk, ndcg_avg))

def cal_ndcg_manual_data(topk=1):
    qw = query_weight(); ndcg_sum = 0.0
    text = [e.strip().split("\t") for e in open("get_jdcv_data/querytrue.txt", encoding="utf8").readlines()[1:159] if e.strip()]
    for (query, label) in tqdm(text, total=len(text)):
        seg_label = label.split()
        rel = {e: len(seg_label)-i-1 for i, e in enumerate(seg_label)}
        dcg, idcg, ndcg = get_one_query_ndcg(qw, query, rel, topk)
        ndcg_sum += ndcg
    ndcg_avg = ndcg_sum / len(text)
    print("ndcg_avg@%d: %.3f" % (topk, ndcg_avg))

def get_one_query_ndcg(qw, query, rel, topk=1):
    res = qw.run_step(query)
    pred = sorted(res, key=lambda d: d[1], reverse=True)
    label_list = [rel.get(k, 0) for k, v in pred]
    dcg, idcg, ndcg = cal_ndcg(label_list, topk)
    return dcg, idcg, ndcg

def aa():
    res = {}
    dcg_avg_old, ndcg_avg_old, query_ndcg_old = cal_feedback_ndcg("get_jdcv_data/feedback2982.res")
    dcg_avg_new, ndcg_avg_new, query_ndcg_new = cal_feedback_ndcg("get_jdcv_data/feedback2983.res")
    for k, v in query_ndcg_old.items():
        dcg_old, ndcg_old, dcg_new, ndcg_new = query_ndcg_old[k][0], query_ndcg_old[k][2], query_ndcg_new.get(k, [0,0,0])[0], query_ndcg_new.get(k, [0,0,0])[2]
        res[k] = [dcg_old, ndcg_old, dcg_new, ndcg_new]
    print(json.dumps(res, ensure_ascii=False))
    pass

if __name__ == "__main__":
    a=len("211") #"211".isdigit()
    #gen_true_data("get_jdcv_data/query.freq.csv", "get_jdcv_data/query.true")
    test(); exit()
    #cal_feedback_ndcg()
    #cal_ndcg_train_data()
    cal_ndcg_manual_data(1)
    #aa()
    pass
