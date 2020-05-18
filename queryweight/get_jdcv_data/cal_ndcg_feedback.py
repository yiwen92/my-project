import math, json

def cal_ndcg(label_list, topk):
    label_list = [int(e) for e in label_list]
    dcg, idcg = 0.0, 0.0
    pred = label_list[:topk]
    label = sorted(label_list, key=lambda d: d, reverse=True)[: topk]
    diff = topk - len(label_list)
    if diff > 0:        # 分数补全
        pred = pred + [0] * diff
        label = label + [0] * diff
    for i in range(len(pred)):
        dcg += (pow(2, pred[i]) - 1) / math.log2(i + 2)
        idcg += (pow(2, label[i]) - 1) / math.log2(i + 2)
    ndcg = dcg / (idcg + 1e-8)
    return dcg, idcg, ndcg

def cal(file_name):
    query_res = {}; ids = file_name.split('.')[-1].split('_')
    taskid_keyword_label, res, file_res = {}, {}, {}
    text = [e.strip().split("\t") for e in open(file_name, encoding="utf8").readlines()]
    for i, line in enumerate(text):
        if i == 0: field2id = {e: i for i, e in enumerate(line)}; continue
        task_id, keyword, is_correct = line[field2id['task_id']], line[field2id['keyword']], line[field2id['is_correct']]
        if task_id not in taskid_keyword_label: taskid_keyword_label[task_id] = {}
        if keyword not in taskid_keyword_label[task_id]: taskid_keyword_label[task_id][keyword] = []
        taskid_keyword_label[task_id][keyword].append(is_correct)
    idkeywordlist = {k+'_'+keyword: labels for k, value in taskid_keyword_label.items() for keyword, labels in value.items()}
    for task_id, keyword_label in taskid_keyword_label.items():
        dcg_avg, ndcg_avg, num = 0.0, 0.0, 0
        compare_id = [e for e in ids if e != task_id][0]
        for keyword, label_list in keyword_label.items():
            if label_list == idkeywordlist[compare_id+'_'+keyword]: continue        # 过滤掉相同的结果
            dcg, idcg, ndcg = cal_ndcg(label_list, 15)
            dcg_avg += dcg; ndcg_avg += ndcg; num += 1
            query_res[task_id + '_' + keyword] = (round(dcg), round(ndcg, 3))
        if task_id not in res: res[task_id] = {}
        res[task_id]['dcg_avg'], res[task_id]['ndcg_avg'] = round(dcg_avg / num, 3), round(ndcg_avg / num, 3)
    print(json.dumps(res))
    return query_res

def cal1():
    query_res = {}
    baseline = [e.strip().split("\t") for e in open("label.3026", encoding="utf8").readlines()]
    new = [e.strip().split("\t") for e in open("label.3025", encoding="utf8").readlines()]
    baseline_ = {k: v for k, v in baseline}   ;   new_ = {k: v for k, v in new}
    dcg_base_avg, ndcg_base_avg, dcg_new_avg, ndcg_new_avg, num = 0.0, 0.0, 0.0, 0.0, 0
    for keyword, labels in baseline:
        if keyword in ['培训', 'linux', '自动化测试', 'hadoop', '京东', '广告销售']: continue  # 临时策略，过滤掉无效的结果
        if labels == new_[keyword]: continue
        dcg, idcg, ndcg = cal_ndcg(labels.split(), 15)
        dcg_base_avg += dcg; ndcg_base_avg += ndcg; num += 1
        query_res['3026_' + keyword] = (round(dcg), round(ndcg, 3))
    for keyword, labels in new:
        if keyword in ['培训', 'linux', '自动化测试', 'hadoop', '京东', '广告销售']: continue  # 临时策略，过滤掉无效的结果
        if labels == baseline_[keyword]: continue
        dcg, idcg, ndcg = cal_ndcg(labels.split(), 15)
        dcg_new_avg += dcg; ndcg_new_avg += ndcg
        query_res['3025_' + keyword] = (round(dcg, 3), round(ndcg, 3))
    dcg_base_avg /= num; ndcg_base_avg /= num; dcg_new_avg /= num; ndcg_new_avg /= num
    print("dcg_base_avg: %.3f\tndcg_base_avg: %.3f\ndcg_new_avg: %.3f\tndcg_new_avg: %.3f" % (dcg_base_avg, ndcg_base_avg, dcg_new_avg, ndcg_new_avg))
    return query_res

if __name__ == "__main__":
    a=cal_ndcg("3 3 3 3 3 3 3 3 3 3 3 3 3 3 1".split(), 15)
    b=cal_ndcg("3 3 3 3 3 3 3 3 3 3 3 3 1 3 3".split(), 15)
    #cal("feedback.3015_3017")
    r1 = cal("feedback.3025_3026")
    #r2 = cal1(); print(json.dumps(r2, ensure_ascii=False))
    pass