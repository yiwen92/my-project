import pymysql, math, openpyxl
from tqdm import tqdm

def cal_ndcg(label_list, topk):
    label_list = [int(e) if e != 'None' else 0 for e in label_list]
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
    return round(dcg, 3), round(idcg, 3), round(ndcg, 3)

def get_search_diff_feedback(task_id):
    """
    /opt/app/mysql/bin/mysql -h192.168.8.218 -ukdd -pkd12934d -P3307    (8.52)
    set names utf8;
    """
    db_params = {
        "host": "192.168.8.218",
        "user": "kdd",
        "password": "kd12934d",
        "port": 3307,
        "db": "feedback",
        "charset": "utf8"
    }
    diff_querys = "(" + ",".join(["\"%s\""%t for t in read_xlsx("diff.xlsx", task_id)]) + ")"

    sql = "select * from cs_detail where task_id = " + str(task_id) + " and kw in " + diff_querys
    connect = pymysql.connect(**db_params)
    cursor = connect.cursor()
    cursor.execute(sql)
    fields = cursor.description
    datas = cursor.fetchall()
    new_res_, baseline_res_ = {}, {}
    field2id = {e[0]: i for i, e in enumerate(fields)}
    for data in datas:  # 记录数据
        kw = str(data[field2id['kw']])
        score = str(data[field2id['score']])
        position_index = data[field2id['position_index']]
        interface = data[field2id['interface']]
        if '8.52:6690' in interface:    # 策略的结果
            if kw not in new_res_: new_res_[kw] = []
            new_res_[kw].append(score)
        else:       # 基线的结果
            if kw not in baseline_res_: baseline_res_[kw] = []
            baseline_res_[kw].append(score)
    new_res = new_res_ #{k: v for k, v in new_res_.items() if k in new_res_ and k in baseline_res_ and new_res_[k] != baseline_res_[k]}
    baseline_res = baseline_res_ #{k: v for k, v in baseline_res_.items() if k in new_res_ and k in baseline_res_ and new_res_[k] != baseline_res_[k]}
    dcg_new, ndcg_new = cal_ndcg_query_labels(new_res)
    dcg_baseline, ndcg_baseline = cal_ndcg_query_labels(baseline_res)
    print("task: %d\tlen_new: %d\tlen_baseline: %d\ndcg_new: %.3f\tndcg_new: %.3f\ndcg_baseline: %.3f\tndcg_baseline: %.3f" % \
        (task_id, len(new_res), len(baseline_res), dcg_new, ndcg_new, dcg_baseline, ndcg_baseline))
    print("dcg_pro: %.3f\tndcg_pro: %.3f" % ((dcg_new - dcg_baseline) / dcg_baseline, (ndcg_new - ndcg_baseline) / ndcg_baseline))
    with open("strategy." + str(task_id), "w", encoding="utf8") as fin:
        for k, v in new_res.items():
            fin.write(k + "\t" + " ".join(v) + "\n")
    with open("baseline." + str(task_id), "w", encoding="utf8") as fin:
        for k, v in baseline_res.items():
            fin.write(k + "\t" + " ".join(v) + "\n")
    cursor.close()
    connect.close

def cal_ndcg_query_labels(query_labels):
    query_res = {}
    dcg_avg, ndcg_avg = 0.0, 0.0
    for k, v in query_labels.items():
        dcg, idcg, ndcg = cal_ndcg(v, 15)
        query_res[k] = [dcg, idcg, ndcg]
        dcg_avg += dcg
        ndcg_avg += ndcg
    dcg_avg /= len(query_labels)
    ndcg_avg /= len(query_labels)
    return round(dcg_avg, 5), round(ndcg_avg, 5)

def read_xlsx(file_name, task_id):
    res = []
    workbook = openpyxl.load_workbook(file_name)
    booksheet = workbook.get_sheet_by_name(str(task_id))
    rows, columns = booksheet.rows, booksheet.columns
    for i, row in enumerate(rows):
        if i == 0: continue
        line = [col.value for col in row]
        res.append(line[1])
    return res

if __name__ == '__main__':
    #read_xlsx("diff.xlsx", 3078)
    cal_ndcg([2,1,3,5,1,3,2], 8)
    get_search_diff_feedback(3086)
    get_search_diff_feedback(3078)
