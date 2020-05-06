

def diff_feedback(file_name):
    diff_order, diff_noorder = [], []
    baseline = {e.strip().split("\t")[0]: e.strip().split("\t")[1] for e in open(file_name + ".baseline", encoding="utf8").readlines() if len(e.strip().split("\t")) > 1}
    new = {e.strip().split("\t")[0]: e.strip().split("\t")[1] for e in open(file_name + "new", encoding="utf8").readlines() if len(e.strip().split("\t")) > 1}
    for e in baseline:
        baseline_ids, new_ids = baseline[e], new[e]
        if baseline_ids != new_ids: diff_order.append(e)
        if set(baseline_ids.split()).symmetric_difference(set(new_ids.split())):
            diff_noorder.append(e)      #a = set(new_ids.split()).symmetric_difference(set(baseline_ids.split()))
    print("number: %d\ndiff_order: %s" % (len(diff_order), '\n'.join(diff_order)))
    #print("diff_order: \n", '\n'.join(diff_order), "\n\ndiff_noorder: \n", '\n'.join(diff_noorder))

def get_test_querys():
    text = [e.strip().split("\t")[-1] for e in open("get_jdcv_data/feedback2982.res", encoding="utf8").readlines()[1:]]
    res = [e.strip() for e in open("get_jdcv_data/test_querys.txt", encoding="utf8").readlines()]
    for e in text:
        if e not in res: res.append(e)
    with open("test_querys.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[:150]))
    a=1

if __name__ == "__main__":
    file_name = "sort_search_data"    #"feedback2982.res"
    #diff_feedback(file_name)
    get_test_querys()
