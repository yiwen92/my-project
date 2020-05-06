# coding:utf8
from __future__ import division
import json, traceback, sys, time, requests, re, random
from collections import defaultdict
from tqdm import tqdm

count = 20
local_url = "http://192.168.8.52:6690/?handle=search&m=jd&count="+str(count)+"&showquery=1&keyword="        # 本地
online_url = "http://192.168.8.196:6688/?handle=search&m=jd&count="+str(count)+"&showquery=1&keyword="      # 线上
baseline_url = "http://192.168.7.218:6690/?handle=search&m=jd&count="+str(count)+"&showquery=1&keyword="      # 基线
edps_url = "http://192.168.7.219:6688/?handle=search&m=jd&count="+str(count)+"&showquery=1&keyword="      # 基线

def getResult(query, url):
    res = []
#    print("url: %s" % (url))
    try:
        query = query.replace(" ", "%20")
        headers = {"content-type": "application/json"}
        response = requests.post(url=url + str(query), headers=headers)
        rd = response.json();
#        print(json.dumps(json.loads(rd['response']['results']['query_param']),ensure_ascii = False, indent=2)); #print(json.dumps(rd['response']['results']['query'], ensure_ascii = False));#exit()
        if 'list' in rd['response']['results']:
            res = rd['response']['results']['list']
        else:
            return res
    except Exception as e:
        tb = traceback.format_exc(); print('traceback:%s, query: %s' % (str(tb), query))
    return res

def getlist(input_file, output_file, url):
    print("input_file: %s\toutput_file: %s\turl: %s" %(input_file, output_file, url))
    querys = []
    ftd = open(output_file, "w")
    text = [e.strip().split("\t") for e in open(input_file).readlines()]
    for e in text:
        if e[-1] not in querys: querys.append(e[-1])
    #print(json.dumps(querys, ensure_ascii=False)); exit()
    for q in tqdm(querys, total=len(querys)):
        res = getResult(q, url)
        ftd.write(q + "\t" + " ".join([str(e) for e in res]) + "\n")

def get_sort_res(url, input_file, output_file):
    random.seed(8)
    print("input_file: %s\toutput_file: %s\turl: %s" % (input_file, output_file, url))
    query_freq = defaultdict(int)
    re_obj = re.compile(r'(.+)\t ([0-9]+)', re.M | re.I)
    ftd = open(output_file, "w")
    for line in open(input_file, encoding="utf8").readlines():
        re_res = re_obj.match(line)
        if not re_res: continue
        query, freq = re_res.group(1).lower(), int(re_res.group(2))
        query_freq[query] += freq
    sortedquery = [k for k, v in sorted(query_freq.items(), key=lambda d: d[1], reverse=True)][:10000]
    random.shuffle(sortedquery)
    sorted_query = sortedquery[:100]
    for q in tqdm(sorted_query, total=len(sorted_query)):
        res = getResult(q, url)
        ftd.write(q + "\t" + " ".join([str(e) for e in res]) + "\n")

def diff_result(query, url1, url2):
    res1 = getResult(query, url1);  print(res1)
    res2 = getResult(query, url2);  print(res2)
    diffres = [(i, res1[i], res2[i]) for i in range(len(res1)) if res1[i] != res2[i]]
    print("diff_result: ", diffres)

def get_diff():
    diff_order, diff_noorder = [], []
    local = {e.strip().split("\t")[0]: e.strip().split("\t")[1] for e in open("sort_search_data.local", encoding="utf8").readlines() if len(e.strip().split("\t")) > 1}
    online = {e.strip().split("\t")[0]: e.strip().split("\t")[1] for e in open("sort_search_data.online", encoding="utf8").readlines() if len(e.strip().split("\t")) > 1}
    for e in local:
        baseline_ids, new_ids = local[e], online[e]
        if baseline_ids != new_ids: diff_order.append(e)
        if set(baseline_ids.split()).symmetric_difference(set(new_ids.split())):
            diff_noorder.append(e)      #a = set(new_ids.split()).symmetric_difference(set(baseline_ids.split()))
    print("len: %d\ndiff_order: \n%s" % (len(diff_order), '\t'.join(diff_order)))
    for e in diff_order:
        res1, res2 = getResult(e, local_url), getResult(e, online_url)
        diffres = [(i, res1[i], res2[i]) for i in range(len(res1)) if res1[i] != res2[i]]
        if diffres: print(e, diffres)

if __name__ == '__main__':
    try: que = sys.argv[1]
    except: que = "java开发"
    file_name = "sort_search_data"  #"sort_search_data"    #"feedback2982.res"
#    get_diff(); exit()
    #diff_result(que, local_url, online_url); exit()
#    diff_result(que, baseline_url, online_url); exit()
    #diff_result(que, baseline_url, local_url); exit()
    '''
    print(getResult(que, baseline_url)); #exit()
    print(getResult(que, online_url)); #exit()
    print(getResult(que, edps_url)); exit()
    '''
#    get_sort_res(local_url, file_name, file_name + ".local")
    get_sort_res(online_url, file_name, file_name + ".online")
#    get_sort_res(baseline_url, file_name, file_name + ".baseline")
    #get_sort_res(edps_url, file_name, file_name + ".edps")

    #get_sort_res(baseline_url, file_name, file_name + ".baseline.expand")
#    get_sort_res(new_url, file_name, file_name + ".new.expand")
    #res = getResult(que, online_url); print(que + "\t" + " ".join([str(e) for e in res]) + "\n")    ; exit()
    #print("baseline..."); getlist(file_name, file_name + ".baseline", baseline_url)
    #print("new..."); getlist(file_name, file_name + ".new", new_url)
    #print("local..."); getlist(file_name, file_name + ".local", local_url)
    #print("online..."); getlist(file_name, file_name + ".online", online_url)

