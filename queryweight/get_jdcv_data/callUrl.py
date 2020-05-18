# coding:utf8
from __future__ import division
import json, traceback, sys, time, requests, re
from collections import defaultdict
from tqdm import tqdm

local_url = "http://192.168.8.52:6690/?handle=search&m=resume&count=20&keyword="        # 本地
online_url = "http://192.168.8.194:6688/?handle=search&m=resume&count=20&keyword="      # 线上
baseline_url = "http://192.168.7.218:6690/?handle=search&m=resume&count=20&keyword="        # 本地
new_url = "http://192.168.8.52:6690/?handle=search&m=resume&count=20&keyword="        # 本地

def getResult(query, url):
    res = []
    try:
        query = query.replace(" ", "%20")
        headers = {"content-type": "application/json"}
        response = requests.post(url=url + str(query), headers=headers)
        response_dict = response.json(); #print(json.dumps(response_dict, ensure_ascii = False));exit()
        if 'list' in response_dict['response']['results']:
            res = response_dict['response']['results']['list']
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
    for q in querys:
        res = getResult(q, url)
        ftd.write(q + "\t" + " ".join([str(e) for e in res]) + "\n")

def get_sort_res(url, input_file, output_file):
    print("input_file: %s\toutput_file: %s\turl: %s" % (input_file, output_file, url))
    query_freq = defaultdict(int)
    re_obj = re.compile(r'(.+)\t ([0-9]+)', re.M | re.I)
    ftd = open(output_file, "w")
    for line in open(input_file, encoding="utf8").readlines():
        re_res = re_obj.match(line)
        if not re_res: continue
        query, freq = re_res.group(1).lower(), int(re_res.group(2))
        query_freq[query] += freq
    sorted_query = [k for k, v in sorted(query_freq.items(), key=lambda d: d[1], reverse=True)[:2000]]
    for q in tqdm(sorted_query, total=len(sorted_query)):
        res = getResult(q, url)
        ftd.write(q + "\t" + " ".join([str(e) for e in res]) + "\n")

if __name__ == '__main__':
    try: que = sys.argv[1]
    except: que = "java开发"
    file_name = "sort_search_data"    #"feedback2982.res"
    get_sort_res(baseline_url, file_name, file_name + ".baseline")
    get_sort_res(new_url, file_name, file_name + ".new")
    #res = getResult(que, online_url); print(que + "\t" + " ".join([str(e) for e in res]) + "\n")    ; exit()
    #print("baseline..."); getlist(file_name, file_name + ".baseline", baseline_url)
    #print("new..."); getlist(file_name, file_name + ".new", new_url)
    #print("local..."); getlist(file_name, file_name + ".local", local_url)
    #print("online..."); getlist(file_name, file_name + ".online", online_url)

