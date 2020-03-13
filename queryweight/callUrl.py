# coding:utf8
from __future__ import division
import json, traceback, sys, time, requests
from urllib import request

local_url = "http://192.168.8.52:6690/?handle=search&m=resume&count=20&keyword="        # 本地
online_url = "http://192.168.8.194:6688/?handle=search&m=resume&count=20&keyword="      # 线上

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
    querys = []
    ftd = open(output_file, "w")
    text = [e.strip().split("\t") for e in open(input_file).readlines()]
    for e in text:
        if e[-1] not in querys: querys.append(e[-1])
    #print(json.dumps(querys, ensure_ascii=False)); exit()
    for q in querys:
        res = getResult(q, url)
        ftd.write(q + "\t" + " ".join([str(e) for e in res]) + "\n")

if __name__ == '__main__':
    try: que = sys.argv[1]
    except: que = "java开发"
    file_name = "feedback2982.res"
    #res = getResult(que, online_url); print(que + "\t" + " ".join([str(e) for e in res]) + "\n")    ; exit()
    print("local..."); getlist(file_name, file_name + ".local", local_url)
    print("online..."); getlist(file_name, file_name + ".online", online_url)

