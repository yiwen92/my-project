# -*- coding: utf-8 -*-
import requests
from parsel import Selector
import json
import time
from copyheaders import headers_raw_to_dict
from requests_toolbelt.multipart.encoder import MultipartEncoder
import urllib
import gzip
import io
import demjson
import os
import http.cookiejar
import re
from bs4 import BeautifulSoup
import random
import csv
import codecs

s = requests.session()
s.headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1',
}

def spider(search_keyword):
    print('search_keyword', search_keyword)
    d = {'q':search_keyword}
    q = urllib.parse.urlencode(d)
    url2 = 'https://www.zhihu.com/r/search?'+q+'&correction=1&type=content&offset=0'  
    z2 = s.get(url2)

    json_obj = demjson.decode(z2.text)
    html = json_obj["htmls"]    # ;print (html)
    #已经获取到了html编码的JSON流
    soup = BeautifulSoup(str(html), "lxml")
    conts = soup.find_all('div',class_='entry-body')

    tit = soup.find_all('div',class_='title')
    cont = soup.find_all('script',class_='content')
    dr = re.compile(r'<[^>]+>',re.S)

    question_set = set()
    for i in range(len(conts)):
        question = tit[i].a.get_text()
        answer = dr.sub("",cont[i].get_text())
        question_set.add(question)

    pgn = 0

# 下面的设置为  pgn != -1时，将抓取所有结果，可能抓取到的结果十分庞大，pgn为3时，抓取前30条检索结果

    while json_obj["paging"]["next"] != "" and pgn != 30:
        pgn = pgn + 1
        time.sleep(random.randint(5,25))    # 避免刷新过快设置的挂起函数

        url2 = 'https://www.zhihu.com'+str(json_obj["paging"]["next"])
        #print (url2)
        z2 = s.get(url2)
        json_obj = demjson.decode(z2.text)
        html = json_obj["htmls"]
        soup = BeautifulSoup(str(html), "lxml")
        conts = soup.find_all('div',class_='entry-body')
        tit = soup.find_all('div',class_='title')
        cont = soup.find_all('script',class_='content')

        for i in range(len(conts)):
            question = tit[i].a.get_text()
            answer = dr.sub("", cont[i].get_text())
            question_set.add(question)
    f = open("./Save/" + str(search_keyword) + '_' + str(len(question_set)) + ".txt", "w", encoding='utf-8')
    for e in question_set:
        f.write(e + '\n')
    f.close()
    print('question number: ', len(question_set))
    a=1


if __name__ == '__main__':
    capacity_txt = [e.strip().split(',') for e in open('./data/capacity_point.csv', 'r', encoding='utf8').readlines()[1:]]
    for e1 in capacity_txt:
        for e in e1:
            spider(e)