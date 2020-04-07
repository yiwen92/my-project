#!/usr/bin/env
# coding:utf8

from __future__ import division
from elasticsearch import Elasticsearch
import logging, json
from service_config import es_ip

class queryMap:
    def __init__(self, key = '', value = '', type = ''):
        self.type = type
        self.termMap = {}
        self.termMap['term'] = {key: value}

class query:
    def __init__(self):
        self.query = {}
        self.query['bool'] = {}
        self.query['bool']['must'] = []
        self.query['bool']['must_not'] = []
        self.query['bool']['should'] = []
        self.source_arr = ["id", "industry", "company", "position", "capacity_point", "function_must",
                           "function_should", "company_id", "industry_id", "feedback"]
        self.hasQuery = False

    def queryBuilder(self, queryMap):
        if queryMap.type == 'must':
            self.query['bool']['must'].append(queryMap.termMap)
            self.hasQuery = True
        elif queryMap.type == 'must_not':
            self.query['bool']['must_not'].append(queryMap.termMap)
            self.hasQuery = True
        elif queryMap.type == 'should':
            self.query['bool']['should'].append(queryMap.termMap)
            self.hasQuery = True

class ElasticObj:
    def __init__(self, index_name, index_type, ip = "127.0.0.1", port = 9200):
        '''
        :param index_name: 索引名称
        :param index_type: 索引类型
        '''
        self.index_name =index_name
        self.index_type = index_type
        try:
            self.es = Elasticsearch([ip], port = port)                                          # 无用户名密码连接 ES
            #self.es = Elasticsearch([ip], http_auth=('elastic', 'password'), port=9200)        # 用户名密码连接 ES
        except Exception as e:
            logging.warn('elastic_connection_err=%s' % repr(e))
            #print('elastic_connection_err=%s' % repr(e))

    def getDataById(self, id):
        try:
            res = self.es.get(index=self.index_name, doc_type=self.index_type, id=id)
            return res
        except Exception as e:
            logging.warn('getDataById_err=%s' % repr(e))
            #print('getDataById_err=%s' % repr(e))
            return None

    def getDataByQuery(self, Query, Size = 10):
        try:
            body={"query": Query.query, "_source": Query.source_arr}
            searched = self.es.search(index=self.index_name, body=body, size = Size)
            return searched
        except Exception as e:
            logging.warn('getDataByQuery_err=%s' % repr(e))
            #print('getDataByQuery_err=%s' % repr(e))
            return None

    def getAllData(self, Size = 1000000000):
        try:
            searched = self.es.search(index=self.index_name, body={'query': {'match_all': {}}}, size = Size)
            return searched
        except Exception as e:
            logging.warn('getAllData_err=%s' % repr(e))
            # print('getAllData_err=%s' % repr(e))
            return None


if __name__=="__main__":
    _ip = es_ip     # 开发：'192.168.1.107'；测试：'10.9.10.7'；线上：'192.168.8.190'
    index_name = 'dashixiong';  index_type = 'consultant'
    es_obj = ElasticObj(index_name, index_type, ip = _ip)
    sech = es_obj.getAllData(); print sech['hits']['total'],  #json.dumps(sech, ensure_ascii=False); exit() # 获取所有数据

    #print json.dumps(es_obj.getDataById(1111), ensure_ascii=False); exit()
    q = query()
    q.queryBuilder(queryMap('id', '66', 'must'))
    q.queryBuilder(queryMap('company', 'e成', 'must'))
    print json.dumps((es_obj.getDataByQuery(q)), ensure_ascii=False)
