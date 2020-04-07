#coding:utf8
import logging
import MySQLdb
import traceback, json
import sys
sys.path.append('../')
import pyrpc
from service_config import corp_tag_ip, func_tag_ip, ner_tag_ip
from request import Predictor

p = Predictor()

def getter_info(base_dict, base_key, default, dict = False):
    if base_key in base_dict:
        ret = str(base_dict[base_key])
        if ret.strip() == '':
            return default
        else:
            if dict:
                return base_dict[base_key]
            else:
                return ret
    else:
        return default

def getCorp(content):
    server_name = corp_tag_ip   # 开发：'http://192.168.1.204:51699'；测试：'http://10.9.10.10:51699'；线上：'http://algo.rpc'
    service_name = 'corp_tag'
    resMap = {}
    req_dict = {
            "header": {
                },
            "request": {
                "p": {
                    "corp_name": content    #"阿里巴巴"
                    },
                "c": "corp_tag_simple",
                "m": "corp_tag_by_name"
                }
            }
    client = pyrpc.HttpRpcClient(server_name)
    response = client.call_service(service_name, json.dumps(req_dict))
    try:
        company_id = json.loads(response)['response']['result']['company_id']
        first_trade_list = json.loads(response)['response']['result']['first_trade_list']
        resMap['company_id'] = company_id
        resMap['first_trade_list'] = first_trade_list
        #print company_id, first_trade_list
    except Exception, e:
        logging.warn('getCorp_error=%s' % (str(repr(e))))
    return resMap

def getFunc(content):
    server_name = func_tag_ip   # 开发：'http://192.168.1.204:51677'；测试：'http://testing2.nlp.rpc'；线上：'http://nlp.rpc'
    service_name = 'function_tag_for_edps'
    results = []
    req_dict = {
            "header": {
                "log_id": "0x666",
                "user_ip": "192.168.8.52",
                "uid": "0x666",
                "product_name": "algo_survey",
                "provider": "algo_survey"
                },
            "request": {
                "p": {
                    #"content": "大客户经理-500强客户（企业移动云产品，月入2万）",
                    "content": content,
                    "id": "12"
                    },
                "c": "edps_query_tag",
                "m": "get_edps_query_tags"
                }
            }
    client = pyrpc.HttpRpcClient(server_name)
    response = client.call_service(service_name, json.dumps(req_dict))
    try:
        should = json.loads(response)['response']['results']['should']
        must = json.loads(response)['response']['results']['must']
        for sh in should:
            results.append(sh)
        for sh in must:
            results.append(sh)
    except Exception, e:
        logging.warn('getFunc_error=%s' % (str(repr(e))))
    return results

def getNerRes(text):
    server_name = ner_tag_ip    # 开发：'http://192.168.1.111:51631'；测试：'http://10.9.10.23:51631'；线上：'http://algo.rpc'
    service_name = 'echeng_ner_server'
    entity_type = ['IMPORTANT_POSITION', 'IMPORTANT_INDUSTRY', 'IMPORTANT_CORPORATION', 'ENTITY']     # 职位，行业，公司, 实体
    nerRes = {}
    req_dict = {
            'header': {
                'product_name': 'algo.echeng_ner_server',
                'uid': '',
                'session_id': '',
                'user_ip': '',
                'local_ip': '127.0.0.1',
                'log_id': '123456'
                },
            'request': {
                'c': 'echeng_ner_server',
                'm': 'echeng_ner_server',
                'p': {
                    'Strings': text
                    }
                }
            }
    client = pyrpc.HttpRpcClient(server_name)
    response = client.call_service(service_name, json.dumps(req_dict))
    #print 'response : ', response       #json.dumps(response, ensure_ascii=False)
    try:
        results = json.loads(response)['response']['results']
        for res in results:
            entities = getter_info(res, 'entities', None, True)
            if entities is None: continue
            for en in entities:
                en_type = en['type']
                en_word = en['word']
                if en_type in entity_type:
                    if en_type not in nerRes:
                        nerRes[en_type] = []
                    nerRes[en_type].append(en_word)
        # 实体扩充行业和职位
        if 'ENTITY' in nerRes:
            if 'IMPORTANT_POSITION' not in nerRes:
                nerRes['IMPORTANT_POSITION'] = []
            if 'IMPORTANT_INDUSTRY' not in nerRes:
                nerRes['IMPORTANT_INDUSTRY'] = []
            for e in nerRes['ENTITY']:
                nerRes['IMPORTANT_POSITION'].append(e)
                nerRes['IMPORTANT_INDUSTRY'].append(e)

    except Exception, e:
        logging.warn('getNerRes_error=%s' % (str(repr(e))))
    return nerRes

def getSentenceLbel(sentence, topk=1):
    res = None
    try:
        if sentence.strip() == '':
            return None
        res = p.predict(sentence, topk)
    except Exception, e:
        logging.warn('getSentenceLbel_error=%s' % (str(repr(e))))
    return res

if __name__ == '__main__':
    '''
    server_name = 'http://algo.rpc'         # 线上
    server_name = 'http://testing2.algo.rpc'         # 测试
    service_name = 'echeng_ner_server'
    '''
    st = "熟悉互联网行业机器学习、深度学习等相关技术 ，担任过产品经理职位，阿里巴巴，百度相关工作经验"
    st = "数据挖掘需要什么技术栈"
    #print json.dumps(getNerRes(st), ensure_ascii=False); exit()   # 实体识别
    #print json.dumps(getFunc('产品经理'), ensure_ascii=False); exit()       # 职位到职能的识别
    #print json.dumps(getCorp('阿里巴巴'), ensure_ascii=False); exit()         # 公司及id识别

    print 'corp_tag_ip -> ', corp_tag_ip, ',\tfunc_tag_ip -> ', func_tag_ip, ',\tner_tag_ip -> ', ner_tag_ip
    #print json.dumps(getNerRes(st), ensure_ascii=False)   # 实体识别
    #print json.dumps(getFunc('产品经理'), ensure_ascii=False)       # 职位到职能的识别
    #print json.dumps(getCorp('阿里巴巴'), ensure_ascii=False)         # 公司及id识别
    print json.dumps(getSentenceLbel('笔试怎么准备',3), ensure_ascii=False)         # 句子标签识别


