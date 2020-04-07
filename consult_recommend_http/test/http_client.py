# -*- coding: utf-8 -*-
import logging
import sys
sys.path.append('../')
import pyrpc
import json

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

def get_user_preferences(server_name, service_name, qesCls, qesStr, adviser_blacklist, cnt, uid):

    req_dict={
            'header':{},
            'request':{
                'c':'',
                'm':'consult_recommend',
                'p':{
                    'consult_orientation': qesCls,
                    'consult_question': qesStr,
                    'adviser_blacklist': adviser_blacklist,
                    'count': cnt,
                    'uid': uid
                    }
                }
            }
    #print 'req_dict', json.dumps(req_dict, ensure_ascii=False)
    client = pyrpc.HttpRpcClient(server_name)
    response = client.call_service(service_name, json.dumps(req_dict))
    #print type(response)
    return response


if __name__ == '__main__':
    import gc
    gc.disable()

    server_name = 'http://127.0.0.1:51645'     # 本机
    #server_name = 'http://testing2.algo.rpc'   # 测试
    #server_name = 'http://algo.rpc'             # 线上

    service_name = 'consult_recommend'

#    '''
    qesCls = ''
    #qesStr = "熟悉互联网行业机器学习、深度学习等相关技术 ，担任过产品经理职位，阿里巴巴，百度相关工作经验"
    qesStr = ""
    adviser_blc = ''
    return_count = 5
    uid = 232
    res = get_user_preferences(server_name, service_name, qesCls, qesStr, adviser_blc, return_count, uid)
    print 'server_name : ', server_name
    print qesCls, qesStr, adviser_blc, '\nrespond\n', len(json.loads(res)['response']['results']['recommend_results']), res
    exit()
#    '''
    q = {}; q['求职应聘']=[]; q['面试辅导']=[]; q['入职准备']=[]; q['产品']=[]; q['运营']=[]; q['技术']=[]
    q['求职应聘'].append('C++后台开发 准备什么样的项目可以应对春招实习?'); q['求职应聘'].append('大三找实习该怎么准备？')
    q['求职应聘'].append('实习经历很少，我该怎么写我的简历？'); q['求职应聘'].append('数据挖掘需要学习什么栈')
    q['面试辅导'].append('下周要去腾讯地图事业部面试，面试流程是咋样的呀？')
    q['面试辅导'].append('杭州银行信息科技部笔试大概什么内容？'); q['面试辅导'].append('自我介绍应该怎么做才好？')
    q['入职准备'].append('想问一下，实习offer如果答应了后来不去会怎么样啊？');q['入职准备'].append('马上要入职携程机票业务部，不知道需要准备些什么？')
    q['入职准备'].append('应届生第一份工作，在平时需要注意一些什么？有没有职场潜规则？')
    q['产品'].append('想要做一名产品经理，我需要去学哪些技能或者课程？'); q['产品'].append('做了一年产品，不知道今后的发展方向是什么？')
    q['产品'].append('产品经理最重要的能力是什么？如何才能有效的锻炼这个能力？')
    q['运营'].append('我19年9月毕业， 没有运营方面的经历，但是对这个岗位很感兴趣，该怎么入行？')
    q['运营'].append('网上看到很多运营方面的工作机会，但是还没明白运营具体有哪些工作内容？'); q['运营'].append('运营菜鸟怎么才能向运营总监进阶？')
    q['技术'].append('想问一下大佬，学习LINUX内核，需要什么基础知识')
    q['技术'].append('C++后台开发 准备什么样的项目可以应对春招实习?'); q['技术'].append('请问现在应用最广泛的语言是什么？哪一种语言更好找工作')
    for k, v in q.items():
        for e in v:
            res = get_user_preferences(server_name, service_name, k, e, '', 3, 8)
            print json.dumps(k, ensure_ascii=False), json.dumps(e, ensure_ascii=False), res
        print '\n'

