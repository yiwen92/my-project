# -*- coding: utf-8 -*-
import logging
import sys
sys.path.append('../')
import pyrpc
import json

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

def get_user_preferences(server_name, service_name, qesStr):

    req_dict={
            'header':{},
            'request':{
                'c':'',
                'm':'consult_recommend',
                'p':{
                    'consult_question': qesStr
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

    server_name = 'http://127.0.0.1:51645'     # 测试
    service_name = 'test_model_http'

    qesStr = "行政助理或文员秘书类的职位都是做什么的？"
    res = get_user_preferences(server_name, service_name, qesStr)
    print 'server_name : ', server_name, '\nrespond\n', res

