# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import pyrpc
import json

def get_user_preferences(server_name, service_name, cid, content, query):

    req_dict={
            'header':{},
            'request':{
                'c':'',
                'm':query,
                'p':{
                    'id': cid,
                    'sql': content,
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

    server_name = 'http://127.0.0.1:51648'
    service_name = 'write_sql'
    strSQL = '{"20": [{"satisfaction": 50, "orientation": 1, "vitality": 3}, {"satisfaction": 8, "orientation": 2, "vitality": 5}, {"cv_function": "4038975:0.93,3000693:0. 63"}, {"cv_industry": "27,28,414,3,2,5"}, {"cv_school": "102269"}, {"cv_discipline": "1130310"}]}'
    print 'respond\n', get_user_preferences(server_name, service_name, 220, strSQL, 'query1')

