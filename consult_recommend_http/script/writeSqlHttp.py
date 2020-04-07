#!/usr/bin/env
# -*- coding: utf-8 -*-
import sys
import json
import traceback
from utils import updateSQL
sys.path.append('../')
import pyrpc

reload(sys)
sys.setdefaultencoding('utf8')

def assemble_response(header, err_no, err_msg, results):
    response = {}
    response["err_no"] = str(err_no)
    response["err_msg"] = err_msg
    response["results"] = results
    ret = {}
    if header == None:
        ret["header"] = {}
        ret["header"]["uid"] = ""
        ret["header"]["local_ip"] = "127.0.0.1"
        ret["header"]["user_ip"] = ""
        ret["header"]["session_id"] = ""
        ret["header"]["product_name"] = "consult_recommend"
        ret["header"]["log_id"] = "123456"
    else:
        ret["header"] = header

    ret["response"] = response
    return json.dumps(ret,ensure_ascii=False)

def task_callback(request, context):
    try:
        try:
            req_dict = json.loads(request)
        except Exception, e:
            req_dict = {}

        sqlStr = req_dict['request']['p']['sql']
        cid = req_dict['request']['p']['id']
        query = req_dict['request']['m']
        if query == 'query':
            retMap = {}
            result = updateSQL(cid, query = True)
            if result:
                retMap['id'] = result[0][0]; retMap['content'] = result[0][1]
                ret = assemble_response(req_dict['header'],0,retMap,'query sucsess')
            else:
                ret = assemble_response(req_dict['header'],0,'No data','query fail')
        else:
            result = updateSQL(cid, sqlStr, False, True)
            #print 'result', json.dumps(result, ensure_ascii=False);exit()
            eid,emsg = [0,""] if result else [4,"computing failed, possible due to sparse information"]
            ret = assemble_response(req_dict['header'],eid,emsg,'updata sucsess')
    except Exception, e:
        tb = traceback.format_exc(); print 'traceback:%s' % str(tb)
        ret = assemble_response(req_dict['header'],3,"exception",None)
    return ret

class MyRpcServer(pyrpc.HttpRpcServer):
    def on_service_complete(self, request, context, response):
        logging.info("on_service_complete")
        return super(MyRpcServer, self).on_service_complete(
                request, context, response)

if __name__== "__main__":

    numworkers = 1
    server = pyrpc.HttpRpcServer(port=51648, num_workers=numworkers)
    server.register_service('write_sql', task_callback)
    server.start()



