#!/usr/bin/env
# -*- coding: utf-8 -*-
import sys
import logging
import logging.config
import datetime
import json
import traceback
from request import Predictor

sys.path.append('../')
import pyrpc

reload(sys)
sys.setdefaultencoding('utf8')

log_conf_file = 'log4ic.conf'
logging.config.fileConfig(log_conf_file)

def check_response(f_request):
    if 'header' not in f_request or 'request' not in f_request:
        return False
    req_dict = f_request['request']
    if 'c' not in req_dict or 'm' not in req_dict or 'p' not in req_dict:
        return False
    method = req_dict['m'].strip().lower()
    if method == "" or method not in {'consult_recommend':'1'}:
        return False
    return True

def check_header(header):
    #后期通过此head字段来辨别调用方，区分服务流程
    return True

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

p = Predictor()

def task_callback(request, context):
    try:
        try:
            req_dict = json.loads(request)
        except Exception, e:
            logging.warn('Not normal json format, req=%s, exception=[%s]' % (str(request), str(e)))
            req_dict = {}
        time_request_in = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logging.debug('received request: %s' % str(req_dict))

        if check_response(req_dict) == False:
            if 'header' not in req_dict:
                ret = assemble_response(None,1,"no header",None)
            elif not check_header(req_dict['header']):
                ret = assemble_response(None,1,"invalid header",None)
            else:
                ret = assemble_response(None,2,"invalid request",None)
            return ret
        consult_question = req_dict['request']['p']['consult_question']
        result = p.predict(consult_question)    #; print json.dumps(result, ensure_ascii=False); exit()
        eid,emsg = [0,""] if len(result) > 0 else [4,"computing failed, possible due to sparse information"]
        ret = assemble_response(req_dict['header'],eid,emsg,result)
        logging.info('request = %s' % json.dumps(req_dict,ensure_ascii=False))
    except Exception, e:
        tb = traceback.format_exc() ; print tb
        logging.warn('__test_model_http_failed, req=%s, exception=[%s]' % (json.dumps(req_dict,ensure_ascii=False), str(e)))
        ret = assemble_response(req_dict['header'],3,"exception",None)
    return ret

class MyRpcServer(pyrpc.HttpRpcServer):
    def on_service_complete(self, request, context, response):
        logging.info("on_service_complete")
        return super(MyRpcServer, self).on_service_complete(
                request, context, response)

if __name__== "__main__":

    numworkers = 24
    server = pyrpc.HttpRpcServer(port=51645, num_workers=numworkers)
    server.register_service('test_model_http', task_callback)
    server.start()
    logging.info('__test_model_http_server_running__ num_workers: %s' % numworkers)



