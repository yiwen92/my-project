# -*- coding: utf-8 -*-
import logging
import sys
sys.path.append('../')
import pyrpc
import json, traceback, re, datetime
reload(sys)
sys.setdefaultencoding('utf8')

#server_name = 'http://testing2.icdc.rpc'   # 测试
server_name = 'http://icdc.rpc'   # 线上
service_name = 'icdc_basic'

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

try: phone = sys.argv[1]
except: phone = 11111111

def get_cv_by_phone(phone):
    req_dict={
            'header':{
                'uname': 'zhanglong',
                'local_ip': '127.0.0.1'
                },
            'request':{
                'w': 'icdc_basic',
                'c':'resumes/logic_resume',
                'm':'get_cv_by_phone',
                'p':{
                    'phone': phone
                    }
                }
            }
    client = pyrpc.HttpRpcClient(server_name)
    response = client.call_service(service_name, json.dumps(req_dict))
    return response

def get_cv_by_id(ids):
    req_dict={
            'header':{
                'uname': 'zhanglong',
                'local_ip': '127.0.0.1'
                },
            'request':{
                'w': 'icdc_basic',
                'c':'resumes/logic_resume',
                'm':'get_multi_all',
                'p':{
                    'ids': ids  #[12345]
                    }
                }
            }
    client = pyrpc.HttpRpcClient(server_name)
    response = client.call_service(service_name, json.dumps(req_dict))
    return response

def getter_info(base_dict, base_key, default, dict = False):
    if base_dict is None:
        return default
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

def format_time(time_str, so_far='n'):
    try:
        if so_far.strip().lower() == 'y':
            return datetime.datetime.now()
        else:
            y = time_str.split('年')[0]
            m = time_str.split('年')[1].split('月')[0]
            s = datetime.datetime(int(y),int(m),1)
            return s
    except:
        return None

def get_sorted_keys(content_dict):
    if type(content_dict) != type({}):
        return []
    workid2time = {}
    for k, v in content_dict.items():
        start_time = None
        if 'start_time' in v:
            start_time = v['start_time']
        if start_time and k not in workid2time:
            ft = format_time(start_time)
            if ft:
                workid2time[k] = [ft]
    if workid2time:
        sorted_work_time = sorted(workid2time.iteritems(), key=lambda d:d[1][0])
    else: sorted_work_time = []
    return sorted_work_time

def resolve_cv(cv_id):
    res = {}
    try:
        cv_str = get_cv_by_id(cv_id)
        cv_dict = json.loads(cv_str)
        response = getter_info(cv_dict, 'response', None, True)
        results = getter_info(response, 'results', None, True)
        cv = getter_info(results, str(cv_id), None, True)
        #cv = getter_info(results, cv_id, None, True)
        algorithm = getter_info(cv, 'algorithm', None, True)
        education = getter_info(cv, 'education', None, True)
        edu_key_sorted = get_sorted_keys(education)
        basic = getter_info(cv, 'basic', None, True)
        industry_ids = getter_info(basic, 'industry_ids', None)
        cv_tag = getter_info(algorithm, 'cv_tag', None)
        cv_education = getter_info(algorithm, 'cv_education', None)
        if cv_tag:  # 职能 should/must
            cv_tag_dict = json.loads(cv_tag)
            tmp = []
            for k, v in cv_tag_dict.items():
                should = getter_info(v, 'should', None, True)
                must = getter_info(v, 'must', None, True)
                if should:
                    for e in should:
                        tmp.append(e)
                if must:
                    for e in must:
                        tmp.append(e)
            if tmp and 'cv_function' not in res:
                res['cv_function'] = []
            for e in list(set(tmp)):
                res['cv_function'].append(e)
        if industry_ids and 'cv_industry' not in res:   # 行业id
            res['cv_industry'] = industry_ids
        if edu_key_sorted:      # 学校和专业id
            edu_dict = getter_info(json.loads(cv_education), edu_key_sorted[-1][0], None, True)
            school_id = getter_info(edu_dict, 'school_id', None)
            major_id = getter_info(edu_dict, 'major_id', None)
            if school_id:
                res['cv_school'] = school_id
            if major_id:
                res['cv_discipline'] = major_id
    except:
        tb = traceback.format_exc(); print tb
        pass
    #print res
    return res

if __name__ == '__main__':
    import gc
    gc.disable()

    ids =  22558537

    print json.dumps(resolve_cv(ids), ensure_ascii=False)

    #res = get_cv_by_phone(phone); print 'server_name : ', server_name, res
    #res = get_cv_by_id(ids); print 'server_name : ', server_name, res

