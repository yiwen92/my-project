#!/usr/bin/env
# -*- coding: utf-8 -*-

from python3_gearman import GearmanClient
import time
import types
import json
import types
import msgpack
import logging
import logging
import sys
import os
import multiprocessing
import traceback

cv_client = GearmanClient(['192.168.8.39:4731'])

def get_cv_info(cv_id, corp_map=None, corp_client=None):
    global cv_client
    """
    get the work experice, eg: corporation_name, position infomation
    """
    skip = int(len(cv_id) / 20+0.5)
    num = 0
    for i in range(0,skip+1):
        req_dict = {
            'header' : {
                'uid': '1',
                'uname': '',
                'version': '1',
                'signid': '2132',
                'provider': '',
                'ip': '1232321',
                'product_name': 'icsearch',
                'appid': '123'
            },
            'request' : {
                'w': 'icdc_basic',
                'c': 'resumes/logic_resume',
                'm': 'get_multi_all',
                'p': {
                    'ids': cv_id[i*20:(i+1)*20],
                    # 'selected': 'basic,work,project'
                    'selected': ''
                }

            }
        }

        current_request = cv_client.submit_job("icdc_basic",json.dumps(req_dict, ensure_ascii=False))
        result_dict = json.loads(current_request.result)

        #print(json.dumps(result_dict, ensure_ascii=False))
        return result_dict

def get_cv_dict(cv_id):
    results = get_cv_info([cv_id])
    cv_dict = results["response"]["results"][str(cv_id)]
    #print(json.dumps(cv_dict, ensure_ascii=False))
    return cv_dict

if __name__ == '__main__':


    try:
        cv_id = sys.argv[1]
    except:
        cv_id = 163
    #get_cv_info([163, 221])
    print(json.dumps(get_cv_dict(cv_id), ensure_ascii=False))

