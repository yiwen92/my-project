# -*- coding: utf-8 -*-
import msgpack
from python3_gearman import GearmanClient
import json
import sys
import re
import logging
import time
import os

logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s",level=logging.INFO)

class JDFromGearman(object):
    def __init__(self,worker_name,worker_info):
        self.client = GearmanClient(worker_info)
        self.worker_name=worker_name

    def getJson(self, jid):
        msg = {
            'header':{
                'product_name':'algo_rec_test',
                'uid':'0x666',
                'log_id':'0x666'
                },
            'request':{
                'c':'positions/logic_position',
                'm':'read',
                'p':{
                'ids': [int(jid)],
                'selected': ""
                    }
            }

        }

        #try:
        resp = self.client.submit_job(self.worker_name, json.dumps(msg))

        res = json.loads(resp.result)
        #print(json.dumps(res, indent=4, ensure_ascii=False))
        #res["response"]["results"][str(jid)]["jd_comment"] = json.loads(res["response"]["results"][str(jid)]["jd_comment"])
        #print(json.dumps(res, ensure_ascii=False, indent="\t"))
        res = res["response"]["results"][str(jid)]
        return res


    def get(self,jid):
        msg = {
            'header':{
                'product_name':'algo_rec_test',
                'uid':'0x666',
                'log_id':'0x666'
                },
            'request':{
                'c':'positions/logic_position',
                'm':'read',
                'p':{
                'ids': [int(jid)],
                'selected': ""
                    }
            }

        }

        #try:
        resp = self.client.submit_job(self.worker_name, json.dumps(msg))

        res = json.loads(resp.result)
        print(json.dumps(res, indent=4, ensure_ascii=False))
        #res["response"]["results"][str(jid)]["jd_comment"] = json.loads(res["response"]["results"][str(jid)]["jd_comment"])
        #print(json.dumps(res, ensure_ascii=False, indent="\t"))
        res = res["response"]["results"][str(jid)]

        #print(json.dumps(res, ensure_ascii=False, indent="\t"))

        title = " ".join([s for s in re.split(r"[ _:：\-]", res["name"]) if not re.search(r"(公司$|集团$|^[0-9]+$)", s)])

        city = list(json.loads(res["city_ids"]).items())
        city = city[0][1] if city else ""

        desc  = res["description"]

        req = res["requirement"]

        try:
            comp = json.loads(res["jd_original_corporations"])[0]["company_info"] if "jd_original_corporations" in res else {}
            ckwds = comp["keyword"]if "keyword" in comp  else []
        except Exception as e:
            ckwds = []

        if not desc or not req:
            arr = sorted(re.split(r"(^|\n|\\n)[^ ]{2,5}[：:]", desc + req), key=lambda x: len(x), reverse=True)
            if len(arr) > 1:
                desc = arr[0]
                req = arr[1]

        return title, city, desc, req, ckwds
        #except Exception as e:
        #    pass
        return "", "", "", "",[]


if __name__=="__main__":
    obj=JDFromGearman("icdc_position_basic_online",["192.168.8.70:4730"])

    try:
        jd_id = sys.argv[1]
    except:
        jd_id = 22357064

    #print(obj.get(jd_id))
    print(json.dumps(obj.getJson(jd_id), ensure_ascii=False))


