#coding:utf8
import os,sys,logging,time,re,json,requests
sys.path.append('../')
import pyrpc
#hukai query expand client
class QueryExpandRewriter:
    def __init__(self):
        self.server_url = 'http://algo.rpc/querymining'

    def expand(self, query):
        result = {}
        expand_str = self.expand_client(query)
        if expand_str is not None:
            result[query] = expand_str
        return result

    def expand_client(self,query):
        try:
            request = '{"header":{"uname":"JK_testing"},"request":{"m":"expand","p":{"query":"'+query+'"}}}'
            logging.info('expand rpc request %s' %(request))
            resp = requests.post(url=self.server_url, data=request.encode(), headers={'Content-Type':'application/json','Accept-Charset':'UTF-8'}, timeout=0.1)
            resp = resp.json()
            logging.info('response from expand %s'%(resp))
            if 'expand' in resp \
                and resp.get('expand') is not None \
                and len(list(resp.get('expand'))) >= 1 :
                expand_dict = resp.get('expand')
                res = {}
                for expand_key in expand_dict:
                    res[expand_key] = float(round(expand_dict.get(expand_key),3))
                return res
        except Exception as ex:
            logging.warn('Rewrite query expand exception,query_tags=%s,err=%s' % ( str(query),repr(ex)))

        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fr = QueryExpandRewriter()
    #term = sys.argv[1]
    #term = "海外营销 java ios"
    term = "上海 产品经理"
    test_query_tags = { '0:' + term : {'FUNCTION': 0.99}}
    #resp = fr.expand(test_query_tags)
    resp = fr.expand(term)
    print(json.dumps(resp,ensure_ascii=False))

