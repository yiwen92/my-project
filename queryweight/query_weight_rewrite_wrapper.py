import logging, json, requests
#zouning query weight client
class QueryWeightRewriter:
    def __init__(self):
        self.server_url = 'http://127.0.0.1:51658/query_weight'
        #self.server_url = 'http://algo.rpc/query_weight'

    def weight(self, query):
        result = {}
        weight_res = self.weight_client(query)
        if weight_res is not None:
            result[query] = weight_res
        return result

    def weight_client(self, query):
        try:
            request = {"header": {}, "request": {"c": "", "m": "query_weight", "p": {"query": query}}}
            logging.info('weight rpc request %s' %(request))
            headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
            resp = requests.post(self.server_url, data=json.dumps(request), headers=headers, timeout=0.1)
            resp = resp.json()
            logging.info('response from weight %s' % (resp))
            if 'response' in resp and 'results' in resp['response']:
                return resp.get('response').get('results')
        except Exception as ex:
            logging.warn('Rewrite query weight exception, query_tags=%s, err=%s' % ( str(query), repr(ex)))
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fr = QueryWeightRewriter()
    test_query_tags = {'0:' + "java" : {'FUNCTION': 0.7}, '2:' + "上海" : {'LOCATION': 0.1}, '1:' + "开发工程师" : {'KEYWORD': 0.2}}
    term = "上海 产品经理"
    resp = fr.weight(term)
    print(json.dumps(resp,ensure_ascii=False))