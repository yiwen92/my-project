import logging, json, requests
#zouning query weight client
class QueryWeightRewriter:
    def __init__(self):
        self.server_url = 'http://127.0.0.1:51658/query_weight'
        #self.server_url = 'http://algo.rpc/query_weight'

    def weight(self, query_tags):
        result = {}
        term_index = [(item.split(':')[1], item.split(':')[0]) for item in query_tags]
        sorted_term_index = sorted(term_index, key=lambda d: d[1])
        origi_query = " ".join([k for k, v in sorted_term_index])
        weight_res = self.weight_client(origi_query)
        if weight_res: result = weight_res
        return result

    def weight_client(self, query):
        try:
            request = {"header": {}, "request": {"c": "", "m": "query_weight", "p": {"query": query}}}
            logging.info('weight rpc request %s' %(request))
            headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
            resp = requests.post(self.server_url, data=json.dumps(request), headers=headers, timeout=0.1)
            resp = json.loads(resp.text)
            if 'response' in resp and 'results' in resp['response']:
                return resp.get('response').get('results')
        except Exception as ex:
            logging.warn('Rewrite query weight exception, query_tags=%s, err=%s' % ( str(query), repr(ex)))
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fr = QueryWeightRewriter()
    test_query_tags = {'0:' + "java" : {'FUNCTION': 0.7}, '2:' + "上海" : {'LOCATION': 0.1}, '1:' + "开发工程师" : {'KEYWORD': 0.2}}
    resp = fr.weight(test_query_tags)
    print(json.dumps(resp,ensure_ascii=False))