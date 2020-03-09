import tornado, json, sys
from tornado.httpclient import HTTPClient

#curl -i -X POST -H 'Content-type':'application/json' -d '{"header":{},"request":{"c":"","m":"query_correct","p":{"query":"andio"}}}' http://192.168.9.140:1111/query_correct

url = "http://%s:%s/%s" % ("127.0.0.1", "51658", "query_weight")
http_client = HTTPClient()

def get_res(txt):
    obj = {"header": {},"request": {"c": "", "m": "query_weight", "p": {"query": txt}}}
    response = http_client.fetch(tornado.httpclient.HTTPRequest(
        url=url,
        method="POST",
        headers={'Content-type': 'application/json'},
        body=json.dumps(obj, ensure_ascii=False)
    ))
    result = json.loads(response.buffer.read().decode("utf-8", errors='ignore'))
    http_client.close()
    return result

if __name__ == '__main__':
    try: que = sys.argv[1]
    except: que = "android开发工程师招聘"
    print(json.dumps(get_res(que), ensure_ascii=False))

