import tornado, json, sys
from tornado.httpclient import HTTPClient

#url = "http://%s:%s/%s" % ("127.0.0.1", "1111", "semantic")
url = "http://%s:%s/%s" % ("192.168.8.52", "1111", "semantic")
http_client = HTTPClient()

try: txt = sys.argv[1]
except: txt = "我喜欢吃苹果##我喜欢吃香蕉"  # "深度学习极大地推广了图像的安防##人工智能算法在自然语言处理领域的应用"

def get_semantic(txt):
    obj = {"header": {},"request": {"c": "", "m": "semantic", "p": {"text": txt}}}
    response = http_client.fetch(tornado.httpclient.HTTPRequest(
        url=url,
        method="POST",
        headers={'Content-type': 'application/json'},
        body=json.dumps(obj, ensure_ascii=False)
    ))
    r = json.loads(response.buffer.read().decode("utf-8", errors='ignore'))
    return r['response']['results']['semantic_vector'][0]

def get_similar(txt1, txt2):
    vec1 = get_semantic(txt1)
    vec2 = get_semantic(txt2)   #; print(vec1); print(vec2)
    vec_dot_sum, vec1_size, vec2_size = 0.0, 0.0, 0.0
    for i in range(len(vec1)):
        vec_dot_sum += vec1[i] * vec2[i]
        vec1_size += vec1[i] * vec1[i]
        vec2_size += vec2[i] * vec2[i]
    #print(vec_dot_sum, vec1_size ** 0.5, vec2_size ** 0.5)
    sim = vec_dot_sum / ((vec1_size ** 0.5) * (vec2_size ** 0.5))
    return sim

if __name__ == "__main__":
    seg_txt = txt.split("##"); print(seg_txt); #exit()
    #res = get_semantic(txt); print(json.dumps(res, ensure_ascii=False))
    print(get_similar(seg_txt[0], seg_txt[1]))
    #print(8**0.5)


