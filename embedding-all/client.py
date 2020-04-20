import tornado, json
from tornado.httpclient import HTTPClient
import requests

#curl -i -X POST -H 'Content-type':'application/json' -d '{"header":{},"request":{"c":"","m":"query_correct","p":{"query":"andio"}}}' http://192.168.9.140:1111/query_correct

url = "http://%s:%s/%s" % ("127.0.0.1", "51688", "query_doc_sim")

def get_res(query, doc):
    obj = {"header": {},"request": {"c": "", "m": "query_doc_sim", "p": {"query": query, "doc": doc}}}
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    response = requests.post(url, data=json.dumps(obj), headers=headers)
    result = json.loads(response.text)
    '''
    http_client = HTTPClient()
    response = http_client.fetch(tornado.httpclient.HTTPRequest(
        url=url,
        method="POST",
        headers={'Content-type': 'application/json'},
        body=json.dumps(obj, ensure_ascii=False)
    ))
    result = json.loads(response.buffer.read().decode("utf-8", errors='ignore'))
    http_client.close()
    '''
    return result

if __name__ == '__main__':
    que = "nlp开发工程师"
    text = "1、熟悉Python、C、C++、Matlab、java编程语言中的一种或多种；2、熟练使用Caffe、Tensorflow、PyTorch等开源深度学习框架中的一种或多种；\
    3、扎实的机器学习、深度学习，NLP理论基础，熟悉深度学习在自然语言处理以及领域知识图谱构建中的应用，实现切实有效的深度学习模型；\
    4、研究语义建模、自然语言理解、机器阅读理解、文本分类、搜索问答等NLP关键前沿技术的研发工作；\
    5、熟悉文本处理相关研发工作, 包括但不限于数据挖掘、情感分析、自然语言NLP处理、信息抽取、智能问答、知识图谱、语义理解、意图识别, 机器翻译, 机器写作等； \
    6、有扎实的数学基础，熟悉并能灵活运用高数、线性代数、概率论等数学知识；\
    7、良好的英文阅读能力和销售动力；\
    8、具备良好的逻辑思维能力与沟通表达能力。hr专员"
    print(json.dumps(get_res(que, text), ensure_ascii=False))

