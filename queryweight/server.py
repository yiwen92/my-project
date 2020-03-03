# -*- coding: UTF-8 -*-
from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import json, logging, logging.config, re, chardet, tornado
from qw import query_weight

qw = query_weight(1000000)

log_conf_file = 'log4ic.conf'
logging.config.fileConfig(log_conf_file)

class Handler(RequestHandler):
    def post(self):
        try:
            #a={e:self.request.body.decode(e) for e in ['utf8','gbk','gb18030','ascii','gb2312']}
            encoding = chardet.detect(self.request.body)
            encode_type = encoding.get("encoding", "utf-8")
            req_body = self.request.body.decode(encode_type)
            query = json.loads(req_body)["request"]["p"]["query"]
            self.set_header('Content-Type', 'application/json') #; q=req_dict['request']['p']['query']
            r = qw.run_step(query)  #;print(json.dumps(r, ensure_ascii=False)); exit()
            res = json.dumps({"header": {}, "response": {"err_no": "0", "err_msg": "", "results": r}}, ensure_ascii=False)
            self.write(res.encode(encode_type))
        except Exception as e:
            logging.warn('__post_failed, req=%s, exception=[%s]' % (json.dumps(req_dict,ensure_ascii=False), str(e)))

if __name__ == '__main__':
    numworkers = 1
    app = Application([(r'/query_weight', Handler)], debug=False)
    http_server = HTTPServer(app)
    http_server.bind(51658)
    http_server.start(numworkers)
    logging.info('__query_weight_server_running__ num_workers: %s' % numworkers)
    IOLoop.current().start()
