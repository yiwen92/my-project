# -*- coding: UTF-8 -*-
from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import json, logging, logging.config, chardet
from query_doc_sim import queryDocSim

qds = queryDocSim()

log_conf_file = 'log4ic.conf'
logging.config.fileConfig(log_conf_file)

class Handler(RequestHandler):
    def post(self):
        try:
            encoding = chardet.detect(self.request.body)
            encode_type = encoding.get("encoding", "utf-8")
            req_body = self.request.body.decode(encode_type)
            req_dict = json.loads(req_body)
            self.set_header('Content-Type', 'application/json')
            similarity, prob_res = qds.cal_sim(req_dict)
            res = json.dumps({"header": {}, "response": {"err_no": "0", "err_msg": "", \
                            "results": {"similarity": similarity, "prob_res": prob_res}}}, ensure_ascii=False)
            self.write(res)
        except Exception as e:
            logging.warn('__post_failed, req=%s, exception=[%s]' % (json.dumps(req_dict,ensure_ascii=False), str(e)))

if __name__ == '__main__':
    numworkers = 1
    app = Application([(r'/query_doc_sim', Handler)], debug=False)
    http_server = HTTPServer(app)
    http_server.bind(51688)
    http_server.start(numworkers)
    logging.info('__query_doc_similariry_server_running__ num_workers: %s' % numworkers)
    IOLoop.current().start()