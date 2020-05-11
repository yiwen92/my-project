from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import json
from semantic import semantic_model

sm = semantic_model()   #; sm.test(); exit()

class Handler(RequestHandler):
    def post(self):
        text = json.loads(self.request.body.decode('utf-8'))["request"]["p"]["text"]
        self.set_header('Content-Type', 'application/json')
        vector = sm.run_step(text)  #;print(type(vector[0][0])); #exit()
        res = json.dumps({"header": {}, "response": {"err_no": "0", "err_msg": "", "results": {"semantic_vector": vector}}}, ensure_ascii=False)
        self.write(res)

if __name__ == '__main__':
    app = Application([(r'/semantic', Handler)], debug=False)
    http_server = HTTPServer(app)
    #app.listen(1111)
    http_server.bind(1111)
    http_server.start(1)
    IOLoop.current().start()
