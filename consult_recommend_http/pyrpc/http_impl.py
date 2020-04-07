# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
import threading
import multiprocessing

import six
import tornado.web
import tornado.wsgi
import tornado.ioloop
import tornado.httpserver
import tornado.httpclient

from .base import BaseRpcServer, BaseRpcClient, BaseRpcContext, RpcError, ServerStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _RpcHttpHandler(tornado.web.RequestHandler):
    def initialize(self, rpc_server, service_name):
        self.rpc_server = rpc_server
        self.service_name = service_name

    def post(self):
        try:
            request = self.request.body
            context = BaseRpcContext(rpc_server=self.rpc_server,
                                     service_name=self.service_name)
            response = self.rpc_server.on_service_execute(request, context)
            self.write(response)
            self.set_header("Content-Type", "application/octet-stream")
            self.finish()
        except Exception as ex:
            logger.error('_RpcHttpHandler post error: %s' % ex)
            self.send_error(500, message='_RpcHttpHandler post error: %s' % ex)


class _HealthCheckHandler(tornado.web.RequestHandler):
    def initialize(self, rpc_server, service_name):
        self.rpc_server = rpc_server
        self.service_name = service_name

    def get(self):
        response = {"name": self.service_name, "status": "200"}
        self.write(response)
        self.set_header("Content-Type", "application/json")
        self.finish()


class HttpRpcServer(BaseRpcServer):
    def __init__(self, port, host=None, mode='reuse', **kwargs):
        if not isinstance(port, six.integer_types) or port <= 0:
            raise ValueError('Invalid port: %s' % port)
        super(HttpRpcServer, self).__init__(**kwargs)
        self.port = port
        self.host = host if host else 'http://localhost:%d' % self.port
        if mode not in ['reuse', 'fork']:
            raise ValueError('Invalid mode: %s' % mode)
        self.mode = mode
        self.app_handlers = []
        self.app = None
        self.http_server = None
        self.workers = []

    def _start_func(self, port, num_workers):

        self.http_server.bind(port)
        self.http_server.start(num_workers)
        for name, (_, init) in six.iteritems(self.service_mappings):
            if init is not None:
                init()
        tornado.ioloop.IOLoop.current().start()

    def reuse_start(self):
        if six.PY3:
            from tornado.platform.asyncio import AnyThreadEventLoopPolicy
            import asyncio
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        th = threading.Thread(target=self._start_func,
                              args=(self.port, self.num_workers))
        th.start()

    def reuse_stop(self, grace):
        pass

    def fork_start(self):
        for i in xrange(self.num_workers):
            p = multiprocessing.Process(target=self._start_func,
                                        args=(self.port + i, 1))
            p.start()
            self.workers.append(p)

    def fork_stop(self):
        # TODO: 
        pass

    def start(self):
        if self.status == ServerStatus.STARTED:
            raise Exception('HttpServer %s is already started.' % self)
        super(HttpRpcServer, self).start()
        for name, (handler, init) in six.iteritems(self.service_mappings):
            service_name = name
            name = name.rstrip('/')
            if not name.startswith('/'):
                name = '/' + name
            self.app_handlers.append((r'%s/?' % name, _RpcHttpHandler,
                                      dict(rpc_server=self,
                                           service_name=service_name)))
            self.app_handlers.append((r'%s/%s/?' % (name, "health_check"),
                                      _HealthCheckHandler,
                                      dict(rpc_server=self,
                                           service_name=service_name)))
        self.app = tornado.wsgi.WSGIAdapter(
            tornado.web.Application(self.app_handlers))
        self.http_server = tornado.httpserver.HTTPServer(
            tornado.wsgi.WSGIContainer(self.app), xheaders=True)
        if self.mode == 'reuse':
            self.reuse_start()
        elif self.mode == 'fork':
            self.fork_start()

    def stop(self, grace=0.):
        if self.mode == 'reuse':
            self.reuse_stop(grace)
        elif self.mode == 'fork':
            self.fork_stop(grace)


class HttpRpcClient(BaseRpcClient):
    def __init__(self, server_host):
        super(HttpRpcClient, self).__init__()
        self.server_host = server_host
        self.client = tornado.httpclient.HTTPClient()

    def call_service(self, service_name, request, **kwargs):
        service_url = '%s/%s' % (self.server_host, service_name)
        headers = {'Content-Type': 'application/octet-stream'}
        response = self.client.fetch(service_url, method='POST', body=request,
                                     headers=headers, **kwargs)
        if response.code != 200:
            raise RpcError('HttpClient fetch data failed, service_url: %s'
                           % service_url)
        return response.body

    def close(self):
        if self.client is not None:
            self.client.close()
