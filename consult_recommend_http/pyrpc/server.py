# -*- coding: utf-8 -*-


from __future__ import absolute_import

import six
import abc
import enum
import logging
import multiprocessing
import threading
import functools
import gearman
import tornado.web
import tornado.wsgi
import tornado.ioloop
import tornado.httpserver
import sys
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@enum.unique
class ServerStatus(enum.Enum):
    STOPPED = 'stopped'
    STARTED = 'started'


class BaseRpcContext(object):
    def __init__(self, rpc_server, service_name):
        self.rpc_server = rpc_server
        self.service_name = service_name


class BaseRpcServer(six.with_metaclass(abc.ABCMeta)):
    def __init__(self, num_workers=1):
        if not isinstance(num_workers, six.integer_types) or num_workers < 0:
            raise ValueError('Invalid num_workers: %s' % num_workers)
        self.num_workers = num_workers
        self.status = ServerStatus.STOPPED
        self.service_mappings = {}

    @abc.abstractmethod
    def start(self):
        self.status = ServerStatus.STARTED

    @abc.abstractmethod
    def stop(self, grace=0):
        self.status = ServerStatus.STOPPED

    def register_service(self, name, handler, initializer=None,
                         args=(), kwargs={}):
        if not callable(handler):
            raise ValueError('handler must be callable.')
        if name in self.service_mappings:
            raise ValueError('%s is already in service_mappings.' % name)
        if initializer is not None:
            args = tuple(args)
            kwargs = None or {}
            initializer = functools.partial(initializer, *args, **kwargs)
        self.service_mappings[name] = (handler, initializer)

    def on_service_execute(self, request, context):
        try:
            handler = self.service_mappings[context.service_name][0]
            response = handler(request, context)
            self.on_service_complete(request, context, response)
            return response
        except Exception:
            exc_info = sys.exc_info()
            self.on_service_exception(request, context, exc_info)
            raise exc_info[1]

    def on_service_complete(self, request, context, response):
        return True

    def on_service_exception(self, request, context, exc_info):
        return False


class _GearmanWorkerProcess(multiprocessing.Process):
    def __init__(self, rpc_server, **kwargs):
        super(_GearmanWorkerProcess, self).__init__(**kwargs)
        self.rpc_server = rpc_server
        self.worker = None
        self.shutdown_event = multiprocessing.Event()

    def gm_task_callback(self, gm_worker, gm_job):
        request = gm_job.data
        context = BaseRpcContext(rpc_server=self.rpc_server,
                                 service_name=gm_job.task)
        response = self.rpc_server.on_service_execute(request, context)
        return response

    def _shutdown_thread(self):
        self.shutdown_event.wait()
        self.worker.shutdown()

    def run(self):
        threading.Thread(target=self._shutdown_thread).start()
        self.worker = gearman.GearmanWorker(self.rpc_server.host_list)
        for service_name, (_, initializer) in six.iteritems(
                self.rpc_server.service_mappings):
            if initializer is not None:
                initializer()
            self.worker.register_task(service_name, self.gm_task_callback)
        try:
            self.worker.work(5.0)
        except gearman.errors.ServerUnavailable:
            pass

    def shutdown(self):
        self.shutdown_event.set()


class GearmanRpcServer(BaseRpcServer):
    def __init__(self, host_list, **kwargs):
        super(GearmanRpcServer, self).__init__(**kwargs)
        self.host_list = host_list
        self.workers = []

    def start(self):
        if self.status == ServerStatus.STARTED:
            raise Exception('GearmanServer %s is already started.' % self)
        for i in six.moves.range(self.num_workers):
            worker = _GearmanWorkerProcess(rpc_server=self)
            worker.start()
            self.workers.append(worker)
        super(GearmanRpcServer, self).start()
        for worker in self.workers:
            worker.join()
        self.workers = []

    def stop(self, grace=0):
        if self.status != ServerStatus.STARTED:
            raise Exception('GearmanServer %s is not started.' % self)
        time.sleep(grace)
        for worker in self.workers:
            worker.shutdown()
        super(GearmanRpcServer, self).stop()


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
