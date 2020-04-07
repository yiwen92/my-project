# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
import sys
import functools

import six
import abc
import enum


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@enum.unique
class ServerStatus(enum.Enum):
    STOPPED = 'stopped'
    STARTED = 'started'


class RpcError(Exception):
    pass


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


class BaseRpcClient(six.with_metaclass(abc.ABCMeta)):
    def __init__(self):
        pass

    @abc.abstractmethod
    def call_service(self, handler_name, request, **kwargs):
        raise NotImplementedError()

    def close(self):
        pass

    def __del__(self):
        self.close()
