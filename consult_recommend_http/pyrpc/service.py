# -*- coding: utf-8 -*-

from __future__ import absolute_import

import six
import functools
import enum
import msgpack
import json
import traceback
from . import utils


@enum.unique
class SerializationType(enum.Enum):
    NONE = 0
    MSGPACK = 1
    JSON = 2


def msgpack_or_json_deserializer(data, context):
    try:
        res = msgpack.unpackb(data)
        return res, SerializationType.MSGPACK
    except:
        res = json.loads(data)
        return res, SerializationType.JSON


def msgpack_or_json_serializer(data, context,
                               ser_type=SerializationType.MSGPACK):
    if ser_type == SerializationType.JSON:
        return json.dumps(data)
    else:
        return msgpack.packb(data)


def json_serializer(data, context):
    return json.dumps(data)


def json_deserializer(data, context):
    return json.loads(data)


def default_rpc_method_name(request, context):
    c, m = request['request']['c'], request['request']['m']


def rpc_handler(handler=None, name=None):
    if handler is None:
        return functools.partial(rpc_handler, name=name)
    handler.__rpc_name__ = name or handler.__name__
    return handler


class ServiceMetaClass(type):
    def __new__(cls, name, bases, attrs):
        rpc_handlers = {}
        for key, value in six.iteritems(attrs):
            rpc_name = getattr(value, '__rpc_name__', None)
            if rpc_name is not None:
                rpc_handlers[rpc_name] = value
        attrs['__rpc_handlers__'] = rpc_handlers
        return super(ServiceMetaClass, cls).__new__(cls, name, bases, attrs)


class BaseService(six.with_metaclass(ServiceMetaClass)):
    def __init__(self, service_name=None,
                 request_deserializer=None,
                 response_serializer=None,
                 logger=None, monitor_logger=None):
        self.service_name = service_name or self.__class__.__name__
        self.request_deserializer = request_deserializer
        self.response_serializer = response_serializer
        self.logger = logger
        self.monitor_logger = monitor_logger
        self.handler_mappings = {}
        for name, handler in six.iteritems(self.__rpc_handlers__):
            handler = getattr(self, handler.__name__)
            self.handler_mappings['/%s' % name] = handler

    def initialize(self):
        pass

    def __call__(self, request, context):
        try:
            if self.request_deserializer is not None:
                request = self.request_deserializer(request)
            handler_name = default_rpc_method_name(request, context)
            handler = self.handler_mappings[handler_name]
            response = handler(request, context)
        except Exception as ex:
            tb = traceback.format_exc()
            # TODO:

    def add_service(self, service):
        pass

    def info_log(self, *args, **kwargs):
        if self.logger:
            self.logger.info(*args, **kwargs)

    def warn_log(self, *args, **kwargs):
        if self.logger:
            self.logger.warn(*args, **kwargs)

    def error_log(self, *args, **kwargs):
        if self.logger:
            self.logger.error('')

    def mon_log(self, *args, **kwargs):
        if self.monitor_logger:
            self.monitor_logger.info(*args, **kwargs)


class ServiceDispatcher(BaseService):
    def __init__(self, **kwargs):
        super(ServiceDispatcher, self).__init__(**kwargs)
