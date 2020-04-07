# -*- coding: utf-8 -*-

import logging
import six
import abc
import gearman
import tornado.httpclient

logger = logging.getLogger(__name__)


class RpcError(Exception):
    pass


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


class GearmanRpcClient(BaseRpcClient):
    def __init__(self, host_list):
        super(GearmanRpcClient, self).__init__()
        self.host_list = host_list
        self.client = gearman.GearmanClient(host_list)

    def call_service(self, handler_name, request, **kwargs):
        job_request = self.client.submit_job(handler_name, request, **kwargs)
        if job_request.timed_out \
           or job_request.state != gearman.constants.JOB_COMPLETE:
            raise RpcError('GearmanClient submit job failed, handler_name: %s'
                           % handler_name)
        return job_request.result

    def close(self):
        pass


class HttpRpcClient(BaseRpcClient):
    def __init__(self, server_host):
        super(HttpRpcClient, self).__init__()
        self.server_host = server_host
        self.client = tornado.httpclient.HTTPClient()

    def call_service(self, service_name, request, **kwargs):
        service_url = '%s/%s' % (self.server_host, service_name)
        #print service_url, request
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

