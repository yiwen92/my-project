# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
import multiprocessing

import gearman

from .base import BaseRpcServer, BaseRpcClient, BaseRpcContext, RpcError, ServerStatus


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
