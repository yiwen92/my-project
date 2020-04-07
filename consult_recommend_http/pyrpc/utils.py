# -*- coding: utf-8 -*-

from __future__ import absolute_import

import socket


def fully_qualified_method(service, method):
    return '/%s/%s' % (service, method)


def get_ip_address():
    my_name = socket.getfqdn(socket.gethostname())
    my_addr = socket.gethostbyname(my_name)
    return my_addr


LOCAL_IP = get_ip_address()

