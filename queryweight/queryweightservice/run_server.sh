#!/bin/bash

SCRIPT_NAME=server.py

#ps -ef | grep $SCRIPT_NAME | grep -v grep | awk '{print $2}' | xargs kill -9

python $SCRIPT_NAME >> run.log 2>&1
#nohup python $SCRIPT_NAME >> run.log 2>&1&
#nohup /opt/userhome/kdd_zouning/anaconda2/envs/python36/bin/python $SCRIPT_NAME >> run.log 2>&1&

