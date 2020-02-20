#!/bin/bash

#SCRIPT_NAME=handle_data.py
#SCRIPT_NAME=data_utils.py
SCRIPT_NAME=train.py

echo "run file: " $SCRIPT_NAME
LOG_NAME=${SCRIPT_NAME}".log"

ps -ef | grep $SCRIPT_NAME | grep -v grep | awk '{print $2}' | xargs kill -9

nohup /opt/userhome/kdd_zouning/anaconda2/envs/python35/bin/python $SCRIPT_NAME > $LOG_NAME 2>&1&
tail -f $LOG_NAME
