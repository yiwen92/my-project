#!/bin/bash

SCRIPT_NAME=consultRecommendHttp.py
#SCRIPT_NAME=writeSqlHttp.py
#SCRIPT_NAME=test_model_http.py

ps -ef | grep $SCRIPT_NAME | grep -v grep | awk '{print $2}' | xargs kill -9

cd script
nohup python $SCRIPT_NAME &

