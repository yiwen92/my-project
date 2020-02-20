#!/bin/bash

#SCRIPT_NAME=handle_data.py
#SCRIPT_NAME=data_utils.py
SCRIPT_NAME=train.py

ps aux | grep "${SCRIPT_NAME}" | grep -v "grep" | awk '{print $2}' | xargs kill -9
