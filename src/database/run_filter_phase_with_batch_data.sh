#!/bin/bash

# Read the first argument
ARG1=$1


cd ./
source init_env

python exps/main_v2/database/run_filter_phase.py --batch_data="$ARG1"
#python run_filter_phase.py


