#!/bin/sh

declare -i gpu=0
declare -i epochs=2048

python main_od.py --dataset asas_sn --device cuda:$gpu --oc 8 --e $epochs