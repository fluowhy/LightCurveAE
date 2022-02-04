#!/bin/sh

declare -i gpu=1
declare -i epochs=2048

python main_od.py --dataset ztf_transient --device cuda:$gpu --oc 0 --e $epochs
python main_od.py --dataset ztf_stochastic --device cuda:$gpu --oc 0 --e $epochs
python main_od.py --dataset ztf_periodic --device cuda:$gpu --oc 0 --e $epochs