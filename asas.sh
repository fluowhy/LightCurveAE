#!/bin/sh

declare -i gpu=3
declare -i epochs=2048

python main_od.py --dataset asas --device cuda:$gpu --oc 0 --e $epochs
python main_od.py --dataset asas --device cuda:$gpu --oc 1 --e $epochs
python main_od.py --dataset asas --device cuda:$gpu --oc 2 --e $epochs
python main_od.py --dataset asas --device cuda:$gpu --oc 3 --e $epochs
python main_od.py --dataset asas --device cuda:$gpu --oc 4 --e $epochs