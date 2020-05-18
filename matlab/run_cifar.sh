#!/bin/bash

nohup matlab -nodesktop -nosplash -r  "gen_gauss_dataset()" -logfile cifar-log.txt >/dev/null 2>&1 &
