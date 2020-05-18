#!/bin/bash



# KFAC-TR
python cifar10_str_batch_kfac.py --batch-size 200 --epochs 10 --model QAlexNetS --seed 1 --init xavier --damp=0.01
