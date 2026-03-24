#!/bin/bash

for i in {1..10}   # 修改 ensemble 参数
do
  for kappa in 10 20 30   # 修改 kappa 参数
  do
    srun --gres=gpu:1 --cpus-per-task=1 --nodelist=gpu2 \
         --job-name="kick_${i}_kappa_${kappa}" kickModel \
         --an 64 --tn 128 --l 0.1 --s 1.0 --kappa $kappa --ensemble $i > kick${i}_kappa${kappa}.txt &
  done
done


