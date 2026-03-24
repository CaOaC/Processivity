#!/bin/bash

for i in {1..1}; do
  # 使用 sbatch 提交任务，传递不同的种子
  sbatch --job-name=mitotic_$i \
         --gpus=1 \
         --cpus-per-task=4 \
         --output=output_$i.log \
         --error=error_$i.log \
         --wrap="./kickModel --ensemble $i --seed $RANDOM"
done


