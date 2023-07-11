#!/bin/bash

export HF_DATASETS_CACHE="/home/yoach/.cache/datasets"
export TRANSFORMERS_CACHE="/home/yoach/.cache/transformers"



# Define the models, batch sizes, and precisions
# "generated_assistant"
# "torch.bfloat16"
models=("ylacombe/bark-small" "ylacombe/bark-large")
optimization_types=("no_optimization" "flash_attention")
precisions=("torch.float32" "torch.float16" )

# Loop through each combination and execute the python command
for model_name in "${models[@]}"
do
  for precision in "${precisions[@]}"
  do
    for optimization_type in "${optimization_types[@]}"
    do
      CUDA_VISIBLE_DEVICES=3 /home/yoach/transformers/transformers-env/bin/python /home/yoach/bark_optimization/main.py --num_runs 1 --num_samples 50 --model_path "$model_name" --optimization_type "$optimization_type" --precision "$precision"
    done
  done
done