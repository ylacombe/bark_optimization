#!/bin/bash

export HF_DATASETS_CACHE="/home/yoach/.cache/datasets"
export TRANSFORMERS_CACHE="/home/yoach/.cache/transformers"



# Define the models, batch sizes, and precisions
# "generated_assistant"
# "torch.bfloat16"
precisions=("torch.float32" "torch.float16")
temperatures=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Loop through each combination and execute the python command
for temperature in "${temperatures[@]}"
do
  for precision in "${precisions[@]}"
  do
    CUDA_VISIBLE_DEVICES=2 /home/yoach/transformers/transformers-env/bin/python /home/yoach/bark_optimization/main.py --num_runs 1 --num_samples 50 --model_path "ylacombe/bark-large" --optimization_type "generated_assistant" --precision "$precision" --temperature "$temperature"
  done
done