#!/bin/bash

export HF_DATASETS_CACHE="/home/yoach/.cache/datasets"
export TRANSFORMERS_CACHE="/home/yoach/.cache/transformers"



# Define the models, batch sizes, and precisions
# "generated_assistant"
# "torch.bfloat16"
models=("ylacombe/bark-small" "ylacombe/bark-large")
optimization_types=("bettertransformer" "no_optimization")
precisions=("torch.float16" "torch.float32")
batch_sizes=(1)

# Loop through each combination and execute the python command
for optimization_type in "${optimization_types[@]}"
do
  for model_name in "${models[@]}"
  do
    for precision in "${precisions[@]}"
    do
      for batch_size in "${batch_sizes[@]}"
      do
        CUDA_VISIBLE_DEVICES=2 /home/yoach/bark_optimization/bark_optimization_env/bin/python /home/yoach/bark_optimization/main.py --use_offload --max_num_tokens 256 --num_runs 2 --num_samples 50 --batch_size "$batch_size" --model_path "$model_name" --optimization_type "$optimization_type" --precision "$precision" --output_file "output_offload_hooks.csv"
        CUDA_VISIBLE_DEVICES=2 /home/yoach/bark_optimization/bark_optimization_env/bin/python /home/yoach/bark_optimization/main.py --max_num_tokens 256 --num_runs 2 --num_samples 50 --batch_size "$batch_size" --model_path "$model_name" --optimization_type "$optimization_type" --precision "$precision" --output_file "output_offload_hooks.csv"
      done
    done
  done
done