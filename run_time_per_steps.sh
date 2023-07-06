#!/bin/bash

export HF_DATASETS_CACHE="/home/yoach/.cache/datasets"
export TRANSFORMERS_CACHE="/home/yoach/.cache/transformers"

CUDA_VISIBLE_DEVICES="0" /home/yoach/transformers/transformers-env/bin/python /home/yoach/bark_optimization/time_per_steps.py --num_runs 4 --num_samples 50 --model_path ylacombe/bark-small