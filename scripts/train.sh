#/bin/bash

model_name=$1
CUDA_VISIBLE_DEVICES=6,7 python ./exps/train_${model_name}.py