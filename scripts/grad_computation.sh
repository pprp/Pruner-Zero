#!/bin/bash

# 125m
CUDA_VISIBLE_DEVICES=0 python lib/gradient_computation.py --nsamples 128 \
    --model /path/to/facebook/opt-125m --llama_version 2 --task gradient

# 350m
CUDA_VISIBLE_DEVICES=1 python lib/gradient_computation.py --nsamples 128 \
    --model /path/to/facebook/opt-350m --llama_version 2 --task gradient

# 1.3b
CUDA_VISIBLE_DEVICES=2 python lib/gradient_computation.py --nsamples 128 \
    --model /path/to/facebook/opt-1.3b --llama_version 2 --task gradient

# 2.7b
CUDA_VISIBLE_DEVICES=4 python lib/gradient_computation.py --nsamples 128 \
    --model /path/to/facebook/opt-2.7b --llama_version 2 --task gradient

# 6.7b
CUDA_VISIBLE_DEVICES=0,1 python lib/gradient_computation.py --nsamples 128 \
    --model /path/to/facebook/opt-6.7b --llama_version 2 --task gradient

# 13b
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lib/gradient_computation.py --nsamples 128 \
    --model /path/to/facebook/opt-13b --llama_version 2 --task gradient

# llama-1-7b
LLAMA1_PATH=/data2/share/llama-1/llama-7b-hf
CUDA_VISIBLE_DEVICES=2,3 python lib/gradient_computation.py --nsamples 128 \
    --model $LLAMA1_PATH --llama_version 1 --task gradient

# llama-2-7b
LLAMA2_PATH=/data2/share/llama/Llama-2-7b-hf
CUDA_VISIBLE_DEVICES=4,5 python lib/gradient_computation.py --nsamples 128 \
    --model $LLAMA2_PATH --llama_version 2 --task gradient
