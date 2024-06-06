#!/bin/bash

# Set common variables
version=$1
cuda_device=$2
sparsity_ratio=$3

echo $version
if [ $version -eq 1 ]; then
    # llama-1
    model="/path/to-1/llama-7b-hf"
    gradient_path="/path/to/gradient/gradients_aggregrate_norm_l2_model_llama-7b-hf.pth"
elif [ $version -eq 2 ]; then
    # llama-2
    model="/path/to/Llama-2-7b-hf"
    gradient_path="/path/to/gradient/gradients_aggregrate_norm_l2_model_llama-2-7b.pth"
fi

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command() {
    python main.py \
        --model $model \
        --gradient_path $gradient_path \
        --prune_method $1 \
        --sparsity_ratio $sparsity_ratio \
        --sparsity_type $2 \
        --save $3 >./out/llama${version}_7b_${2}_${1}.txt
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
CUDA_VISIBLE_DEVICES=1 run_python_command "wanda" "unstructured" "out/llama${version}_7b/unstructured/wanda/"
run_python_command "wanda" "2:4" "out/llama${version}_7b/2-4/wanda/"
run_python_command "wanda" "4:8" "out/llama${version}_7b/4-8/wanda/"
echo "Finished wanda pruning method"

# llama-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/llama${version}_7b/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "out/llama${version}_7b/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "out/llama${version}_7b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# llama-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/llama${version}_7b/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "out/llama${version}_7b/2-4/magnitude/"
run_python_command "magnitude" "4:8" "out/llama${version}_7b/4-8/magnitude/"
echo "Finished magnitude pruning method"

# llama-7b with pruner-zero pruning method
echo "Running with pruner-zero pruning method"
run_python_command "pruner-zero" "unstructured" "out/llama${version}_7b/unstructured/pruner-zero/"
run_python_command "pruner-zero" "2:4" "out/llama${version}_7b/2-4/pruner-zero/"
run_python_command "pruner-zero" "4:8" "out/llama${version}_7b/4-8/pruner-zero/"
echo "Running with pruner-zero pruning method"
