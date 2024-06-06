#!/bin/bash

# Set common variables
version=$1
sparsity_ratio=0.5
cuda_device=$2

echo $version
if [ "$version" == "1.3" ]; then
    model="/path/to/facebook/opt-1.3b"
    gradient_path="gradients/opt/gradients_aggregate_norm_l2_opt-1.3b.pth"
elif [ "$version" == "2.7" ]; then
    model="/path/to/facebook/opt-2.7b"
    gradient_path="gradients/opt/gradients_aggregate_norm_l2_opt-2.7b.pth"
elif [ "$version" == "6.7" ]; then
    model="/path/to/facebook/opt-6.7b"
    gradient_path="gradients/opt/gradients_aggregate_norm_l2_opt-6.7b.pth"
elif [ "$version" == "13" ]; then
    model="/path/to/facebook/opt-13b"
    gradient_path="gradients/opt/gradients_aggregate_norm_l2_opt-13b.pth"
elif [ "$version" == "125" ]; then
    model="/path/to/facebook/opt-125m"
    gradient_path="gradients/opt/gradients_aggregate_norm_l2_opt-125m.pth"
elif [ "$version" == "350" ]; then
    model="/path/to/facebook/opt-350m"
    gradient_path="gradients/opt/gradients_aggregate_norm_l2_opt-350m.pth"
fi

echo $model
echo $gradient_path

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command() {
    python main_opt.py \
        --model $model \
        --gradient_path $gradient_path \
        --prune_method $1 \
        --sparsity_ratio $sparsity_ratio \
        --sparsity_type $2 \
        --save $3 \
        --save_model $4
}

# opt-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/opt${version}_${version}/unstructured/wanda/"
run_python_command "wanda" "2:4" "out/opt${version}_${version}/2-4/wanda/"
run_python_command "wanda" "4:8" "out/opt${version}_${version}/4-8/wanda/"
echo "Finished wanda pruning method"

# opt-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/opt${version}_${version}/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "out/opt${version}_${version}/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "out/opt${version}_${version}/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# opt-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/opt${version}_${version}/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "out/opt${version}_${version}/2-4/magnitude/"
run_python_command "magnitude" "4:8" "out/opt${version}_${version}/4-8/magnitude/"
echo "Finished magnitude pruning method"

# opt-7b with pruner-zero pruning method
echo "Running with pruner-zero pruning method"
run_python_command "pruner-zero" "unstructured" "out/opt${version}_${version}/unstructured/pruner-zero/"
run_python_command "pruner-zero" "2:4" "out/opt${version}_${version}/2-4/pruner-zero/"
run_python_command "pruner-zero" "4:8" "out/opt${version}_${version}/4-8/pruner-zero/"
echo "Running with pruner-zero pruning method"

run_python_command "pruner-zero" "unstructured" "out/opt${version}_${version}/unstructured/pruner-zero/" "saved_model/opt${version}_${version}"
