#!/bin/bash
set -Eeuo pipefail

# default argument values
# directory where trained models are stored
compare_directory="runs/detect/"
# directory where evaluation data are stored
evaluation_common_path="evaluation_results"
# path to test yaml of all test datasets
unitem_test_data_path="datasets/unitem_all/test.yaml"
# path to dir tree with test yamls
splitted_test_data_path="datasets/test/"
# output directory
compare_directory_output="comparison"

regenerate_evaluation_results=false
evaluate_single_flag=true
evaluate_splitted_dataset_flag=false

while getopts "d:i:m:f:o:r:e:s" flag; do
    case "${flag}" in
        d) compare_directory="$OPTARG";;
        i) evaluation_common_path="$OPTARG";;
        m) unitem_test_data_path="$OPTARG";;
        f) splitted_test_data_path="$OPTARG";;
        o) compare_directory_output="$OPTARG";;
        r) regenerate_evaluation_results=true;;
        e) evaluate_single_flag=true;;
        s) evaluate_splitted_dataset_flag=true;;
        \?) echo "Wrong flag: -$OPTARG" >&2;;
    esac
done

echo "Generating current yolo comparison"

# common settings and arguments
model_common_path="weights/best.pt"
args_common_path="args.yaml"

if "$regenerate_evaluation_results"; then
    echo "Regenerating evaluation results for all available models"
    models_to_compare=($(ls -1 $compare_directory))
    for model_name in "${models_to_compare[@]}"
    do
    model_path="$compare_directory/$model_name/$model_common_path"
    args_path="$compare_directory/$model_name/$args_common_path"

    result_save_dir="$compare_directory/$model_name/$evaluation_common_path"
    echo "Evaluating model: $model_name"
    rm -rf "$result_save_dir"

    if "$evaluate_single_flag"; then
    evaluate_single $unitem_test_data_path --args-path="$args_path" --model-path="$model_path"
    fi

    if "$evaluate_splitted_dataset_flag"; then
    evaluate_splitted_dataset $splitted_test_data_path --args-path="$args_path" --model-path="$model_path"
    fi
    done
else
    echo "Using existing evaluation results for all available models. Make sure that they are up to date."
fi

compare_models $compare_directory --output-dir $compare_directory_output
