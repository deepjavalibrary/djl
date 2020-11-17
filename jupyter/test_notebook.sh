#!/usr/bin/env bash

# test_notebook.sh [filename]
# If no filename is passed, it runs all files in current directory and subdirectories

set -e

function run_test {
    base=$(basename $1)
    # Workaround on crashes
    if [[ "$base" == transfer_learning_on_cifar10* ||  "$base" == rank_classification_using_BERT* ]]; then
        jupyter nbconvert --to notebook --inplace $1
    else
        jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace $1
    fi
}

if [[ $# -eq 0 ]]; then
    for f in {**,.}/*.ipynb
    do
        dir=$(dirname f)
        run_test "$f"
    done
else
    run_test $1
fi
