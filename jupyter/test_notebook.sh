#!/usr/bin/env bash

# test_notebook.sh [filename]
# If no filename is passed, it runs all files in current directory and subdirectories

set -e

function run_test {
    jupyter-nbconvert --to notebook --execute --output outFile $1
    # grep "output_type" outFile.ipynb | grep "error"
    rm outFile.ipynb
}

if [[ $# -eq 0 ]]; then
    for f in {**,.}/*.ipynb
    do
        run_test "$f"
    done
else
    run_test $1
fi
