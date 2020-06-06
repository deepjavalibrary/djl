#!/usr/bin/env bash

# test_notebook.sh [filename]
# If no filename is passed, it runs all files in current directory and subdirectories

set -e

function run_test {
    base=$(basename $1)
    jupyter nbconvert --to html --execute --output $base $1
    mv "${1}.html" test_output/
}

mkdir test_output

if [[ $# -eq 0 ]]; then
    for f in {**,.}/*.ipynb
    do
        run_test "$f"
    done
else
    run_test $1
fi
