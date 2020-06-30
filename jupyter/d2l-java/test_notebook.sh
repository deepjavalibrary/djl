#!/usr/bin/env bash

# test_notebook.sh [filename]
# If no filename is passed, it runs all files in current directory and subdirectories

set -e

function run_test {
    base=$(basename $1)
    dir=$(dirname $1)
    jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=600 --output $base $1
    mkdir -p test_output/$dir
    mv "${1}.html" test_output/$dir/
}

if [[ $# -eq 0 ]]; then
    for f in **/*.ipynb
    do
        run_test "$f"
    done
else
	for file in "$@"
	do
	    if [[ "$file" == *ipynb ]]; then
            full_path=$(find . -name "$file")
	 	    run_test "$full_path"
	 	else
            for f in ${file}/*.ipynb
            do
                run_test "$f"
            done
        fi
   done
fi
