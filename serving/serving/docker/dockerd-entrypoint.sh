#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    /usr/bin/djl-serving "$@"
else
    eval "$@"
fi
