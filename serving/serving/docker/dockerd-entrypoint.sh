#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    java $JAVA_OPTS -cp "libs/*" ai.djl.serving.ModelServer "$APP_ARGS"
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
