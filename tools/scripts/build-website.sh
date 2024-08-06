#!/usr/bin/env bash

set -ex

VERSION_NUMBER=$1
if [[ "$VERSION_NUMBER" == "" ]]; then
  VERSION_NUMBER=master
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Updating online runner ..."
python3 "$BASE_DIR/add_online_runner.py"

# Configure Git User, mike will create gh-pages branch
git config user.name "nobody"
git config user.email "nobody@localhost"

echo "Building docs for $VERSION_NUMBER"
mike deploy "$VERSION_NUMBER" -F docs/mkdocs.yml -b gh-pages
