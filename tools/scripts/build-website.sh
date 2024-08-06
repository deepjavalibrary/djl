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

git checkout gh-pages

echo "generating versions.json"
pwd
current_version=$(awk -F '=' '/djl / {gsub(/ ?"/, "", $2); print $2}' "/home/runner/work/djl/djl/gradle/libs.versions.toml" | awk -F '.' '{print $2}')
versions='[{"version":"master","title":"master","aliases":[]}'
for i in {1..4}; do
  version="0.$((current_version - i)).0"
  versions="$versions, {\"version\":\"$version\",\"title\":\"$version\",\"aliases\":[]}"
done
versions="$versions]"
echo "$versions" | jq "." > "./versions.json"
