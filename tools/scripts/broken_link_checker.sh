#!/usr/bin/env bash

# installation
sudo apt-get update
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo apt-get install npm
sudo npm install -g markdown-link-check
sudo npm install broken-link-checker -g


# check brokenlinks in md files
for i in $(find ../../ -name '*.md');
do markdown-link-check "$i"; 
done

VERSION=0.3.0
# check broken link in website and java doc site
blc https://djl.ai/ -ro
blc https://javadoc.djl.ai/ -ro
blc https://javadoc.djl.ai/api/${VERSION}/index.html -ro
blc https://javadoc.djl.ai/basicdataset/${VERSION}/index.html -ro
blc https://javadoc.djl.ai/model-zoo/${VERSION}/index.html -ro
blc https://javadoc.djl.ai/mxnet-engine/${VERSION}/index.html -ro
blc https://javadoc.djl.ai/mxnet-model-zoo/${VERSION}/index.html -ro
blc https://javadoc.djl.ai/repository/${VERSION}/index.html -ro