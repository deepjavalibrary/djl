# Release Process

## Overview

This document outlines the procedure to release Deep Java Library (DJL) project to maven central. 

## Step 1: Preparing the Release Candidate

### Step 1.1 Publish javadoc to S3 bucket

Make sure you are using correct aws credential and run the following command:

```shell script
cd djl
./gradlew -Prelease uploadJavadoc
```

### Step 1.2: Bump up versions in documents to point to new url

Edit [README Release Notes section](../../README.md#release-notes) to add link to new release. 

Update build version with the following command:
```shell script
cd djl
./gradlew -PpreviousVersion=X.X.X iFV
```
Make a commit, get reviewed, and then merge it into github.

### Step 1.3: Upload javadoc-index.html to S3 bucket

```shell script
aws s3 cp website/javadoc-index.html s3://javadoc-djl-ai/index.html
```

### Step 1.4: Publish mxnet-native library to sonatype staging server

Run the following command to trigger mxnet-native publishing job:
```shell script
curl -XPOST -u "USERNAME:PERSONAL_TOKEN" -H "Accept: application/vnd.github.everest-preview+json" -H "Content-Type: application/json" https://api.github.com/repos/awslabs/djl/dispatches --data '{"event_type": "mxnet-staging-pub"}'
```

### Step 1.5: Remove -SNAPSHOT in examples and jupyter notebooks

Run the following command with correct version value:
```shell script
cd djl
git clean -xdff
./gradlew release
git commit -a -m 'Remove -SNAPSHOT for release vX.X.X'
git tag -a vX.X.X -m "Releasing version vX.X.X"
git push origin vX.X.X
```

### Step 1.6 Create a Release Notes

Navigate to DJL github site, select "Release" tab and click "Draft a new Release" button.
Select tag that created by previous step. Check "This is a pre-release" checkbox.

Release notes content should include the following:
- list of new features
- list of bug fixes
- limitations and known issues
- API changes and migration document

Once "Publish release" button is clicked, a github Action will be triggered, and release packages
will be published sonatype staging server.

## Step 2: Validate release on staging server

Login to https://oss.sonatype.org/, and find out staging repo name.

Run the following command to point maven repository to staging server:
```shell script
cd djl
git checkout vX.X.X
./gradlew -PstagingRepo=aidjl-XXXX staging
```

### Validate examples project are working fine

```shell script
cd examples
./gradlew run
mvn exec:java -Dexec.mainClass="ai.djl.examples.inference.ObjectDetection" 
```

### Validate jupyter notebooks

Make sure jupyter notebook and running properly and all javadoc links are accessible.
```shell script
cd jupyter
jupyter notebook
```

## Step 3: Validate javadoc url in documents

Navigate to DJL github site, select tag created by earlier step, open markdown files and
check javadoc links are accessible. 

## Step 4: Publish staging package to maven central

Login to https://oss.sonatype.org/, close staging packages and publish to maven central.

## Step 5: Upgrade version for next release

```shell script
cd djl
./gradlew -PtargetVersion=X.X.X iBV
```

Create a PR to get reviewed and merge into github.

## Step 6: Publish new snapshot to sonatype

Manually trigger a nightly build with the following command:
```shell script
curl -XPOST -u "USERNAME:PERSONAL_TOKEN" -H "Accept: application/vnd.github.everest-preview+json" -H "Content-Type: application/json" https://api.github.com/repos/awslabs/djl/dispatches --data '{"event_type": "nightly-build"}'
```
