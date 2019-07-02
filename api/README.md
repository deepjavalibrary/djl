Joule - core API
================

## Overview

This module is core API of the Joule project. It include the following packages:

- engine - An interface to interact with backend Deep Learning framework
- inference - The module used for inference
- metric - Metrics toolkit for benchmark purpose
- modality - Contains CV and NLP toolkit
- ndarray - NDArray data structure and tools
- nn - Blocks containing common Nerual Network functions
- training - The module used for training


## Generate javadoc

You can find javadoc in build/doc/javadoc folder.

If you only want to build javadoc you can do:

```sh
./gradlew javadoc
```

## Publish the package

You can create a jar file by doing the followings:

```bash
./gradlew -Plocal publish
```