# HDFS support for DJL

## Overview

HDFS is widely used in Spark applications. We introduce HDFS integration for DJL to better support Spark use case.

### Load model from HDFS

With this module, you can directly load model from HDFS url.

The following pseudocode demonstrates how to load model from HDFS url:

```java
    Criteria<Image, Classifications> criteria =
        Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optModelUrls("hdfs://localhost:63049/resnet.tar.z")
                .optModelName("resnet18-v1")
                .build();

    ZooModel<Image, Classifications> model = criteria.loadModel();
```

See [How to load a model](../../docs/load_model.md) for more detail.

`HdfsRepositoryFactory` will be registered automatically in DJL as long as you add this module in your class path.
If you want to customize your Hadoop configuration, you can manually register a customized `HdfsRepositoryFactory`:

```java
    Configuration config = new Configuration();
    Repository.registerRepositoryFactory(new HdfsRepositoryFactory(config));
```

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/ai.djl.hadoop/hadoop/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```

The javadocs output is built in the build/doc/javadoc folder.


## Installation
You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.hadoop</groupId>
    <artifactId>hadoop</artifactId>
    <version>0.12.0</version>
</dependency>
```
