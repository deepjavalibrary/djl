# AWS AI toolkit for DJL

## Overview

The aws-ai module contains classes that make it easy for DJL to access AWS services.

### Load model from AWS S3 bucket

With this module, you can easily load model from AWS S3 bucket. As long as you include
this module in your project, [ModelZoo](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/ModelZoo.html) class
can load models from your s3 bucket.

The following pseudocode demonstrates how to load model from S3:

```java
    Criteria<Image, Classifications> criteria =
        Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optModelUrls("s3://djl-misc/test/models/resnet18")
                .optModelName("resent18_v1")
                .optProgress(new ProgressBar())
                .build();

    ZooModel<Image, Classifications> model = criteria.loadModel();
```

See [How to load a model](../../docs/load_model.md) for more detail.

If you want to customize your AWS credentials and region, you can manually register a customized
`S3RepositoryFactory`:

```java
    S3Client client = S3Client.builder()
        .credentialsProvider(provider)
        .region(Region.US_EAST_1)
        .build();

    Repository.registerRepositoryFactory(new S3RepositoryFactory(client));
```

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/ai.djl.aws/aws-ai/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.


## Installation
You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.aws</groupId>
    <artifactId>aws-ai</artifactId>
    <version>0.12.0</version>
</dependency>
```
