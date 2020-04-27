# AWS AI toolkit for DJL

## Overview

The aws-ai module contains classes that make it easy for DJL to access AWS services.

### Load model from AWS S3 bucket

With this module, you can easily load model from AWS S3 bucket. As long as you include
this module in your project, [ModelZoo](../../api/src/main/java/ai/djl/repository/zoo/ModelZoo.java) class
can load models from your s3 bucket.

The following pseudocode demonstrates how to load model from S3:
```java
    // Set model zoo search path system property. The value can be
    // comma delimited url string. You can add multiple s3 url.
    // The S3 url should point to a folder in your s3 bucket.
    // In current implementation, DJL will only download files directly
    // in that folder. The archive file like .zip, .tar.gz, .tgz, .tar.z
    // files will be extracted automatically. This is useful for the models
    // that are created by AWS SageMaker.
    // The folder name will be interpreted as artifactId and modelName.
    // If your model file has a different name then the folder name, you
    // need use query string to tell DJL which model you want to load.
    System.setProperty("ai.djl.repository.zoo.location",
            "s3://djl-misc/test/models/resnet18?artifact_id=resnet&model_name=resent18_v1");

    // group "ai.djl.localmodelzoo" is optional. With explicity group id,
    // you limit the search scope in your search locations only. Otherwise, it
    // will search from all model zoo for the artificat "resnet" 
    Criteria<BufferedImage, Classifications> criteria =
        Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(BufferedImage.class, Classifications.class)
                .optArtifactId("ai.djl.localmodelzoo:resnet")
                .optProgress(new ProgressBar())
                .build();

    ZooModel<BufferedImage, Classifications> model = ModelZoo.loadModel(criteria);
```

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
    <version>0.5.0</version>
</dependency>
```
