# Google Cloud toolkit for DJL

## Overview

The google-ai module contains classes that make it easy for DJL to access Google Cloud services.

### Load model from a Google Cloud Storage bucket

With this module, you can easily load a model from a Google Cloud Storage (GCS) bucket. As long as
you include this module in your project, the
[ModelZoo](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/ModelZoo.html) class
can load models from your GCS bucket using a `gs://` URL.

The following pseudocode demonstrates how to load a model from GCS:

```java
    Criteria<Image, Classifications> criteria =
        Criteria.builder()
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .setTypes(Image.class, Classifications.class)
                .optModelUrls("gs://djl-misc/test/models/resnet18")
                .optModelName("resnet18_v1")
                .optProgress(new ProgressBar())
                .build();

    ZooModel<Image, Classifications> model = criteria.loadModel();
```

The `gs://` URL should point to a folder in your GCS bucket. DJL will download the files directly in
that folder. Archive files (`.zip`, `.tar.gz`, `.tgz`, `.tar.z`) are extracted automatically. The
folder name is interpreted as the `artifactId` and `modelName`. If your model file has a different
name than the folder, use the `model_name` query string to tell DJL which model to load:

```
gs://djl-misc/test/models/resnet18?model_name=resnet18_v1
```

See [How to load a model](../../docs/load_model.md) for more detail.

### Authentication

By default, the module uses
[Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials)
to authenticate with Google Cloud. This works automatically when you run on Google Cloud, or
locally after running `gcloud auth application-default login`, or by setting the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of a service account key file.

If you want to customize your credentials or project, you can manually register a customized
`GcsRepositoryFactory`:

```java
    Storage storage = StorageOptions.newBuilder()
        .setProjectId("my-project")
        .setCredentials(credentials)
        .build()
        .getService();

    Repository.registerRepositoryFactory(new GcsRepositoryFactory(storage));
```

To access a public bucket without credentials, you can build the client anonymously:

```java
    Storage storage = StorageOptions.newBuilder()
        .setCredentials(NoCredentials.getInstance())
        .build()
        .getService();

    Repository.registerRepositoryFactory(new GcsRepositoryFactory(storage));
```

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.google/google-ai/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.


## Installation
You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.google</groupId>
    <artifactId>google-ai</artifactId>
    <version>0.37.0</version>
</dependency>
```
