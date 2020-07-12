# How to load model

A model is a collection of artifacts that is created by the training process.
In Deep Learning, running inference on a Model usually involves pre-processing and post-processing.
DJL provides a [ZooModel](https://javadoc.io/static/ai.djl/api/0.6.0/index.html?ai/djl/repository/zoo/ZooModel.html) 
class, which makes it easy to combine data processing with the model.

This document will show you how to load a pre-trained model in various scenarios.

## Using ModelZoo API to load a Model

We recommend you to use the [ModelZoo](https://javadoc.io/static/ai.djl/api/0.6.0/index.html?ai/djl/repository/zoo/ModelZoo.html)
API to load models.

The ModelZoo API provides a unified way to load models. The declarative nature of this API allows you to store model
information in a configuration file. This gives you great flexibility to test and deployment your model.  
See reference project: [DJL Spring Boot Starter](https://github.com/awslabs/djl-spring-boot-starter#spring-djl-mxnet-autoconfiguration). 

### Criteria class

You can use [Criteria](https://javadoc.io/static/ai.djl/api/0.6.0/index.html?ai/djl/repository/zoo/Criteria.html) class 
to narrow down your search condition and locate the model you want to load.

- Engine: defines on which engine you want your model to be loaded
- Device: defines on which device (GPU/CPU) you want your model to be loaded
- Application: defines model application
- Input/Output data type: defines desired input and output data type
- artifact id: defines the artifact id of the model you want to load, you can use fully-qualified name that includes group id
- group id: defines the group id of the pre loaded ModelZoo that the model belongs to
- ModelZoo: specifies a ModelZoo in which to search model
- model urls: a comma delimited string defines at where the models are stored 
- Translator: defines custom data processing functionality to be used to ZooModel
- Progress: specifies model loading progress
- filters: defines search filters that must match the properties of the model
- options: defines engine/model specific options to load the model
- arguments: defines model specific arguments to load the model

*Note:* If multiple model matches the criteria you specified, the first one will be returned. The result is not deterministic.

### Load model from ModelZoo repository

The advantage of using ModelZoo repository is it provides a way to manage models versions. DJL allows you
to update your model in the repository without conflict with existing models. Model consumer can pick up new model without code change.
DJL searches classpath and locate available ModelZoo's in the system. 

DJL provide several built-in ModelZoo:

- [ai.djl:model-zoo](https://search.maven.org/search?q=g:ai.djl%20AND%20a:model-zoo) Engine-agnostic imperative model zoo
- [ai.djl.mxnet:mxnet-model-zoo](https://search.maven.org/search?q=g:ai.djl.mxnet%20AND%20a:mxnet-model-zoo) MXNet symbolic model zoo
- [ai.djl.pytorch:pytorch-model-zoo](https://search.maven.org/search?q=g:ai.djl.pytorch%20AND%20a:pytorch-model-zoo) PyTorch torch script model zoo
- [ai.djl.tensorflow:tensorflow-model-zoo](https://search.maven.org/search?q=g:ai.djl.tensorflow%20AND%20a:tensorflow-model-zoo) TensorFlow saved bundle model zoo

Users can create theirs own model zoo if needed, we are working on improving tools to help developer create their own model zoo repository.

### Load model from local file system

The following shows how to load a pre-trained model from a file path:
```java
Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class) // defines input and output data type
        .optTranslator(ImageClassificationTranslator.builder().setSynsetArtifactName("synset.txt").build())
        .optModelUrls("file:///var/models/my_resnet50") // search models in specified path
        .optArtifactId("ai.djl.localmodelzoo:my_resnet50") // defines which model to load
        .build();

ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
```

DJL supports loading a pre-trained model from a local directory or an [archive file](#current-supported-archive-formats).

#### Current supported archive formats

- .zip
- .tar
- .tar.gz, .tgz, .tar.z

#### Customize artifactId and modelName

By default, DJL will use the directory/file name of the URL as the model's `artifactId` and `modelName`.
Some engines (MXNet, PyTorch) are sensitive to the model name, you usually need name the directory or archive
file to be the same as model. You can use URL query string to tell DJL how to load model if the model name are
different from the directory or archive file:

- model_name: the file name (or prefix) of the model.

    You need to include the relative path to the model file if it's in a sub folder of the archive file. 
- artifact_id: define a `artifactId` other than the file name

For example:
```
files:///var/models/resnet.zip?artifact_id=resenet-18&model_name=resnet-18v1
```

If your the directory or archive file has nested folder, are include the folder name in url to let DJL know where
to find model files:
```
files://var/models/resnet.zip?artifact_id=resenet-18&model_name=saved_model/resnet-18
```

### Load model from a URL

DJL supports loading a model from a HTTP(s) URL. Since a model consists multiple files, the URL must be
an [archive file](#current-supported-archive-formats).

```java
Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class) // defines input and output data type
        .optTranslator(ImageClassificationTranslator.builder().setSynsetArtifactName("synset.txt").build())
        .optModelUrls("https://djl-ai.s3.amazonaws.com/resources/benchmark/squeezenet_v1.1.tar.gz") // search models in specified path
        .optArtifactId("ai.djl.localmodelzoo:squeezenet_v1.1") // defines which model to load
        .build();

ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
```

You can [customize artifactId and modelName](#customize-artifactid-and-modelname) the same way as loading model from local file system.

### Load model from AWS S3 bucket
DJL supports loading a model from S3 bucket using `s3://` URL, see [here](../3rdparty/aws-ai/README.md) for detail.

### Load model from Hadoop HDFS
DJL supports loading a model from Hadoop HDFS file system using `hdfs://` URL, see [here](../3rdparty/hadoop/README.md) for detail.

### Implement your own Repository
Users may want to access their model using varies protocol, such as:

- ftp://
- sftp://
- tftp://
- rsync://
- smb://
- mvn://
- jdbc://

DJL is highly extensible, our API allows you to create your own URL protocol handling by extending `Repository` class:

- Create a class that implements `RepositoryFactory` interface
    make sure `getSupportedScheme()` returns URI schemes that you what to handle
- Create a class that implements `Repository` interface.
- DJL use ServiceLoader to automatically register your `RepositoryFactory`. You need add a file `META-INF/services/ai.djl.repository.RepositoryFactory`
    See [java ServiceLoader](https://docs.oracle.com/javase/9/docs/api/java/util/ServiceLoader.html) for more detail.

You can refer to [AWS S3 Repostory](../3rdparty/aws-ai/README.md) for an example.

## Configure model zoo search path

DJL provides a way for developers to configure a system wide model search path by setting a `ai.djl.repository.zoo.location`
system properties:

```
-Dai.djl.repository.zoo.location=https://djl-ai.s3.amazonaws.com/resnet.zip,s3://djl-misc/test/models,file:///myModels
```

The value can be comma delimited url string.
