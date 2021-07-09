# Model Loading

A model is a collection of artifacts that is created by the training process.
In deep learning, running inference on a Model usually involves pre-processing and post-processing.
DJL provides a [ZooModel](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/ZooModel.html) 
class, which makes it easy to combine data processing with the model.

This document will show you how to load a pre-trained model in various scenarios.

## Using the ModelZoo API to load a Model

We recommend you use the [ModelZoo](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/ModelZoo.html)
API to load models.

The ModelZoo API provides a unified way to load models. The declarative nature of this API allows you to store model
information inside a configuration file. This gives you great flexibility to test and deploy your model.
See our reference project: [DJL Spring Boot Starter](https://github.com/deepjavalibrary/djl-spring-boot-starter#spring-djl-mxnet-autoconfiguration). 

### Criteria class

You can use the [Criteria](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/Criteria.html) class 
to narrow down your search condition and locate the model you want to load.
[Criteria](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/Criteria.html) class follows
DJL Builder convention. The methods start with `set` are required fields, and `opt` for optional fields.
You must call `setType()` method when creating a `Criteria` object:

```
Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class)
        .build();
```

The criteria accept the following optional information:

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

*Note:* If multiple models match the criteria you specified, the first one will be returned. The result is not deterministic.

### Load model from the ModelZoo repository

The advantage of using the ModelZoo repository is it provides a way to manage models versions. DJL allows you
to update your model in the repository without conflict with existing models. The model consumer can pick up new models without any code changes.
DJL searches the classpath and locates the available ModelZoos in the system. 

DJL provide several built-in ModelZoos:

- [ai.djl:model-zoo](https://search.maven.org/search?q=g:ai.djl%20AND%20a:model-zoo) Engine-agnostic imperative model zoo
- [ai.djl.mxnet:mxnet-model-zoo](https://search.maven.org/search?q=g:ai.djl.mxnet%20AND%20a:mxnet-model-zoo) MXNet symbolic model zoo
- [ai.djl.pytorch:pytorch-model-zoo](https://search.maven.org/search?q=g:ai.djl.pytorch%20AND%20a:pytorch-model-zoo) PyTorch torch script model zoo
- [ai.djl.tensorflow:tensorflow-model-zoo](https://search.maven.org/search?q=g:ai.djl.tensorflow%20AND%20a:tensorflow-model-zoo) TensorFlow saved bundle model zoo

You can create your own model zoo if needed, but we are still working on improving the tools to help create custom model zoo repositories.

### Load models from the local file system

The following shows how to load a pre-trained model from a file path:

```java
Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class) // defines input and output data type
        .optTranslator(ImageClassificationTranslator.builder().setSynsetArtifactName("synset.txt").build())
        .optModelUrls("file:///var/models/my_resnet50") // search models in specified path
        .optModelName("resnet50") // specify model file prefix
        .build();

ZooModel<Image, Classifications> model = criteria.loadModel();
```

DJL supports loading a pre-trained model from a local directory or an [archive file](#current-supported-archive-formats).

#### Current supported archive formats

- .zip
- .tar
- .tar.gz, .tgz, .tar.z

#### Customize artifactId and modelName

By default, DJL will use the directory/file name of the URL as the model's `artifactId` and `modelName`.
Some engines (MXNet, PyTorch) are sensitive to the model name, you usually need name the directory or archive
file to be the same as model. You can use the URL query string to tell DJL how to load model if the model name are
different from the directory or archive file:

- model_name: the file name (or prefix) of the model. You need to include the relative path to the model file if it's in a sub folder of the archive file. 
- artifact_id: define a `artifactId` other than the file name

For example:

```
file:///var/models/resnet.zip?artifact_id=resenet-18&model_name=resnet-18v1
```

If your the directory or archive file has nested folder, are include the folder name in url to let DJL know where
to find model files:

```
file:///var/models/resnet.zip?artifact_id=resenet-18&model_name=saved_model/resnet-18
```

### Load model from a URL

DJL supports loading a model from a URL. Since a model consists multiple files, some of URL must be
an [archive file](#current-supported-archive-formats).

Current supported URL scheme:

- file:// load a model from local directory or archive file
- http(s):// load a model from an archive file from web server  
- jar:// load a model from an archive file in the class path

```java
Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class) // defines input and output data type
        .optTranslator(ImageClassificationTranslator.builder().setSynsetArtifactName("synset.txt").build())
        .optModelUrls("https://resources.djl.ai/benchmark/squeezenet_v1.1.tar.gz") // search models in specified path
        .build();

ZooModel<Image, Classifications> model = criteria.loadModel();
```

You can [customize the artifactId and modelName](#customize-artifactid-and-modelname) the same way as loading model from the local file system.

### Load model from AWS S3 bucket
DJL supports loading a model from an S3 bucket using `s3://` URL and the AWS plugin. See [here](../extensions/aws-ai/README.md) for details.

### Load model from Hadoop HDFS
DJL supports loading a model from a Hadoop HDFS file system using `hdfs://` URL and the Hadoop plugin. See [here](../extensions/hadoop/README.md) for details.

### Implement your own Repository
You may want to create additional model zoos using other protocols such as:

- ftp://
- sftp://
- tftp://
- rsync://
- smb://
- mvn://
- jdbc://

DJL is highly extensible and our API allows you to create your own URL protocol handling by extending `Repository` class:

- Create a class that implements `RepositoryFactory` interface
    make sure `getSupportedScheme()` returns URI schemes that you what to handle
- Create a class that implements `Repository` interface.
- DJL use ServiceLoader to automatically register your `RepositoryFactory`. You need add a file `META-INF/services/ai.djl.repository.RepositoryFactory`
    See [java ServiceLoader](https://docs.oracle.com/javase/9/docs/api/java/util/ServiceLoader.html) for more detail.

You can refer to [AWS S3 Repostory](../extensions/aws-ai/README.md) for an example.

## Configure model zoo search path

DJL provides a way for developers to configure a system wide model search path by setting a `ai.djl.repository.zoo.location`
system properties:

```
-Dai.djl.repository.zoo.location=https://djl-ai.s3.amazonaws.com/resnet.zip,s3://djl-misc/test/models,file:///myModels
```

The value can be comma delimited url string.

### Debug model loading issues

You may run into `ModelNotFoundException` issue. In most cases, it's caused by the `Criteria` you specified
doesn't match the desired model.

Here is a few tips you can use to help you debug model loading issue:

#### Enable debug log
See [here](development/configure_logging.md#configure-logging-level) for how to enable debug log

#### List models programmatically in your code
You can use [ModelZoo.listModels()](https://javadoc.io/static/ai.djl/api/0.12.0/ai/djl/repository/zoo/ModelZoo.html#listModels--) API to query available models.

#### List available models using DJL command line

Use the following command to list models in examples module for MXNet engine:

```shell
./gradlew :examples:listmodels

[INFO ] - CV.ACTION_RECOGNITION ai.djl.mxnet:action_recognition:0.0.1 {"backbone":"vgg16","dataset":"ucf101"}
[INFO ] - CV.ACTION_RECOGNITION ai.djl.mxnet:action_recognition:0.0.1 {"backbone":"inceptionv3","dataset":"ucf101"}
[INFO ] - CV.IMAGE_CLASSIFICATION ai.djl.zoo:resnet:0.0.1 {"layers":"50","flavor":"v1","dataset":"cifar10"}
[INFO ] - CV.IMAGE_CLASSIFICATION ai.djl.zoo:mlp:0.0.2 {"dataset":"mnist"}
[INFO ] - NLP.QUESTION_ANSWER ai.djl.mxnet:bertqa:0.0.1 {"backbone":"bert","dataset":"book_corpus_wiki_en_uncased"}

...

```

You can list models from your model folder and only list models for specific Engine with debug log:

```shell
./gradlew :examples:listmodels -Dai.djl.default_engine=PyTorch -Dai.djl.logging.level=debug -Dai.djl.repository.zoo.location=file:///mymodels
```
