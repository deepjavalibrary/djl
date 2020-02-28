# DJL - model zoo

## Introduction

The Deep Java Library (DJL) model zoo contains framework-agnostic models. All the models have a built-in Translator and
can be used for inference out of the box.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.djl.ai/model-zoo/0.3.0/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.

## Installation
You can pull the model zoo from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>model-zoo</artifactId>
    <version>0.3.0</version>
</dependency>
```

## Pre-trained models

You can find the Multilayer Perceptrons (MLP) and Resnet50 pre-trained models in the model zoo.

### How to find a pre-trained model in the model zoo

In a model zoo repository, there can be many pre-trained models that belong to the same model family.
You can use the `ModelZoo` class to search for the model you need.
First, decide which model family you want to use. Then, define your key/values search criteria
to narrow down the model you want. If there are multiple models that match your search criteria, the first
model found is returned. A *ModelNotFoundException* will be thrown if no matching model is found.

The following is an example of the criteria to find a Resnet50-v1 model that has been trained on the imagenet dataset:

```java
    Criteria<BufferedImage, Classification> criteria =
            Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(BufferedImage.class, Classification.class)
                    .optFilter("layer", "50")
                    .optFilter("flavor", "v1")
                    .optFilter("dataset", "cifar10")
                    .build();

    ZooModel<BufferedImage, Classification> ssd = ModelZoo.loadModel(criteria));
```

If you already known which `ModelLoader` to use, you can simply do the following:
```java
    Map<String, String> filter = new HashMap<>();
    filter.put("layers", "50");
    filter.put("flavor", "v1");
    filter.put("dataset", "cifar10");

    ZooModel<BufferedImage, Classification> model = BasicModelZoo.RESNET.loadModel(filter);
```

### List available models

You can use [ModelZoo.listModels()](../repository/src/main/java/ai/djl/repository/zoo/ModelZoo.java) API to query available models.

Use the following command to list built-in models in examples module:
```shell script
./gradlew :examples:run -Dmain=ai.djl.examples.inference.ListModels

[INFO ] - CV.ACTION_RECOGNITION ai.djl.mxnet:action_recognition:0.0.1 {"backbone":"vgg16","dataset":"ucf101"}
[INFO ] - CV.ACTION_RECOGNITION ai.djl.mxnet:action_recognition:0.0.1 {"backbone":"inceptionv3","dataset":"ucf101"}
[INFO ] - CV.IMAGE_CLASSIFICATION ai.djl.zoo:resnet:0.0.1 {"layers":"50","flavor":"v1","dataset":"cifar10"}
[INFO ] - CV.IMAGE_CLASSIFICATION ai.djl.zoo:mlp:0.0.2 {"dataset":"mnist"}
[INFO ] - NLP.QUESTION_ANSWER ai.djl.mxnet:bertqa:0.0.1 {"backbone":"bert","dataset":"book_corpus_wiki_en_uncased"}

...

```

## Contributor Guides and Documentation

### [How to add new models the to model zoo](../docs/development/add_model_to_model-zoo.md)
