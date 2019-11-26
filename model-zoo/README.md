# DJL - model zoo

## Introduction

The Deep Java Library (DJL) model zoo contains framework-agnostic models. All the models have a built-in Translator and
can be used for inference out of the box.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.djl.ai/model-zoo/0.2.0/index.html).

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
    <version>0.2.0</version>
</dependency>
```


## Pre-trained models

In the 0.2.0 release, you can find the Multilayer Perceptrons (MLP) and Resnet50 pre-trained models in the model zoo.

### How to find a pre-trained model in the model zoo

In a model zoo repository, there can be many pre-trained models that belong to the same model family.
You can use the `ModelZoo` class to search for the model you need.
First, decide which model family you want to use. Then, define your key/values search criteria
to narrow down the model you want. If there are multiple models that match your search criteria, the first
model found is returned. A *ModelNotFoundException* will be thrown if no matching model is found.

The following is an example of the criteria to find a Resnet50-v1 model that has been trained on the imagenet dataset:
```java
    Map<String, String> criteria = new HashMap<>();
    criteria.put("layers", "50");
    criteria.put("flavor", "v1");
    criteria.put("dataset", "cifar10");

    ZooModel<BufferedImage, Classification> model = ModelZoo.RESNET.loadModel(criteria);
```

### List of search criteria of each model

The following table illustrates the possible search criteria for all models in the model zoo:

| Category | Application           | Model Family      | Criteria | Possible values |
|----------|-----------------------|-------------------|----------|-----------------|
| CV       | Image Classification  | MLP               | dataset  | mnist, cifar10  |
|          |                       | Resnet            | layers   | 50              |
|          |                       |                   | flavor   | v1              |

## Contributor Guides and Documentation

### [How to add new models the to model zoo](../docs/development/add_model_to_model-zoo.md)
