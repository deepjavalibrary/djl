# djl.ai - model zoo

## Introduction

The djl.ai model zoo contains framework agnostic models. All of the models have built-in Translator, and
can be used for inference out of the box.

## Installation
You can pull it from the central Maven repositories:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>model-zoo</artifactId>
    <version>0.2.0</version>
</dependency>
```

## Pre-trained models

In 0.2.0 release, we only provide MLP and Resnet50 in our model zoo.

## How to find a pre-trained model in model zoo

In a model zoo repository, there can be many pre-trained that belongs to the same model family.
You can use the `ModelZoo` class to search for the model that you need.
First, you need to decide which model family you want to use, then define key/values search criteria
to narrow down the model you want. If there are multiple models that match your search criteria, the first
model found will be returned. *ModelNotFoundException* will be thrown if no matching model is found.

The following is an example to find resnet50-v1 that trained on imagenet dataset:
```java
    Map<String, String> criteria = new HashMap<>();
    criteria.put("layers", "50");
    criteria.put("flavor", "v1");
    criteria.put("dataset", "cifar10");

    ZooModel<BufferedImage, Classification> model = ModelZoo.RESNET.loadModel(criteria);
```

## List of search criteria of each model

| Category | Application           | Model Family      | Criteria | Possible values |
|----------|-----------------------|-------------------|----------|-----------------|
| CV       | Image Classification  | MLP               | dataset  | mnist           |
|          |                       | Resnet            | layers   | 50              |
|          |                       |                   | flavor   | v1              |
|          |                       |                   | dataset  | cifar10         |

## Developer guide

### [How to add new models to model zoo](../docs/development/add_model_to_model-zoo.md)
