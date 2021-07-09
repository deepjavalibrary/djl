# Model Zoo

## Introduction

The Deep Java Library (DJL) model zoo contains engine-agnostic models. All the models have a built-in Translator and
can be used for inference out of the box.

You can find general ModelZoo and model loading document here:

- [Model Zoo](../docs/model-zoo.md)
- [How to load model](../docs/load_model.md)

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl/model-zoo/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\gradlew javadoc
```

The javadocs output is built in the build/doc/javadoc folder.

## Installation
You can pull the model zoo from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>model-zoo</artifactId>
    <version>0.12.0</version>
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
    Criteria<Image, Classifications> criteria =
            Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("layer", "50")
                    .optFilter("flavor", "v1")
                    .optFilter("dataset", "cifar10")
                    .build();

    ZooModel<Image, Classifications> ssd = criteria.loadModel());
```

If you already known which `ModelLoader` to use, you can simply do the following:

```java
    Map<String, String> filter = new HashMap<>();
    filter.put("layers", "50");
    filter.put("flavor", "v1");
    filter.put("dataset", "cifar10");

    ZooModel<Image, Classifications> model = BasicModelZoo.RESNET.loadModel(filter);
```


## Contributor Guides and Documentation

### [How to add new models the to model zoo](../docs/development/add_model_to_model-zoo.md)
