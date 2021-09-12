# DJL - Apache MXNet model zoo

## Introduction

The model zoo contains symbolic models from Apache MXNet (incubating) that can be used for inference and training. All the models in this model zoo contain pre-trained parameters for their specific datasets.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.mxnet/mxnet-model-zoo/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.


## Installation
You can pull the MXNet engine from the central Maven repository by including the following dependency in you `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-model-zoo</artifactId>
    <version>0.12.0</version>
</dependency>
```

## Pre-trained models

The MXNet model zoo contains two major categories: Computer Vision (CV) and Natural Language Processing (NLP). 
All the models are grouped by task under these two categories as follows:


* CV
  * Action Recognition
  * Image Classification
  * Object Detection
  * Pose Estimation
  * Semantic Segmentation/Instance Segmentation
* NLP
  * Question and Answer

### How to find a pre-trained model in model zoo

In a model zoo repository, there can be many pre-trained models that belong to the same model family.
You can use the `ModelZoo` class to search for the model that you need.
First, decide which model family you want to use. Then, define your key/values search criteria
to narrow down the model you want. If there are multiple models that match your search criteria, the first
model found is returned. *ModelNotFoundException* will be thrown if no matching model is found.

The following is an example of the criteria to find a Resnet50-v1 model that has been trained on the imagenet dataset:

```java
Criteria<Image, Classifications> criteria = Criteria.builder()
        .setTypes(Image.class, Classifications.class)
        .optArtifactId("resnet")
        .optFilter("layers", "50")
        .optFilter("flavor", "v1")
        .optFilter("dataset", "imagenet")
        .optDevice(device)
        .build();

ZooModel<Image, Classifications> model = criteria.loadModel();
```

### List of search criteria for each model

See: [List available model](../../model-zoo/README.md#list-available-models).

The following table illustrates the possible search criteria for all models in the model zoo:

| Category | Application           | Model Family      | Criteria | Possible values                                            |
|----------|-----------------------|-------------------|----------|------------------------------------------------------------|
| CV       | Action Recognition    | ActionRecognition | backbone | vgg16, inceptionv3                                         |
|          |                       |                   | dataset  | ucf101                                                     |
|          | Image Classification  | MLP               | dataset  | mnist                                                      |
|          |                       | Resnet            | layers   | 18, 34, 50, 101, 152                                       |
|          |                       |                   | flavor   | v1, v2, v1d                                                |
|          |                       |                   | dataset  | imagenet, cifar10                                          |
|          |                       | Resnext           | layers   | 101, 150                                                   |
|          |                       |                   | flavor   | 32x4d, 64x4d                                               |
|          |                       |                   | dataset  | imagenet                                                   |
|          |                       | Senet             | layers   | 154                                                        |
|          |                       |                   | dataset  | imagenet                                                   |
|          |                       | SeResnext         | layers   | 101, 150                                                   |
|          |                       |                   | flavor   | 32x4d, 64x4d                                               |
|          |                       |                   | dataset  | imagenet                                                   |
|          | Instance Segmentation | mask_rcnn         | backbone | resnet18, resnet50, resnet101                              |
|          |                       |                   | flavor   | v1b, v1d                                                   |
|          |                       |                   | dataset  | coco                                                       |
|          | Object Detection      | SSD               | size     | 300, 512                                                   |
|          |                       |                   | backbone | vgg16, mobilenet, resnet18, resnet50, resnet101, resnet152 |
|          |                       |                   | flavor   | atrous, 1.0, v1, v2                                        |
|          |                       |                   | dataset  | coco, voc                                                  |
|          | Pose Estimation       | SimplePose        | backbone | resnet18, resnet50, resnet101, resnet152                   |
|          |                       |                   | flavor   | v1b, v1d                                                   |
|          |                       |                   | dataset  | imagenet                                                   |
| NLP      | Question and Answer   | BertQA            | backbone | bert                                                       |
|          |                       |                   | dataset  | book_corpus_wiki_en_uncased                                |

**Note:** Not all combinations in the above table are available. For more information, see the `metadata.json` file
in the `src/test/resources/mlrepo/model` folder.

## Contributor Guides and Documentation

### [How to add new models to the model zoo](../../docs/development/add_model_to_model-zoo.md)
