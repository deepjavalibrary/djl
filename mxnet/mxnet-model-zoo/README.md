# MXNet model zoo

## Introduction

MXNet model zoo contains most of the Symbolic model that can be used for inference and training.

All of the models in this model zoo contain pre-trained parameters for their specific datasets.

## Pre-trained models

MXNet model zoo contains two major categories, CV and NLP. All of the models are grouped by their application under these categories:

* CV
  * Action Recognition
  * Image Classification
  * Object Detection
  * Pose Estimation
  * Semantic Segmentation/Instance Segmentation
* NLP
  * Question and Answer

## How to find a pre-trained model in model zoo

In a model zoo repository, there can be many pre-trained that belongs to the same model family.
You can use the `MxModelZoo` class to search for the model that you need.
First, you need to decide which model family you want to use, then define key/values search criteria
to narrow down the model you want. If there are multiple models that match your search criteria, the first
model found will be returned. *ModelNotFoundException* will be thrown if no matching model is found.

The following is an example of criteria to find a resnet50-v1 model that is trained on the imagenet dataset:
```java
    Map<String, String> criteria = new HashMap<>();
    criteria.put("layers", "50");
    criteria.put("flavor", "v1");
    criteria.put("dataset", "imagenet");

    ZooModel<BufferedImage, Classification> model = MxModelZoo.RESNET.loadModel(criteria, device);
``` 

## List of search criteria for each model

| Category | Application           | Model Family      | Criteria | Possible values                                            |
|----------|-----------------------|-------------------|----------|------------------------------------------------------------|
| CV       | Action Recognition    | ActionRecognition | backbone | vgg16, inceptionv3                                         |
|          |                       |                   | dataset  | ucf101                                                     |
|          | Image Classification  | MLP               | dataset  | mnist                                                      |
|          |                       | Resnet            | layers   | 18, 34, 50, 101, 152                                       |
|          |                       |                   | flavor   | v1, v2, v1b, v1c, v1d, v1e, v1s                            |
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

**Note:** Not all combinations in the above table are available. Please check `metadata.json` files
in `src/test/resources/mlrepo/model` folder for detail.

## Developer guide

### [How to add new models to model zoo](../../docs/development/add_model_to_model-zoo.md)
