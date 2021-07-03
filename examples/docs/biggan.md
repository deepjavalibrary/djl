# Image Generation with BigGAN from the Model Zoo

[Generative Adversarial Networks](https://en.wikipedia.org/wiki/Generative_adversarial_network) (GANs) are a branch of deep learning used for generative modeling. 
They consist of 2 neural networks that act as adversaries, the Generator and the Discriminator. The Generator is assigned to generated fake images that look real, and the Discriminator needs to correctly identify the fake ones.

In this example, you will learn how to use a [BigGAN](https://deepmind.com/research/open-source/biggan) generator to create images, using the generator directly from the [ModelZoo](../../docs/model-zoo.md).

The source code for this example can be found at [BigGAN.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/BigGAN.java).

## Setup guide

To configure your development environment, follow [setup](../../docs/development/setup.md).

## Run Generation

### Introduction 

BigGAN is trained on a subset of the [ImageNet dataset](https://en.wikipedia.org/wiki/ImageNet) with 1000 categories.
You can see the labels in [this file](https://github.com/deepjavalibrary/djl/blob/master/model-zoo/src/test/resources/mlrepo/model/cv/image_classification/ai/djl/zoo/synset_imagenet.txt).
The training was done such that the input to the model uses the ID of the category, between 0 and 999. For us, the ID is the line number in the file, starting at 0. 

Thus, the input to the translator will be an array of category IDs:

```java
int[] input = {100, 207, 971, 970, 933};
```

### Build the project and run
Use the following commands to run the project:

```
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.BigGAN -Dai.djl.default_engine=PyTorch
```

### Output

Your output will vary since the generation depends on a random seed. Here are a few examples:

Black Swan                 |  Golden Retriever          |  Bubble |  Alp  |  Cheeseburger
:-------------------------:|:-------------------------: |:-------------------------: | :----------------------: | :----------------------:
![]( https://resources.djl.ai/images/biggan/black-swan.png) | ![]( https://resources.djl.ai/images/biggan/golden-retriever.png)| ![]( https://resources.djl.ai/images/biggan/bubble.png) | ![]( https://resources.djl.ai/images/biggan/hills.png) | ![]( https://resources.djl.ai/images/biggan/cheeseburger.png)
