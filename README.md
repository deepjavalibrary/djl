
![DeepJavaLibrary](website/img/deepjavalibrary.png?raw=true "Deep Java Library")

![Continuous](https://github.com/deepjavalibrary/djl/workflows/Continuous/badge.svg)
![Docs](https://github.com/deepjavalibrary/djl/workflows/Docs/badge.svg)
![Nightly Publish](https://github.com/deepjavalibrary/djl/workflows/Nightly%20Publish/badge.svg)

# Deep Java Library (DJL)

## Overview

Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework for deep learning. DJL is designed to be easy to get started with and simple to
use for Java developers. DJL provides a native Java development experience and functions like any other regular Java library.

You don't have to be machine learning/deep learning expert to get started. You can use your existing Java expertise as an on-ramp to learn and use machine learning and deep learning. You can
use your favorite IDE to build, train, and deploy your models. DJL makes it easy to integrate these models with your
Java applications.

Because DJL is deep learning engine agnostic, you don't have to make a choice
between engines when creating your projects. You can switch engines at any
point. To ensure the best performance, DJL also provides automatic CPU/GPU choice based on hardware configuration.

DJL's ergonomic API interface is designed to guide you with best practices to accomplish
deep learning tasks.
The following pseudocode demonstrates running inference:

```java
    // Assume user uses a pre-trained model from model zoo, they just need to load it
    Criteria<Image, Classifications> criteria =
            Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION) // find object detection model
                    .setTypes(Image.class, Classifications.class)    // define input and output
                    .optFilter("backbone", "resnet50")               // choose network architecture
                    .build();

    Image img = ImageFactory.getInstance().fromUrl("http://...");    // read image
    try (ZooModel<Image, Classifications> model = criteria.loadModel();
         Predictor<Image, Classifications> predictor = model.newPredictor()) {
        Classifications result = predictor.predict(img);

        // get the classification and probability
        ...
    }
```

The following pseudocode demonstrates running training:

```java
    // Construct your neural network with built-in blocks
    Block block = new Mlp(28 * 28, 10, new int[] {128, 64});

    Model model = Model.newInstance("mlp"); // Create an empty model
    model.setBlock(block);                  // set neural network to model

    // Get training and validation dataset (MNIST dataset)
    Dataset trainingSet = new Mnist.Builder().setUsage(Usage.TRAIN) ... .build();
    Dataset validateSet = new Mnist.Builder().setUsage(Usage.TEST) ... .build();

    // Setup training configurations, such as Initializer, Optimizer, Loss ...
    TrainingConfig config = setupTrainingConfig();
    Trainer trainer = model.newTrainer(config);
    /*
     * Configure input shape based on dataset to initialize the trainer.
     * 1st axis is batch axis, we can use 1 for initialization.
     * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
     */
    trainer.initialize(new Shape(1, 28 * 28));
    EasyTrain.fit(trainer, epoch, trainingSet, validateSet);

    // Save the model
    model.save(modelDir, "mlp");

    // Close the resources
    trainer.close();
    model.close();
```

## [Getting Started](docs/quick_start.md)

## Resources

- [Documentation](docs/README.md#documentation)
- [DJL's D2L Book](https://d2l.djl.ai/)
- [JavaDoc API Reference](https://djl.ai/website/javadoc.html)

## Release Notes

* [0.29.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.29.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.29.0))
* [0.28.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.28.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.28.0))
* [0.27.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.27.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.27.0))
* [0.26.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.26.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.26.0))
* [0.25.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.25.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.25.0))
* [0.24.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.24.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.24.0))
* [+25 releases](https://github.com/deepjavalibrary/djl/releases)

The release of DJL 0.30.0 is planned for Sep 2024.

## Building From Source

To build from source, begin by checking out the code.
Once you have checked out the code locally, you can build it as follows using Gradle:

```sh
# for Linux/macOS:
./gradlew build

# for Windows:
gradlew build
```

To increase build speed, you can use the following command to skip unit tests:

```sh
# for Linux/macOS:
./gradlew build -x test

# for Windows:
gradlew build -x test
```

### Importing into eclipse

to import source project into eclipse

```sh
# for Linux/macOS:
./gradlew eclipse

# for Windows:
gradlew eclipse

```

in eclipse 

file->import->gradle->existing gradle project

**Note:** please set your workspace text encoding setting to UTF-8

## Community

You can read our guide to [community forums, following DJL, issues, discussions, and RFCs](docs/forums.md) to figure out the best way to share and find content from the DJL community.

Join our [<img src='https://cdn3.iconfinder.com/data/icons/social-media-2169/24/social_media_social_media_logo_slack-512.png' width='20px' /> slack channel](http://tiny.cc/djl_slack) to get in touch with the development team, for questions and discussions.

Follow our [<img src='https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Twitter_colored_svg-512.png' width='20px' /> twitter](https://twitter.com/deepjavalibrary) to see updates about new content, features, and releases.

关注我们 [<img src='https://www.iconfinder.com/icons/5060515/download/svg/512' width='20px' /> 知乎专栏](https://zhuanlan.zhihu.com/c_1255493231133417472) 获取DJL最新的内容！

## Useful Links

* [DJL Website](https://djl.ai/)
* [Documentation](https://docs.djl.ai/)
* [DJL Demos](https://docs.djl.ai/docs/demos/index.html)
* [Dive into Deep Learning Book Java version](https://d2l.djl.ai/)

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
