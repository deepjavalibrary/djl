
![DeepJavaLibrary](website/img/deepjavalibrary.png?raw=true "Deep Java Library")

![Continuous](https://github.com/deepjavalibrary/djl/workflows/Continuous/badge.svg)
![Continuous PyTorch](https://github.com/deepjavalibrary/djl/workflows/Continous%20PyTorch/badge.svg)
![Continuous Tensorflow](https://github.com/deepjavalibrary/djl/workflows/Continuous%20Tensorflow/badge.svg)
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
                    .optApplication(Application.CV.OBJECT_DETECTION) // find object dection model
                    .setTypes(Image.class, Classifications.class) // define input and output
                    .optFilter("backbone", "resnet50") // choose network architecture
                    .build();

    try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
        try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Image img = ImageFactory.getInstance().fromUrl("http://..."); // read image
            Classifications result = predictor.predict(img);

            // get the classification and probability
            ...
        }
    }
```

The following pseudocode demonstrates running training:

```java
    // Construct your neural network with built-in blocks
    Block block = new Mlp(28, 28);

    try (Model model = Model.newInstance("mlp")) { // Create an empty model
        model.setBlock(block); // set neural network to model

        // Get training and validation dataset (MNIST dataset)
        Dataset trainingSet = new Mnist.Builder().setUsage(Usage.TRAIN) ... .build();
        Dataset validateSet = new Mnist.Builder().setUsage(Usage.TEST) ... .build();

        // Setup training configurations, such as Initializer, Optimizer, Loss ...
        TrainingConfig config = setupTrainingConfig();
        try (Trainer trainer = model.newTrainer(config)) {
            /*
             * Configure input shape based on dataset to initialize the trainer.
             * 1st axis is batch axis, we can use 1 for initialization.
             * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
             */
            Shape inputShape = new Shape(1, 28 * 28);
            trainer.initialize(new Shape[] {inputShape});

            EasyTrain.fit(trainer, epoch, trainingSet, validateSet);
        }

        // Save the model
        model.save(modelDir, "mlp");
    }
```

## [Getting Started](docs/quick_start.md)

## Resources

- [Documentation](docs/README.md#documentation)
- [DJL's D2L Book](https://d2l.djl.ai/)
- [JavaDoc API Reference](https://javadoc.djl.ai/)

## Release Notes
* [0.12.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.12.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.12.0))
* [0.11.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.11.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.11.0))
* [0.10.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.10.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.10.0))
* [0.9.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.9.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.9.0))
* [0.8.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.8.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.8.0))
* [0.6.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.6.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.6.0))
* [0.5.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.5.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.5.0))
* [0.4.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.4.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.4.0))
* [0.3.0](https://github.com/deepjavalibrary/djl/releases/tag/v0.3.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.3.0))
* [0.2.1](https://github.com/deepjavalibrary/djl/releases/tag/v0.2.1) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.2.1))
* [0.2.0 Initial release](https://github.com/deepjavalibrary/djl/releases/tag/v0.2.0) ([Code](https://github.com/deepjavalibrary/djl/tree/v0.2.0))

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
