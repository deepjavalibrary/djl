
![DeepJavaLibrary](docs/website/img/deepjavalibrary.png?raw=true "Deep Java Library")

![](https://github.com/awslabs/djl/workflows/nightly%20build/badge.svg)

# Deep Java Library (DJL)

## Overview

DJL is designed to be extremely easy to get started and simple to
use deep learning framework for Java developers. DJL does not required user to be ML/DL experts to get started
and can start from their existing Java expertise as an on-ramp to learn and use ML/DL. They can
use their favorite IDE to build/train/deploy their models and integrate these models with their
Java applications.

DJL is deep learning engine agnostic, developer does not have to make a choice
between framework while they started their project. They can switch to different framework at any
time they want. DJL also provides automatic CPU/GPU chosen based on the hardware configuration to ensure the best performance.

DJL provide native Java development experience, just another regular java library.
DJL's ergonomic API interface is designed to guide developer with best practice to accomplish
deep learning task.

The following is pseudo code of how to write inference code:

```java
    // Assume user uses a pre-trained model from model zoo, they just need to load it
    Map<String, String> criteria = new HashMap<>();
    criteria.put("layers", "18");
    criteria.put("flavor", "v1");

    // Load pre-trained model from model zoo
    try (Model<BufferedImage, Classifications> model = MxModelZoo.RESNET.loadModel(criteria)) {
        try (Predictor<BufferedImage, Classifications> predictor = model.newPredictor()) {
            BufferedImage img = readImage(); // read image
            Classifications result = predictor.predict(img);

            // get the classification and probability
            ...
        }
    }
```

The following is pseudo code of how to write training code:

```java
    // Construct your neural network with built-in blocks
    Block block = new Mlp(28, 28);

    try (Model model = Model.newInstance()) { // Create an empty model
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

            TrainingUtils.fit(trainer, epoch, trainingSet, validateSet);
        }

        // Save the model
        model.save(modelDir, "mlp");
    }
```


## Release Notes
* 0.1.0 Initial release

## Building From Source

Once you check out the code, you can build it using gradle:

```sh
./gradlew build
```

If you want to skip unit test:
```sh
./gradlew build -x test
```

**Note:** SpotBugs is not compatible with JDK 11+, SpotBugs will not be executed if you are using JDK 11+.

## License

This project is licensed under the Apache-2.0 License.
