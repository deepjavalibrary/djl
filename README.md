djl.ai - Deeplearning Java Library
==================================

## Overview

djl.ai API is designed to be extremely easy to get started and simple to
use deep learning framework for Java developers. djl.ai does not required user to be ML/DL experts to get started
and can start from their existing Java expertise as an on-ramp to learn and use ML/DL. They can
use their favorite IDE to build/train/deploy their models and integrate these models with their
Java applications.

djl.ai API is deep learning framework agnostic, developer does not have to make a choice
between framework while they started their project. They can switch to different framework at any
time they want. djl.ai also provides automatic CPU/GPU chosen based on the hardware configuration to ensure the best performance.

djl.ai API provide native Java development experience, just another regular java library.
djl.ai's ergonomic API interface is designed to guide developer with best practice to accomplish
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
    Block block = new ResnetV1();

    // setup your training configurations, such as Initializer, Optimizer, Loss ...
    TrainingConfig config = setupTrainingConfig();

    // configure inputShape based on batch size and number of GPU
    Shape inputShape = new Shape(batchSize / numGpu, 28 * 28);

    try (Model model = Model.newInstance()) { // Create an empty model
        model.setBlock(block); // set neural network to model

        // Prepare training and validating data set
        Mnist trainSet = new Mnist.Builder().setUsage(Usage.TRAIN).build();
        Mnist validateSet = new Mnist.Builder().setUsage(Usage.VALIDATION).build();

        try (Trainer trainer = model.newTrainer(config)) { // Create training session
            trainer.init(new DataDesc[] {new DataDesc(inputShape)}); // initialize trainer

            // Train the model with train/validate dataset             
            TrainingUtils.fit(trainer, trainSet, validateSet);
        }

        // Save the model
        model.save(modelDir, "myMnist");
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

**Note:** Due to SpotBugs issue, default build will fail if you are using JDK 11+.
You can skip SpotBugs if you are using JDK 11+:
```sh
./gradlew build -x spotBugsMain -x spotBugsTest
```
