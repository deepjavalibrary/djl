Joule - Deep learning for Java
==============================

## Overview

Joule API is designed to be extremely easy to get started and simple to
use deep learning framework for Java developers. Joule does not required user to be ML/DL experts to get started
and can start from their existing Java expertise as an on-ramp to learn and use ML/DL. They can
use their favorite IDE to build/train/deploy their models and integrate these models with their
Java applications. 

Joule API is deep learning framework agnostic, developer does not have to make a choice
between framework while they started their project. They can switch to different framework at any
time they want. Joule also provides automatic CPU/GPU chosen based on the hardware configuration to ensure the best performance.

Joule API provide native Java development experience, just another regular java library.
Joule's ergonomic API interface is designed to guide developer with best practice to accomplish
deep learning task.

The following is an example of how to write inference code:

```java
    // Assume user has a pre-trained already, they just need load it
    Model model = Model.loadModel(modelDir, modelName);

    // User must implement Translator interface, read Translator document for detail.
    Translator translator = new MyTranslator();

    // User can specify GPU/CPU Context to run inference session.
    // This context is optional, Predictor can pick up default Context if not specified.
    // See Context.defaultContext()
    Context context = Context.defaultContext();

    // Next user need create a Predictor, and use Predictor.predict()
    // to get prediction.
    try (Predictor<BufferedImage, List<DetectedObject>> predictor =
            Predictor.newInstance(model, translator, context)) {
        List<DetectedObject> result = predictor.predict(img);
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
