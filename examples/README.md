Joule - examples
================

## Overview

The Joule API is designed to be an extremely easy to use deep learning framework for Java developers. Joule does not require you to be a Machine Learning/Deep Learning expert to get started. You can use your existing Java expertise as an on-ramp to learn and use ML/DL. You can
use your favorite IDE to build/train/deploy your models and integrate these models with your
Java applications. 

Joule API is deep learning framework agnostic, so you don't have to make a choice
between frameworks when starting your project. You can switch to a different framework at any
time you want. Joule also provides automatic CPU/GPU choice based on the hardware configuration to ensure the best performance.

Joule API provides native Java development experience. It functions similarly to any other Java library.
Joule's ergonomic API interface is designed to guide you with best practices to accomplish your
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

## Joule API reference

You can find more information here: [Javadoc](https://joule.s3.amazonaws.com/java-api/index.html)

## Examples project

Read [Examples project](examples.md) for more detail about how to setup development environment and dependencies.

You can also read individual examples: 

1. [Image classification example](CLASSIFY.md)
2. [Single-shot Object detection example](SSD.md)
3. [Bert question and answer example](BERTQA.md)
