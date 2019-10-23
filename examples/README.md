djl.ai - examples
=================

## Overview

The djl.ai API is designed to be an extremely easy to use deep learning framework for Java
developers. djl.ai does not require you to be a Machine Learning/Deep Learning expert to get
started. You can use your existing Java expertise as an on-ramp to learn and use ML/DL. You can
use your favorite IDE to build/train/deploy your models and integrate these models with your
Java applications.

djl.ai API is deep learning framework agnostic, so you don't have to make a choice
between frameworks when starting your project. You can switch to a different framework at any
time you want. djl.ai also provides automatic CPU/GPU choice based on the hardware configuration
to ensure the best performance.

djl.ai API provides native Java development experience. It functions similarly to any other Java library.
djl.ai's ergonomic API interface is designed to guide you with best practices to accomplish your
deep learning task.

The following is an example of how to write inference code:

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

## djl.ai API reference

You can find more information here: [Javadoc](https://djl-ai.s3.amazonaws.com/java-api/0.1.0/index.html)

## Examples project

Read [Examples project](examples.md) for more detail about how to setup development environment and dependencies.

You can also read individual examples: 

1. [Image classification example](CLASSIFY.md)
2. [Single-shot Object detection example](SSD.md)
3. [Bert question and answer example](BERTQA.md)
