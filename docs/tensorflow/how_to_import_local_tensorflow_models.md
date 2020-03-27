## How to import local TensorFlow models in DJL

DJL TensorFlow model zoo comes with pre-trained TensorFlow models and you can easily import them to run inference.
However, there are many models currently not covered by DJL TensorFlow model zoo, and you may have custom models with 
weights trained on your own dataset. This document shows you how to load a local TensorFlow model in DJL.

In general, importing a local model is the same for all engines in DJL. For local TensorFlow models, you just need
specify the path to your [SavedModel](https://www.tensorflow.org/guide/saved_model). Currently SavedModel is the only format
supported in DJL TensorFlow engine.

For example, if you have a ResNet50 in SavedModel format, it will have the following folder structure:

```bash
resnet50
└── 1
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

Then you just need to pass the directory using `model.load()`, here is an example code snippet.

```java
try (Model model = Model.newInstance()) {
    model.load(Paths.get("resnet50/1/"));
    // create new predictor
    try (Predictor<BufferedImage, Classifications> predictor =
                    model.newPredictor(myTranslator)) {
        // run prediction on image file
        Classifications result =
            predictor.predict(BufferedImageUtils.fromFile(Paths.get("cat.jpg")));
        // show result
        // ...
    }
}
```

Simple as that! You just need to specify the path. 
For details and full code, please refer to [ImageClassification.java](../../examples/src/main/java/ai/djl/examples/inference/ImageClassification.java).
