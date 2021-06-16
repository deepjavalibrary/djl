# Import TensorFlow models in DJL

This document will how you how to import various TensorFlow models in DJL. DJL only supports loading TensorFlow models in 
SavedModel format, we will walk you through the steps to convert other model formats to SavedModel.

## How to import Keras models in DJL

In DJL TensorFlow engine and model zoo, only [SavedModel](https://www.tensorflow.org/guide/saved_model) format (.pb files)
is supported. However, many Keras users save their model using [keras.model.save](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save) API
and it produce a .h5 file.

This document shows you how to convert a .h5 model file into TensorFlow SavedModel(.pb) file so it can be imported in DJL.
All the code here are Python code, you need to install [TensorFlow for Python](https://www.tensorflow.org/install/pip).

For example, if you have a ResNet50 with trained weight, you can directly save it in SavedModel format using `tf.saved_model.save`.
Here we just use pre-trained weights of ResNet50 from Keras Applications:

```python
import tensorflow as tf
import tensorflow.keras as keras
resnet = keras.applications.ResNet50()
tf.saved_model.save(resnet, "resnet/1/")
```

However, if you already saved your model in .h5 file using `keras.model.save` like below, you need a simple python script
to convert it to SavedModel format.

```python
resnet.save("resnet.h5")
```

Just load your .h5 model file back to Keras and save it to SavedModel:

```python
loaded_model = keras.models.load_model("resnet.h5")
tf.saved_model.save(loaded_model, "resnet/1/")
```

Once you have a SavedModel, you can load your Keras model using DJL TensorFlow engine.
Refer to [How to load models in DJL](../load_model.md).


## How to import [TensorFlow Hub](https://tfhub.dev/) models

DJL support loading models directly from TensorFlow Hub, you can use the `optModelUrls` method in `Critera.Builder` to specify the model URL.

The URL can be a http link that points to the TensorFlow Hub models, or local path points to the model downloaded from TensorFlow Hub.
Note that you need to click the download model button to find the actual Google Storage link that hosts the TensorFlow Hub model.

Please refer to these two examples:

1. [Object Detection with TensorFlow](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/ObjectDetection.java) for loading from TensorFlow Hub url.
2. [BERT Classification](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/BertClassification.java) for loading from local downloaded model.

## How to load TensorFlow Checkpoints

To load an TensorFlow Estimator checkpoint, you need to convert it to SavedModel format in using Python.
1. Load the checkpoint file back as an Estimator using Python API.
2. Export the Estimator to SavedModel using the 
[Estimator.export_saved_model](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#export_saved_model) API.

Here is an example:

```python
# define model_fn, serving_input_receiver_fn based on training script

estimator = tf.estimator.Estimator(model_fn, "./checkpoint_dir")
estimator.export_saved_model("./export_dir", serving_input_receiver_fn)
```

For more information on saving, loading and exporting checkpoints, please refer to TensorFlow [documentation](https://www.tensorflow.org/guide/checkpoint).


## How to load DJL TensorFlow model zoo models

The steps are the same as loading any other DJL model zoo models, you can use the `Criteria` API as documented [here](https://docs.djl.ai/docs/load_model.html#criteria-class).

Note for TensorFlow image classification models, you need to manually specify the translator instead of using the built-in one because
TensorFlow requires channels last ("NHWC") image formats while DJL use channels first ("NCHW") image formats. By default, DJL will add
`ToTensor` transformation in the translator, which will convert the image to channels first format. TensorFlow models does not need this step. 

Here is an example:

```java
ImageClassificationTranslator myTranslator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(224, 224))
                        .addTransform(array -> array.div(127.5f).sub(1f))
                        .build();

Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optTranslator(myTranslator)
                        .optFilter("flavor", "v2")
                        .optProgress(new ProgressBar())
                        .build();

ZooModel<Image, Classifications> model = criteria.loadModel();
```

## How to import TensorFlow 1.x models

DJL supports TensorFlow models trained using both 1.x and 2.x. When loading the model we need to know the Tags name saved 
in the SavedModel(.pb) file. By default, the tag name for 1.x models is "" (an empty String), and for 2.x models it's "serve".
DJL by default will use "serve" to load the model. 

You will also need to know what's the [SignatureDefs](https://www.tensorflow.org/tfx/serving/signature_defs) needed, they can be specified by SignatureDefKeys.
By default, DJL will use `"serving_default"` as the key which is common for TF 2.x models. 
Most TF 1.x models use "default" as the key.

In summary, you need to specify "Tags" and "SignatureDefKey" using `optOption` or `optOptions` when loading 1.x models in `Criteria`.

Here is an example loading SSD model trained in TensorFlow 1.x:

```java
Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", "mobilenet_v2")
                        .optEngine("TensorFlow")
                        .optOption("Tags", "")
                        .optOption("SignatureDefKey", "default")
                        .optProgress(new ProgressBar())
                        .build();

ZooModel<Image, DetectedObjects> model = criteria.loadModel();
```

## Tips and tricks when writing Translator for TensorFlow models

It's very common you have to write your own translator for a new TensorFlow model, you can follow these steps:

- Identify the required model inputs and outputs shapes, orders and names:

You can find out by either calling the `model.describeInput` and `model.describeOutput` methods after you load the model,
or using the [Saved Model CLI tool](https://www.tensorflow.org/guide/saved_model#install_the_savedmodel_cli).
It will show you the number of inputs and number of outputs required, their names, and orders.

- Write `processInput` and `processOutput` methods:

The input of `processInput` can be any Object and output must be `NDList` for DJL to run inference.
You can convert your inputs to NDArrays in the order described by `model.describeInput()`,
and add names for each NDArray using `NDArray.setName` method.
Then you can assemble the NDArrays into a NDList and return it to the `Predictor`.

Similarly, `processOutput` takes a NDList as input and convert it back to any Object.
For models with multiple outputs, you already know the outputs orders by calling `model.describeOutput` in step 1.
You can also use `NDArray.getName` to figure out what does the output represent.
