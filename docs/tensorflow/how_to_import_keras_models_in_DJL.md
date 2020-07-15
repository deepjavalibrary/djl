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
