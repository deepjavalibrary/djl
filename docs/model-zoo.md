# Model Zoo

Deep Java Library's (DJL) Model Zoo is more than a collection of pre-trained models. It's a bridge between a model vendor and a consumer. It provides a framework for developers to create and publish their own models. 

A [ZooModel](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/ZooModel.html) has the following characteristics:

- Globally unique: similar to Java maven packages, a model has its own group ID and artifact ID that uniquely identify it.
- Versioned: the model version scheme allows developers to continuously update their model without causing a backward compatibility issue.
- Ready to use out of box: the model contains predefined pre-process and post-process functionality, which
allows the user to run inference with a plain java object. 
- Can be published anywhere: models can be published to an S3 bucket, a web server, or a local folder.

## [Basic model zoo](../model-zoo/README.md)

We provide engine agnostic `ZooModel`s in basic model zoo package. They can be used on any DJL backend engine.

## [Huggingface model zoo](../extension/tokenizers/README.md)

We created a Huggingface model zoo to make it easy for users to consume them.

## [PyTorch model zoo](../engines/pytorch/pytorch-model-zoo/README.md)

We created a PyTorch model zoo to make it easy for users to consume them.

## [TensorFlow model zoo](../engines/tensorflow/tensorflow-model-zoo/README.md)

We created an TensorFlow model zoo to make it easy for users to consume them.

## [MXNet symbolic model zoo](../engines/mxnet/mxnet-model-zoo/README.md)

Apache MXNet has a large number of existing pre-trained models. We created an Apache MXNet model zoo to make it easy for users to consume them.


## Publish your own model to the model zoo

You can create your own model in the model zoo so customers can easily consume it.
For more information, see [Add a new Model to the model zoo ](development/add_model_to_model-zoo.md).

## Load models from ModelZoo

See: [How to load model](load_model.md)
