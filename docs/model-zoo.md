# DJL - Model Zoo

Deep Java Library's (DJL) Model Zoo is more than a collection of pre-trained models. It's a bridge between model vendor and consumer.

We provide a framework for developers to create and publish their model in an organized way. A ZooModel has the following
characteristic:
- Globally unique: similar to java maven package, a model has it's own group ID and artifact ID that uniquely identify it.
- A ZooModel is versioned: the model version scheme allows developer to continuously update their model without causing a backward compatibility issue.
- A ZooModel is ready to use out of box: the model contains predefined pre-process and post-process functionality, which
allows the user to run inference with a plain java object. 
- A ZooModel can be published to anywhere, whether its a S3 bucket, a web server, or a local folder.

## [DJL model zoo](../model-zoo/README.md)

We provide framework agnostic `ZooModel`s in our model zoo. They can be used on any DJL backend engine.

## [MXNet symbolic model zoo](../mxnet/mxnet-model-zoo/README.md)

MXNet has a large volume of existing pre-trained models. We created a MXNet model zoo to make it easy for users to consume them.

## Publish your own model zoo
You can also create your own model zoo so your customers can easily consume it.
See [here](development/add_model_to_model-zoo.md) for detail.
