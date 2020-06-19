# Documentation

This folder contains examples and documentation for the Deep Java Library (DJL) project.

## Modules

- [Core API](../api/README.md) - Engine-agnostic API
- [Basic Dataset](../basicdataset/README.md) - Built-in datasets
- [Model Zoo](../model-zoo/README.md) - Built-in engine-agnostic model zoo

#### MXNet
- [MXNet Engine](../mxnet/mxnet-engine/README.md) - MXNet Engine implementation
- [MXNet Model Zoo](../mxnet/mxnet-model-zoo/README.md) - MXNet symbolic model zoo
- [MXNet Gluon Importing](mxnet/how_to_convert_your_model_to_symbol.md) - MXNet Gluon Importing

#### PyTorch
- [PyTorch Engine](../pytorch/pytorch-engine/README.md) - PyTorch Engine implementation
- [PyTorch Model Zoo](../pytorch/pytorch-model-zoo/README.md) - PyTorch TorchScript model zoo
- [PyTorch Importing](pytorch/how_to_convert_your_model_to_torchscript.md) - Import TorchScript model
- [Pytorch Inference Optimization](pytorch/how_to_optimize_inference_performance.md) - Improve inference performance


#### TensorFlow
- [TensorFlow Engine](../tensorflow/tensorflow-engine/README.md) - TensorFlow Engine implementation
- [TensorFlow Model Zoo](../tensorflow/tensorflow-model-zoo/README.md) - TensorFlow SavedModel model zoo
- [Keras Imporint](tensorflow/how_to_import_keras_models_in_DJL.md)
- [Tensorflow Local Importing](tensorflow/how_to_import_local_tensorflow_models.md)


#### ONNX Runtime
- [Hybrid Engine Operations](onnxruntime/hybrid_engine.md) - Using a second Engine for supplemental API support

## [JavaDoc API Reference](https://javadoc.djl.ai/)

## [Jupyter notebook tutorials](../jupyter)

- **[Beginner Jupyter Tutorial](../jupyter/tutorial)**
- [Run object detection with model zoo](../jupyter/object_detection_with_model_zoo.ipynb)
- [Load pre-trained PyTorch model](../jupyter/load_pytorch_model.ipynb)
- [Load pre-trained MXNet model](../jupyter/load_mxnet_model.ipynb)
- [Transfer learning example](../jupyter/transfer_learning_on_cifar10.ipynb)
- [Question answering example](../jupyter/BERTQA.ipynb)

## [API Examples](../examples/README.md)

- [Single-shot Object Detection example](../examples/docs/object_detection.md)
- [Train your first model](../examples/docs/train_mnist_mlp.md)
- [Image classification example](../examples/docs/image_classification.md)
- [Transfer learning example](../examples/docs/train_cifar10_resnet.md)
- [Train SSD model example](../examples/docs/train_pikachu_ssd.md)
- [Multi-threaded inference example](../examples/docs/multithread_inference.md)
- [Bert question and answer example](../examples/docs/BERT_question_and_answer.md)
- [Instance segmentation example](../examples/docs/instance_segmentation.md)
- [Pose estimation example](../examples/docs/pose_estimation.md)
- [Action recognition example](../examples/docs/action_recognition.md)

## Cheat sheet

- [How to load a model](load_model.md)
- [How to collect metrics](how_to_collect_metrics.md)
- [How to use a dataset](development/how_to_use_dataset.md)

## [Memory Management](development/memory_management.md)

## [Contributor Documentation](development/README.md)

## [FAQ](faq.md)
