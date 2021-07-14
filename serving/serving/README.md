# DJL Serving

## Overview

DJL Serving is an universal model serving solution. You can use djl-serving serve the
following models out of the box:

- PyTorch TorchScript model
- TensorFlow SavedModel bundle
- Apache MXNet model

You can install extra extensions to enable the following models:

- ONNX model
- PaddlePaddle model
- TFLite model
- Neo DLR (TVM) model
- XGBoost model
- Sentencepiece model
- fastText/BlazingText model

## Installation

For macOS (Working in progress)

```
brew install djl-serving

# Start djl-serving as service:
brew services start djl-serving

# Stop djl-serving service
brew services stop djl-serving
```

For Ubuntu

```
curl -O https://publish.djl.ai/djl-serving/djl-serving_0.12.0-1_all.deb
sudo dpkg -i djl-serving_0.12.0-1_all.deb
```

For Windows

We are considering to create a `chocolatey` package for Windows. For the time being, we can run
download djl-serving.zip file from [here](https://publish.djl.ai/djl-serving/serving-0.12.0.zip).

### Docker

You can also use docker to run DJL Serving:

```
docker run -itd -p 8080:8080 deepjavalibrary/djl-serving
```

## Run DJL Serving

Use the following command to start model server locally:

```sh
djl-serving
```

The model server will be listening on port 8080. You can also load a model for serving on start up:

```sh
djl-serving -m "https://resources.djl.ai/demo/mxnet/resnet18_v1.zip"
```

Open another terminal, and type the following command to test the inference REST API:

```sh
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST http://localhost:8080/predictions/resnet18_v1 -T kitten.jpg

or:

curl -X POST http://localhost:8080/predictions/resnet18_v1 -F "data=@kitten.jpg"

[
  {
    "className": "n02123045 tabby, tabby cat",
    "probability": 0.4838452935218811
  },
  {
    "className": "n02123159 tiger cat",
    "probability": 0.20599420368671417
  },
  {
    "className": "n02124075 Egyptian cat",
    "probability": 0.18810515105724335
  },
  {
    "className": "n02123394 Persian cat",
    "probability": 0.06411745399236679
  },
  {
    "className": "n02127052 lynx, catamount",
    "probability": 0.010215568356215954
  }
]
```

For more command line options:

```sh
djl-serving --help
usage: djl-serving [OPTIONS]
 -f,--config-file <CONFIG-FILE>    Path to the configuration properties file.
 -h,--help                         Print this help.
 -m,--models <MODELS>              Models to be loaded at startup.
 -s,--model-store <MODELS-STORE>   Model store location where models can be loaded.
```

## REST API

DJL Serving use RESTful API for both inference and management calls.

When DJL Serving startup, it starts two web services:
* [Inference API](docs/inference_api.md)
* [Management API](docs/management_api.md)

By default, DJL Serving listening on 8080 port and only accessible from localhost.
Please see [DJL Serving Configuration](docs/configuration.md) for how to enable access from remote host.

# Plugin management

DJL Serving supports plugins, user can implement their own plugins to enrich DJL Serving features.
See [DJL Plugin Management](docs/plugin_management.md) for how to install plugins to DJL Serving.

## Logging
you can set the logging level on the command-line adding a parameter for the JVM

```sh
-Dai.djl.logging.level={FATAL|ERROR|WARN|INFO|DEBUG|TRACE}
```
