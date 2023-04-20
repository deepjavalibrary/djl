# Spark Support for DJL

## Overview

This module contains the Spark support extension, which allows DJL to be used seamlessly with [Apache Spark](https://spark.apache.org/).

Some key features of the DJL Spark Extension include:

- Easy integration with Apache Spark: The DJL Spark Extension provides a simple and intuitive API for integrating DJL with Apache Spark, allowing Java developers to easily use DJL in their Spark applications.

- Distributed inference: The DJL Spark Extension allows developers to easily scale their deep learning models to large datasets by leveraging the distributed processing power of Apache Spark.

- Support for popular deep learning engines: The DJL Spark Extension provides support for popular deep learning frameworks such as MXNet, PyTorch, TensorFlow and ONNXRuntime, allowing developers to use their preferred framework when working with Spark and DJL.

- Support for PySpark: The DJL Spark Extension provides support for PySpark, allowing developers to use DJL in their PySpark applications.

- Support for other popular libraries and frameworks: The DJL Spark Extension provides support for other popular libraries and frameworks, such as HuggingFace tokenizers.

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.spark/spark/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```

## Installation

You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.spark</groupId>
    <artifactId>spark_2.12</artifactId>
    <version>0.22.1</version>
</dependency>
```

## Usage

Using the DJL Spark Extension is simple and straightforward. Here is an example of how to use it to run image classification on a large dataset using Apache Spark and DJL:

### Scala

```scala
import ai.djl.spark.SparkTransformer
import ai.djl.spark.translator.SparkImageClassificationTranslator

val transformer = new SparkTransformer[Classifications]()
  .setInputCols(Array("input_col1", "input_col2"))
  .setOutputCols(Array("value"))
  .setEngine("PyTorch")
  .setModelUrl("model_url")
  .setOutputClass(classOf[Classifications])
  .setTranslator(new SparkImageClassificationTranslator())
val outputDf = transformer.transform(df)
```

### Python

```python
from djl_spark.transformer import SparkTransformer
from djl_spark.translator import SparkImageClassificationTranslator

transformer = SparkTransformer(input_cols=["input_col1", "input_col2"],
                               output_cols=["value"],
                               engine="PyTorch",
                               model_url="model_url",
                               output_class="ai.djl.modality.Classifications",
                               translator=SparkImageClassificationTranslator())
outputDf = transformer.transform(df)
```

See [examples](https://github.com/deepjavalibrary/djl-demo/tree/master/apache-spark/spark3.0) for more details.
