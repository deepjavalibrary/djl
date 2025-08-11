# DJL external dependencies

This document contains external libraries that DJL depends on and their versions.

| packages              | version     | Release notes                                                                    | maven                                                                                                                     |
|-----------------------|-------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| PyTorch               | 1.13.1[^1]  | [2.0.1](https://github.com/pytorch/pytorch/releases)                             |                                                                                                                           |
| Tensorflow/java       | 2.10.1[^2]  | [2.10.1](https://github.com/tensorflow/java/releases)                            | [org.tensorflow:tensorflow-core-api](https://mvnrepository.com/artifact/org.tensorflow/tensorflow-core-api)               |
| Apache MXNet          | 1.9.1       | [1.9.1](https://github.com/apache/mxnet/releases)                                |                                                                                                                           |
| TensorFlow/lite       | 2.6.2[^3]   | [2.13.0](https://github.com/tensorflow/tensorflow/releases)                      |                                                                                                                           |
| OnnxRuntime           | 1.15.0      | [1.15.0](https://github.com/microsoft/onnxruntime/releases)                      | [com.microsoft.onnxruntime:onnxruntime](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime)         |
| sentencepiece         | 0.1.97[^6]  | [0.1.99](https://github.com/google/sentencepiece/releases)                       |                                                                                                                           |
| HuggingFace tokenizer | 0.13.2[^7]  | [0.13.3](https://github.com/huggingface/tokenizers/releases)                     |                                                                                                                           |
| fasttext              | 0.9.2       | [0.9.2](https://github.com/facebookresearch/fastText/releases)                   |                                                                                                                           |
| XGBoost               | 1.7.5       | [1.7.5](https://github.com/dmlc/xgboost/releases)                                | [ml.dmlc:xgboost4j_2.12](https://mvnrepository.com/artifact/ml.dmlc/xgboost4j)                                            |
| LightGBM              | 3.2.110[^8] | [3.3.5](https://github.com/microsoft/LightGBM/releases)                          | [com.microsoft.ml.lightgbm:lightgbmlib](https://mvnrepository.com/artifact/com.microsoft.ml.lightgbm/lightgbmlib)         |
| rapis                 | 22.12.0[^9] | [23.06.00](https://github.com/rapidsai/cudf/releases)                            | [ai.rapids:cudf::cuda11](https://mvnrepository.com/artifact/ai.rapids/cudf)                                               |
| commons-cli           | 1.5.0       | [1.5.0](https://commons.apache.org/proper/commons-cli/changes-report.html)       | [commons-cli:commons-cli](https://mvnrepository.com/artifact/commons-cli/commons-cli)                                     |
| commons-compress      | 1.23.0      | [1.23.0](https://commons.apache.org/proper/commons-compress/changes-report.html) | [org.apache.commons:commons-compress](https://mvnrepository.com/artifact/org.apache.commons/commons-compress)             |
| commons-csv           | 1.10.0      | [1.10.0](https://commons.apache.org/proper/commons-csv/changes-report.html)      | [org.apache.commons:commons-csv](https://mvnrepository.com/artifact/org.apache.commons/commons-csv)                       |
| commons-logging       | 1.2         | [1.2](https://commons.apache.org/proper/commons-logging/)                        | [commons-logging:commons-logging](https://mvnrepository.com/artifact/commons-logging/commons-logging)                     |
| gson                  | 2.10.1      | [2.10.1](https://github.com/google/gson/releases)                                | [com.google.code.gson:gson](https://mvnrepository.com/artifact/com.google.code.gson/gson)                                 |
| JNA                   | 5.13.0      | [5.13.0](https://github.com/java-native-access/jna/blob/master/CHANGES.md)       | [net.java.dev.jna:jna](https://mvnrepository.com/artifact/net.java.dev.jna/jna)                                           |
| slf4j                 | 1.7.36      | [1.7.36](https://mvnrepository.com/artifact/org.slf4j/slf4j-api)                 | [org.slf4j:slf4j-api](https://mvnrepository.com/artifact/org.slf4j/slf4j-api)                                             |
| log4j-slf4j           | 2.20.0      | [2.20.0](https://logging.apache.org/log4j/2.x/release-notes/index.html)          | [org.apache.logging.log4j:log4j-slf4j-impl](https://mvnrepository.com/artifact/org.apache.logging.log4j/log4j-slf4j-impl) |
| awssdk                | 2.20.85     | [2.20.85](https://github.com/aws/aws-sdk-java-v2/tags)                           | [software.amazon.awssdk:bom](https://mvnrepository.com/artifact/software.amazon.awssdk/bom)                               |
| hadoop                | 3.3.5       | [3.3.5](https://hadoop.apache.org/release.html)                                  | [org.apache.hadoop:hadoop-client](https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-client)                     |
| javacpp               | 1.5.9       | [1.5.9](https://github.com/bytedeco/javacpp/releases)                            | [org.bytedeco:javacpp](https://mvnrepository.com/artifact/org.bytedeco/javacpp)                                           |
| javacv                | 1.5.9       | [1.5.9](https://github.com/bytedeco/javacv/releases)                             | [org.bytedeco:javacv](https://mvnrepository.com/artifact/org.bytedeco/javacv)                                             |
| ffmpeg                | 6.0-1.5.9   | [6.0-1.5.9](https://github.com/bytedeco/javacv/releases)                         | [org.bytedeco:ffmpeg](https://mvnrepository.com/artifact/org.bytedeco/ffmpeg)                                             |
| protobuf              | 3.23.3      | [3.23.3](https://mvnrepository.com/artifact/com.google.protobuf/protobuf-java)   | [com.google.protobuf:protobuf-java](https://mvnrepository.com/artifact/com.google.protobuf/protobuf-java)                 |
| tablesaw              | 0.43.1      | [0.43.1](https://github.com/jtablesaw/tablesaw/releases)                         | [tech.tablesaw:tablesaw-core](https://mvnrepository.com/artifact/tech.tablesaw/tablesaw-core/0.43.1)                      |
| spark                 | 3.4.0[^10]  | [3.4.0](https://github.com/apache/spark/tags)                                    | [org.apache.spark:spark-core_2.12](https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.12)                   |
| openpnp-opencv        | 4.7.0-0     | [4.7.0-0](https://github.com/openpnp/opencv/releases)                            | [org.openpnp:opencv](https://mvnrepository.com/artifact/org.openpnp/opencv)                                               |
| antlr4                | 4.11.1[^11] | [4.13.0](https://github.com/antlr/antlr4/releases)                               | [org.antlr:antlr4-runtime](https://mvnrepository.com/artifact/org.antlr/antlr4-runtime)                                   |
| testng                | 7.8.0       | [7.8.0](https://github.com/testng-team/testng/releases)                          | [org.testng:testng](https://mvnrepository.com/artifact/org.testng/testng)                                                 |
| junit                 | 4.13.2      | [4.13.2](https://junit.org/junit4/)                                              | [junit:junit](https://mvnrepository.com/artifact/junit/junit)                                                             |
| mockito               | 5.2.0       | [5.3.1](https://github.com/mockito/mockito/releases)                             | [org.mockito:mockito-core](https://mvnrepository.com/artifact/org.mockito/mockito-core)                                   |


[^1]: PyTorch 2.0.1 will crash on GPU when running multithreading inference. 
[^2]: TensorFlow/Java only support 2.10.1 
[^3]: No plan to upgrade
[^4]: Deferred upgrade
[^5]: No plan to upgrade
[^6]: Deferred upgrade
[^7]: Deferred upgrade
[^8]: LightGBM 3.3.5 doesn't support centos 7
[^9]: Deferred upgrade
[^10]: Scala 2.12 only
[^11]: Antlr 4.13.0 is not compatible with existing code.


