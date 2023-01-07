# -*- mode: dockerfile -*-
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
FROM 895885662937.dkr.ecr.us-west-2.amazonaws.com/spark/emr-6.9.0:latest
LABEL maintainer="djl-dev@amazon.com"

# Add DJL jars
USER root
ARG DJL_VERSION=0.20.0
ARG JNA_VERSION=5.12.1
ADD https://repo1.maven.org/maven2/ai/djl/api/${DJL_VERSION}/api-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-api-${DJL_VERSION}.jar
ADD https://repo1.maven.org/maven2/ai/djl/spark/spark/${DJL_VERSION}/spark-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-spark-${DJL_VERSION}.jar
ADD https://repo1.maven.org/maven2/ai/djl/pytorch/pytorch-engine/${DJL_VERSION}/pytorch-engine-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-pytorch-engine-${DJL_VERSION}.jar
ADD https://repo1.maven.org/maven2/ai/djl/mxnet/mxnet-engine/${DJL_VERSION}/mxnet-engine-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-mxnet-engine-${DJL_VERSION}.jar
ADD https://repo1.maven.org/maven2/ai/djl/onnxruntime/onnxruntime-engine/${DJL_VERSION}/onnxruntime-engine-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-onnxruntime-engine-${DJL_VERSION}.jar
ADD https://repo1.maven.org/maven2/ai/djl/tensorflow/tensorflow-engine/${DJL_VERSION}/tensorflow-engine-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-tensorflow-engine-${DJL_VERSION}.jar
ADD https://repo1.maven.org/maven2/net/java/dev/jna/jna/${JNA_VERSION}/jna-${JNA_VERSION}.jar /usr/lib/spark/jars/
RUN chmod -R +r /usr/lib/spark/jars/

# Set environment
ENV PYTORCH_PRECXX11 true

USER hadoop:hadoop
