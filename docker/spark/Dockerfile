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
FROM 711395599931.dkr.ecr.us-east-2.amazonaws.com/spark/emr-6.9.0:latest
LABEL maintainer="djl-dev@amazon.com"

# Install DJL dependencies
USER root
ARG DJL_VERSION=0.21.0
ARG JNA_VERSION=5.12.1
ARG JAVACV_VERSION=1.5.8
ARG JAVACPP_VERSION=1.5.8
ARG FFMPEG_VERSION=5.1.2-1.5.8
ARG TENSORFLOW_CORE_VERSION=0.4.2
ARG PROTOBUF_VERSION=3.21.9

RUN yum -y install python3-devel gcc
COPY extensions/spark/setup/dist/ dist/
RUN pip3 install --no-cache-dir dist/djl_spark-*-py3-none-any.whl && \
    rm -rf dist
RUN pip3 install --no-cache-dir torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir pillow pandas numpy pyarrow librosa 'transformers[torch]'

ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/api/${DJL_VERSION}/api-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-api-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/spark/spark/${DJL_VERSION}/spark-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-spark-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/huggingface/tokenizers/${DJL_VERSION}/tokenizers-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-tokenizers-${DJL_VERSION}.jar

ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/audio/audio/${DJL_VERSION}/audio-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-audio-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/org/bytedeco/javacv/${JAVACV_VERSION}/javacv-${JAVACV_VERSION}.jar /usr/lib/spark/jars/
ADD --chmod=644 https://repo1.maven.org/maven2/org/bytedeco/javacpp/${JAVACPP_VERSION}/javacpp-${JAVACPP_VERSION}.jar /usr/lib/spark/jars/
ADD --chmod=644 https://repo1.maven.org/maven2/org/bytedeco/ffmpeg/${FFMPEG_VERSION}/ffmpeg-${FFMPEG_VERSION}.jar /usr/lib/spark/jars/
ADD --chmod=644 https://repo1.maven.org/maven2/org/bytedeco/ffmpeg/${FFMPEG_VERSION}/ffmpeg-${FFMPEG_VERSION}-linux-x86_64.jar /usr/lib/spark/jars/
ADD --chmod=644 https://repo1.maven.org/maven2/org/bytedeco/ffmpeg-platform/${FFMPEG_VERSION}/ffmpeg-platform-${FFMPEG_VERSION}.jar /usr/lib/spark/jars/

ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/pytorch/pytorch-engine/${DJL_VERSION}/pytorch-engine-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-pytorch-engine-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/pytorch/pytorch-model-zoo/${DJL_VERSION}/pytorch-model-zoo-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-model-zoo-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/net/java/dev/jna/jna/${JNA_VERSION}/jna-${JNA_VERSION}.jar /usr/lib/spark/jars/

ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/tensorflow/tensorflow-engine/${DJL_VERSION}/tensorflow-engine-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-tensorflow-engine-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/tensorflow/tensorflow-model-zoo/${DJL_VERSION}/tensorflow-model-zoo-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-tensorflow-model-zoo-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/org/tensorflow/tensorflow-core-api/${TENSORFLOW_CORE_VERSION}/tensorflow-core-api-${TENSORFLOW_CORE_VERSION}.jar /usr/lib/spark/jars/
RUN rm /usr/lib/spark/jars/protobuf-java-*.jar
ADD --chmod=644 https://repo1.maven.org/maven2/com/google/protobuf/protobuf-java/${PROTOBUF_VERSION}/protobuf-java-${PROTOBUF_VERSION}.jar /usr/lib/spark/jars/

RUN echo 'export SPARK_JAVA_OPTS="$SPARK_JAVA_OPTS -Dai.djl.pytorch.graph_optimizer=false"' >> /usr/lib/spark/conf/spark-env.sh

# Set environment
ENV PYTORCH_PRECXX11 true
ENV OMP_NUM_THREADS 1

USER hadoop:hadoop
