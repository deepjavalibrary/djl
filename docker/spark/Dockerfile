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
FROM 314815235551.dkr.ecr.us-east-2.amazonaws.com/sagemaker-spark-processing:3.5-cpu
LABEL maintainer="djl-dev@amazon.com"

# Install dependencies
ARG DJL_VERSION=0.36.0
ARG JNA_VERSION=5.14.0
ARG JAVACV_VERSION=1.5.10
ARG JAVACPP_VERSION=1.5.10
ARG FFMPEG_VERSION=6.1.1-1.5.10
ARG TENSORFLOW_CORE_VERSION=1.0.0-rc.1
ARG PROTOBUF_VERSION=3.25.3
ARG PYTORCH_VERSION=2.3.1

RUN pip3 install --no-cache-dir torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir pillow pandas numpy pyarrow librosa 'transformers[torch]'

COPY extensions/spark/setup/dist/ dist/
RUN pip3 install --no-cache-dir dist/djl_spark-*-py3-none-any.whl && \
    rm -rf dist

ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/api/${DJL_VERSION}/api-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-api-${DJL_VERSION}.jar
ADD --chmod=644 https://repo1.maven.org/maven2/ai/djl/spark/spark_2.12/${DJL_VERSION}/spark_2.12-${DJL_VERSION}.jar /usr/lib/spark/jars/djl-spark_2.12-${DJL_VERSION}.jar
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
ADD --chmod=644 https://repo1.maven.org/maven2/com/google/protobuf/protobuf-java/${PROTOBUF_VERSION}/protobuf-java-${PROTOBUF_VERSION}.jar /usr/lib/spark/jars/

# Set environment
ENV PYTORCH_PRECXX11 true
ENV OMP_NUM_THREADS 1
ENV DJL_CACHE_DIR /tmp/.djl.ai
ENV HUGGINGFACE_HUB_CACHE /tmp
ENV TRANSFORMERS_CACHE /tmp

RUN echo 'export SPARK_JAVA_OPTS="$SPARK_JAVA_OPTS -Dai.djl.pytorch.graph_optimizer=false"' >> /opt/hadoop-config/spark-env.sh
RUN echo "export PYTORCH_PRECXX11=true" >> /opt/hadoop-config/spark-env.sh
RUN echo "export OMP_NUM_THREADS=1" >> /opt/hadoop-config/spark-env.sh
RUN echo "export DJL_CACHE_DIR=/tmp/.djl.ai" >> /opt/hadoop-config/spark-env.sh
RUN echo "export HUGGINGFACE_HUB_CACHE=/tmp" >> /opt/hadoop-config/spark-env.sh
RUN echo "export TRANSFORMERS_CACHE=/tmp" >> /opt/hadoop-config/spark-env.sh
RUN echo "spark.yarn.appMasterEnv.PYTORCH_PRECXX11 true" >> /opt/hadoop-config/spark-defaults.conf
RUN echo "spark.executorEnv.PYTORCH_PRECXX11 true" >> /opt/hadoop-config/spark-defaults.conf
RUN echo "spark.sql.execution.arrow.maxRecordsPerBatch 500" >> /opt/hadoop-config/spark-defaults.conf
RUN echo "spark.hadoop.fs.s3a.connection.maximum 1000" >> /opt/hadoop-config/spark-defaults.conf
