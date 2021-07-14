FROM ubuntu:18.04
ARG djl_version=0.12.0

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    openjdk-8-jdk-headless \
    curl

RUN curl -O https://publish.djl.ai/djl-serving/djl-serving_${djl_version}-1_all.deb && \
    dpkg -i djl-serving_${djl_version}-1_all.deb && \
    rm djl-serving_${djl_version}-1_all.deb

RUN mkdir -p /opt/djl
COPY config.properties /opt/djl/conf/

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

EXPOSE 8080

WORKDIR /opt/djl
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV MXNET_ENGINE_TYPE=NaiveEngine
ENV JAVA_OPTS="-Dai.djl.pytorch.num_interop_threads=1"
ENV MODEL_SERVER_HOME=/opt/djl

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="frankfliu2000@gmail.com"
