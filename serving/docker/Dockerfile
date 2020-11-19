FROM ubuntu:18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    openjdk-8-jdk-headless

RUN mkdir -p /opt/djl

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY config.properties /opt/djl/
COPY serving /opt/djl/tmp

RUN cd /opt/djl/tmp \
    && ./gradlew copyLibs \
    && cd .. \
    && rm -rf tmp \
    && rm -rf ~/.gradle

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

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

LABEL maintainer="frankfliu2000@gmail.com"
