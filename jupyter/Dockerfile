FROM ubuntu:18.04

ARG NB_USER="sagemaker-user"
ARG NB_UID="1000"
ARG NB_GID="100"

RUN apt-get update || true && \
    apt-get install -y sudo && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chmod g+w /etc/passwd && \
    echo "${NB_USER}    ALL=(ALL)    NOPASSWD:    ALL" >> /etc/sudoers && \
    # Prevent apt-get cache from being persisted to this layer.
    rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

RUN apt-get install -y \
    openjdk-11-jdk-headless \
    python3-pip git curl unzip

RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install jupyter ipykernel awscli

# Install Java kernel
RUN git clone https://github.com/frankfliu/IJava.git
RUN cd IJava/ && ./gradlew zK && \
    unzip -q build/distributions/ijava-1.3.0.zip -d ijava-kernel && \
    cd ijava-kernel && \
    python install.py --sys-prefix && \
    cd ../.. && rm -rf IJava/ && rm -rf ~/.gradle

RUN jupyter kernelspec list

ENV SHELL=/bin/bash
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
USER $NB_UID

WORKDIR /home/jupyter

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''",  "--NotebookApp.password=''"]
