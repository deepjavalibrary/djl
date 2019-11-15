FROM ubuntu:18.04

RUN apt-get update || true
RUN apt-get install -y openjdk-11-jdk-headless
RUN apt-get install -y python3-pip git
RUN pip3 install jupyter
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

RUN git clone https://github.com/SpencerPark/IJava.git
RUN cd IJava/ && chmod u+x gradlew && ./gradlew installKernel && ./gradlew clean && cd
RUN rm -rf .gradle
RUN rm -rf .m2

WORKDIR /home/jupyter

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''",  "--NotebookApp.password=''"]
