FROM ubuntu:18.04

RUN apt-get update || true
RUN apt-get install -y openjdk-11-jdk-headless
RUN apt-get install -y maven
COPY pom.xml .
RUN mvn dependency:resolve
RUN apt-get install -y python3-pip git
RUN pip3 install jupyter
RUN git clone https://github.com/SpencerPark/IJava.git
RUN cd IJava/ && chmod u+x gradlew && ./gradlew installKernel
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
