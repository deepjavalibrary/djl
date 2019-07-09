Setup
=====

## Install jdk

Joule example requires JDK 8 (or later), and we recommend use JDK8 since there are some known glitches with JDK 11+.

You must make sure java is on available in $PATH environment variable. If you have multiple java installed,
you can use $JAVA_HOME environment variable to control which java to use.

For ubuntu:
```bash
sudo apt-get install openjdk-8-jre-headless
```

For centos
```bash
sudo yum install java-1.8.0-openjdk
```

For Mac:
```bash
brew tap caskroom/versions
brew update
brew cask install adoptopenjdk8
```

You can also download and install [Oracle JDK](https://www.oracle.com/technetwork/java/javase/overview/index.html)
manually if you have trouble with above commands.

## Install IntelliJ (Optional)

You can use any IDE at your choice. We recommend you to use IntelliJ since we are using IntelliJ as examples in our documents.

## Gradle/Maven (Not required)

We are using gradle and maven wrapper to build our project, you don't need to install gradle or maven.
However, we assume you have basic knowledge about gradle or maven build system.
