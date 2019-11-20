# Setup development environment

## Install jdk

The Deep Java Library(DJL)  project requires JDK 8 (or later). We recommend using JDK8 since there are some known glitches with JDK 11+.

Verify that Java is available in your $PATH environment variable using the following commands. If you have multiple versions of Java installed,
you can use the $JAVA_HOME environment variable to control which version of Java to use.

For ubuntu:
```bash
sudo apt-get install openjdk-8-jdk-headless
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
manually if you have trouble with the previous commands.

## Install IntelliJ (Optional)

You can use the IDE of your choice. We recommend using IntelliJ.

### Import the DJL project into IntelliJ

1. Open IntelliJ and click `Import Project`.
2. Navigate to the DJL project root folder and click "Open".
3. Choose `Import project from existing model` and select `Gradle`.
4. Select the default configuration and click `OK`.

## Import using Gradle/Maven wrappers(Optional)

You use gradle and maven wrappers to build the project, so you don't need to install gradle or maven.
However, you should have basic knowledge about the gradle or maven build system.
