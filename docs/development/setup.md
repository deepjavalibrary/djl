# Setup development environment

## Install jdk

Our project requires JDK 8 (or later). We recommend use JDK8 since there are some known glitches with JDK 11+.

Verify Java is available in your $PATH environment variable. If you have multiple versions of Java installed,
you can use $JAVA_HOME environment variable to control which version of Java to use.

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

### Import DJL project into IntelliJ

1. Open IntelliJ and click `Import Project`.
2. Navigate to the project root folder and click "Open".
3. Choose `Import project from existing model`, you can select `Gradle`
4. Use the default configuration and click `OK`.

## Gradle/Maven (Not required)

You use gradle and maven wrappers to build the project, so you don't need to install gradle or maven.
However, you should have basic knowledge about the gradle or maven build system.
