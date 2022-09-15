# Logging

DJL uses [slf4j-api](http://www.slf4j.org/) to print logs. DJL library itself does not define the
logging framework. Instead, users have to choose their own logging framework at deployment time.
Please refer to [slf4j user manual](http://www.slf4j.org/manual.html) to how to configure your logging framework.

## Adding logging framework to your project

If you didn't add any logging framework to your project, you will see following out:

```shell
SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
```

You can resolve this issue by adding a popular logging framework to your project.

### Use slf4j-simple

For a quick prototyping, you can include [slf4j-simple](https://mvnrepository.com/artifact/org.slf4j/slf4j-simple)
to your project to enable logging (slf4j-simple is not recommended for production deployment):

For Maven:

```
<dependency>
    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-simple</artifactId>
    <version>1.7.30</version>
</dependency>
```

For Gradle:

```
    implementation "org.slf4j:slf4j-simple:1.7.30"
```

Then you can use system properties to configure slf4j-simple log level:

```
-Dorg.slf4j.simpleLogger.defaultLogLevel=debug
```

See [SimpleLogger](https://www.slf4j.org/api/org/slf4j/simple/SimpleLogger.html) for more detail.

### Use log4j2

In our examples module, we use [log4j2 binding](https://github.com/deepjavalibrary/djl/blob/master/examples/build.gradle#L13).
While using log4j2 binding, you also need add a [log4j2.xml](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/resources/log4j2.xml) file.

### Use logback

If you want to use other logging framework such as `logback`, you can just add the following dependency into your build.gradle:

```
    implementation "ch.qos.logback:logback-classic:1.2.3"
```

or for Maven:

```
<dependency>
    <groupId>ch.qos.logback</groupId>
    <artifactId>logback-classic</artifactId>
    <version>1.2.3</version>
    <scope>test</scope>
</dependency>
```

## Configure logging level

`log4j2` allows you to customize logging level using system properties. See our examples [log4j2.xml](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/resources/log4j2.xml#L13).
With this configuration, you can easily change your logging level using java command line options:

```shell
cd examples
./gradlew run -Dai.djl.logging.level=debug
```

Change logging level to debug is particularly useful to troubleshooting following issues:

- Debugging Model loading issue

    When using [ModelZoo](../model-zoo.md) API, sometime run into `ModelNotFoundException`, and wondering
    why the desired model doesn't match the search criteria. DJL prints out model loading details at `debug`
    level to help you nail down the issue.

- Debugging Engine loading issue

    DJL support multiple engines, engine initialization is highly depends on system environment.
    You might run into various problems. Using debug level logging can help you identify configuration issues.

## Common logging issues

Most of the logging issues are not DJL specific. Please refer to SLF4J official document.
 
### log4j version clash
log4j has 1.2 and 2.0 two major versions; and they are not compatible. Both of them might be indirectly included
in your project and may cause class loading issue. Please make sure only one log4j version is included.

### missing logging configuration file
Many logging frameworks require a configuration file. Please make sure add your configuration file to the project.
