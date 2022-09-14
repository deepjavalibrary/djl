# Tabular Dataset Support with Tablesaw

This module contains the tabular dataset support with [Tablesaw](https://github.com/jtablesaw/tablesaw).

The following functions have been implemented:

+ a `TablesawDataset` class extending `RandomAccessDataset` to support for importing tabular datasets in Tablesaw format.

## Documentation

You can build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation

You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.tablesaw</groupId>
    <artifactId>tablesaw</artifactId>
    <version>0.19.0-SNAPSHOT</version>
</dependency>
```
