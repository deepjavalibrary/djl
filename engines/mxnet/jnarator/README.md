# DJL - JNA code generator

## Overview

This module is used by the Deep Java Library (DJL) project to generate Java Native Access (JNA) code. This code is used to call Native
C_API.

Most C API calls require developers to write code to convert from the Java data type to the C data type.
The goal of this module is to generate the mapping code based on the C header file. 

This module's output is a .jar file that can be executed from the command line.
It takes a list of C header files and, optionally, a data type mapping file.

The jnarator module uses antlr to create a customized C language parser. This parser
reads the C header file, and parses it into an AST tree. The module then
walks through the tree to find C API calls and generates their corresponding Java methods.

The following example demonstrates how to use this module in the Apache MXNet module:

```groovy
task jnarator(dependsOn: ":jnarator:jar") {
    doLast {
        File jnaGenerator = project(":jnarator").jar.outputs.files.singleFile
        javaexec {
            main = "-jar"
            args = [
                    jnaGenerator.absolutePath,
                    "-l",
                    "mxnet",
                    "-p",
                    "org.apache.mxnet.jna",
                    "-o",
                    "${project.buildDir}/generated-src",
                    "-m",
                    "${project.projectDir}/src/main/jna/mapping.properties",
                    "-f",
                    "src/main/include/mxnet/c_api.h",
                    "src/main/include/nnvm/c_api.h"
            ]
        }
    }
}

```
