djl.ai - JNA generator
======================

## Overview

This module is a tool that used by djl.ai to generate JNA code that used to call Native
C_API.

Most of C API call requires developer write code to convert java data type to C data type.
The goal of this module is to generate the mapping code based in C header file. 

This module's output is a jar file that can be executed from command line.
It takes a list of C header files, and optional a data type mapping file.

jnarator module uses antlr to create a customized C language parser, the parser
read the C header file, and parse it into a AST tree. Then the code in this module 
walk through the tree find C API calls and generate corresponding Java methods.

Here is how we use this module in mxnet module:

```
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
