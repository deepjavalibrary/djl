# DJL - Bill of Materials (BOM)

BOM stands for Bill Of Materials. Maven lets us define the versions of the dependencies or transitive
dependencies in a separate POM. A BOM package is a POM only jar file that is used to control the versions
of a project’s dependencies and provide a central place to define and update those versions. 
See [Maven document](https://maven.apache.org/guides/introduction/introduction-to-dependency-mechanism.html#dependency-management)
about how BOM works.

DJL's BOM package provides a flexibility way for developers to add DJL dependencies to their project
without worrying about each modules' version that we should depend on.

## How to use DJL's BOM

### Use BOM in Maven

- First you need add BOM into your pom.xml file in <dependencyManagement> section (notice that you
will need to mention the type as pom and the scope as import) as the following:

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>ai.djl</groupId>
            <artifactId>bom</artifactId>
            <version>0.12.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

- Then you import the desired DJL modules into to you pom.xml file (no version is needed): 

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>ai.djl</groupId>
            <artifactId>bom</artifactId>
            <version>0.12.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
<dependencies>
    <dependency>
        <groupId>ai.djl</groupId>
        <artifactId>api</artifactId>
    </dependency>
    <dependency>
        <groupId>ai.djl.mxnet</groupId>
        <artifactId>mxnet-engine</artifactId>
    </dependency>
    <dependency>
        <groupId>ai.djl.mxnet</groupId>
        <artifactId>mxnet-engine</artifactId>
    </dependency>
    <dependency>
        <groupId>ai.djl.mxnet</groupId>
        <artifactId>mxnet-native-auto</artifactId>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

### Use BOM in Gradle

- First you need add BOM into your build.gradle file as the following:

```
    implementation platform("ai.djl:bom:0.12.0")
```

- Then you import the desired DJL modules into to you pom.xml file (no version is needed):

```
    implementation "ai.djl.pytorch:pytorch-model-zoo" // No version required
    implementation "ai.djl.pytorch:pytorch-native-auto"  // No version required
```
