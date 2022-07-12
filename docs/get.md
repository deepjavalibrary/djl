# Getting DJL

## Maven Central

There are several options you can take to get DJL for use in your own project. The most common is to access our builds from [Maven Central](https://mvnrepository.com/artifact/ai.djl). The dependencies are usually added to your project in the Gradle `build.gradle` file or the Maven `pom.xml` file.

Most of our documentation including the module documentation provides explanations for how to get the specific module from Maven Central, but you can also find many examples of this in our [Demo Repository](https://github.com/deepjavalibrary/djl-demo).

## Nightly Snapshots
  
If you are looking for the latest features from the DJL development team, using the releases will mean that you have to wait for the next release to get them. Instead, you could also use the DJL nightly snapshots.

The nightly snapshots, like the releases, can also be added through a build system like Maven or Gradle. The first thing you will need to do is add the snapshot repository `https://oss.sonatype.org/content/repositories/snapshots/` to your build system.

In Gradle, this is done by adding it to the repository block such as in [this example](https://github.com/deepjavalibrary/djl-serving/blob/master/build.gradle#L23):

```groovy
repositories {
    maven {
        url 'https://oss.sonatype.org/content/repositories/snapshots/'
    }
}
```

In Maven, you can create a repositories tag such as [here](https://github.com/deepjavalibrary/djl/blob/master/examples/pom.xml#L17)

```xml
<repositories>
    <repository>
        <id>djl.ai</id>
        <url>https://oss.sonatype.org/content/repositories/snapshots/</url>
    </repository>
</repositories>

```

Once you have added the snapshots repository, you are then able to access the snapshots version of DJL. This is the version of the upcoming release of DJL appended with snapshot. So, if DJL has just released version `0.1000.0`, then you would use `0.1001.0-SNAPSHOT` as your DJL version. This version can then be passed to the [DJL BOM](../bom/README.md) and/or the other DJL dependencies.

One thing to note about the snapshot versions is that they are only kept in the snapshot repository for two weeks. This means that when DJL has a new release such as the actual release of `0.1001.0`, you will have to update your project dependencies. You can either migrate to the actual release version of the newly released `0.1001.0` or move to the next release snapshot of `0.1002.0-SNAPSHOT`. If you don't migrate in time, your build system may be unable to find DJL to build your project, but you can still change the version at any time.

If you are interested in how the nightly snapshots are produced, you can see the full definition in the [nightly publish GitHub action](https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/nightly_publish.yml).

## Build from Source

The final option is to build from source.
To build from source, begin by checking out the code.
Once you have checked out the code locally, you can build it as follows using Gradle:

```sh
# for Linux/macOS:
./gradlew build

# for Windows:
gradlew build
```

To increase build speed, you can use the following command to skip unit tests:

```sh
# for Linux/macOS:
./gradlew build -x test

# for Windows:
gradlew build -x test
```

### Importing into eclipse

to import source project into eclipse

```sh
# for Linux/macOS:
./gradlew eclipse


# for Windows:
gradlew eclipse

```

in eclipse 

file->import->gradle->existing gradle project

**Note:** please set your workspace text encoding setting to UTF-8

### Using a release version

If you build following the above instructions, you will use the version of the code on the development master branch. You can also checkout one of the releases by using `git checkout v0.1000.0` to get the release for version `0.1000.0` (and so on for actual release numbers). Then, run the build just like above.

You can look here to find the [list of DJL releases](https://github.com/deepjavalibrary/djl/releases).

### Using built-from-source version in another project

If you have another project and want to use a custom version of DJL in it, you can do the following. First, build DJL from source by running `./gradlew build` inside djl folder. Then run `./gradlew publishToMavenLocal`, which will install DJL to your local maven repository cache, located on your filesystem at `~/.m2/repository`. After publishing it here, you can add the DJL snapshot version dependencies as shown below 

```groovy
dependencies {
    implementation platform("ai.djl:bom:0.19.0-SNAPSHOT")
}
```

This snapshot version is the same as the custom DJL repository. 

You also need to change directory to `djl/bom`. Then build and publish it to maven local same as what was done in `djl`.

From there, you may have to update the Maven or Gradle build of the project importing DJL to also look at the local maven repository cache for your locally published versions of DJL. For Maven, no changes are necessary. If you are using Gradle, you will have to add the maven local repository such as this [example](https://github.com/deepjavalibrary/djl-demo/blob/135c969d66d98d1672852e53a37e52ca1da3e325/pneumonia-detection/build.gradle#L11):

```groovy
repositories {
    mavenLocal()
}
```

Note that `mavenCentral()` may still be needed for applications like log4j and json.