import java.net.URL
import java.net.URLEncoder

plugins {
    ai.djl.javaProject
    ai.djl.cppFormatter
    ai.djl.publish
    signing
}

group = "ai.djl.pytorch"

var VERSION = libs.versions.pytorch.get()
if (project.hasProperty("pt_version") && project.property("pt_version") != "")
    VERSION = project.property("pt_version").toString()

val isRelease = project.hasProperty("release") || project.hasProperty("staging")
val isPrecxx11 = project.hasProperty("precxx11")
val isAarch64 = project.hasProperty("aarch64") || arch == "aarch64"

val FLAVOR = when {
    project.hasProperty("cuda") -> project.property("cuda").toString()
    else -> "cpu"
}

val BINARY_ROOT = buildDirectory / "download"

version = VERSION + if (isRelease) "" else "-SNAPSHOT"

fun downloadBuild(ver: String, os: String, flavor: String, isPrecxx11: Boolean = false, isAarch64: Boolean = false) {
    val arch = if (isAarch64) "aarch64" else "x86_64"
    exec {
        if (os == "win")
            commandLine(project.projectDir / "build.cmd", ver, flavor)
        else
            commandLine("bash", "build.sh", ver, flavor, if (isPrecxx11) "precxx11" else "cxx11", arch)
    }

    // for nightly ci
    // the reason why we duplicate the folder here is to insert djl_version into the path
    // so different versions of JNI wouldn't override each other. We don't also want publishDir
    // to have djl_version as engine would require to know that during the System.load()
    val classifier = "$os-$arch"
    val maybePrecxx11 = if (isPrecxx11) "-precxx11" else ""
    val ciDir = project.projectDir / "jnilib/${libs.versions.djl.get()}/$classifier/$flavor$maybePrecxx11"
    copy {
        val tree = fileTree(buildDirectory)
        tree.include("**/libdjl_torch.*", "**/djl_torch.dll")
        from(tree.files)
        into(ciDir)
    }
}

fun downloadBuildAndroid(ver: String) {
    for (abi in listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")) {
        exec {
            commandLine("bash", "build_android.sh", ver, abi)
        }
        val ciDir = project.projectDir / "jnilib/${libs.versions.djl.get()}/android/$abi"
        copy {
            from(buildDirectory / "libdjl_torch.so")
            into(ciDir)
        }
        delete("$buildDirectory/")
    }
}

fun prepareNativeLib(binaryRoot: String, ver: String) {
    if ("mac" !in os)
        throw GradleException("This command must be run from osx")

    val officialPytorchUrl = "https://download.pytorch.org/libtorch"
    val aarch64PytorchUrl = "https://djl-ai.s3.amazonaws.com/publish/pytorch"
    val cuda = "cu121"
    // @formatter:off
    val files = mapOf("cpu/libtorch-cxx11-abi-shared-with-deps-$ver%2Bcpu.zip"     to "cpu/linux-x86_64",
                      "cpu/libtorch-macos-$ver.zip"                                to "cpu/osx-x86_64",
                      "cpu/libtorch-macos-arm64-$ver.zip"                          to "cpu/osx-aarch64",
                      "cpu/libtorch-win-shared-with-deps-$ver%2Bcpu.zip"           to "cpu/win-x86_64",
                      "$cuda/libtorch-cxx11-abi-shared-with-deps-$ver%2B$cuda.zip" to "$cuda/linux-x86_64",
                      "$cuda/libtorch-win-shared-with-deps-$ver%2B$cuda.zip"       to "$cuda/win-x86_64",
                      "cpu/libtorch-shared-with-deps-$ver%2Bcpu.zip"               to "cpu-precxx11/linux-x86_64",
                      "$cuda/libtorch-shared-with-deps-$ver%2B$cuda.zip"           to "$cuda-precxx11/linux-x86_64")

    val aarch64Files = mapOf("$ver/libtorch-shared-with-deps-$ver-aarch64.zip" to "cpu-precxx11/linux-aarch64")
    // @formatter:on
    copyNativeLibToOutputDir(files, binaryRoot, officialPytorchUrl)
    copyNativeLibToOutputDir(aarch64Files, binaryRoot, aarch64PytorchUrl)

    exec {
        commandLine("install_name_tool", "-add_rpath", "@loader_path", "$binaryRoot/cpu/osx-x86_64/native/lib/libtorch_cpu.dylib")
    }
    exec {
        commandLine("install_name_tool", "-add_rpath", "@loader_path", "$binaryRoot/cpu/osx-x86_64/native/lib/libtorch.dylib")
    }
    exec {
        commandLine("install_name_tool", "-add_rpath", "@loader_path", "$binaryRoot/cpu/osx-aarch64/native/lib/libtorch_cpu.dylib")
    }
    exec {
        commandLine("install_name_tool", "-add_rpath", "@loader_path", "$binaryRoot/cpu/osx-aarch64/native/lib/libtorch.dylib")
    }
}

fun copyNativeLibToOutputDir(fileStoreMap: Map<String, String>, binaryRoot: String, url: String) {
    for ((key, value) in fileStoreMap) {
        project.logger.lifecycle("Downloading $url/$key")
        val outputDir = File("$binaryRoot/$value")
        val file = outputDir / "libtorch.zip"
        file.parentFile.mkdirs()
        "$url/$key".url into file
        copy {
            from(zipTree(file))
            into(outputDir)
        }
        delete("$outputDir/libtorch/lib/*.lib")
        delete("$outputDir/libtorch/lib/*.a")

        copy {
            from("$outputDir/libtorch/lib/") {
                include("libarm_compute*", "libc10_cuda.so", "libc10.*", "libcaffe2_nvrtc.so", "libcu*", "libgfortran-*", "libgomp*", "libiomp*", "libnv*", "libopenblasp-*", "libtorch_cpu.*", "libtorch_cuda*.so", "libtorch.*", "asmjit.dll", "c10_cuda.dll", "c10.dll", "caffe2_nvrtc.dll", "cu*.dll", "fbgemm.dll", "nv*.dll", "torch_cpu.dll", "torch_cuda*.dll", "torch.dll", "uv.dll", "zlibwapi.dll")
            }
            into("$outputDir/native/lib")
        }
        if ("-precxx11" in value) {
            val libstd = outputDir / "native/lib/libstdc++.so.6"
            val stdcUrl = when {
                "aarch64" in value -> "https://publish.djl.ai/extra/aarch64/libstdc%2B%2B.so.6"
                else -> "https://publish.djl.ai/extra/libstdc%2B%2B.so.6"
            }
            stdcUrl.url into libstd
        }
        if ("osx-aarch64" in value) {
            val libomp = outputDir / "native/lib/libomp.dylib"
            val ompUrl = "https://publish.djl.ai/extra/macos-arm64/libomp.dylib"
            ompUrl.url into libomp
        }
        delete(file)
        delete(outputDir / "libtorch")
    }
}

tasks {
    register("compileAndroidJNI") {
        doFirst {
            downloadBuildAndroid(VERSION)
        }
    }

    register("cleanJNI") {
        doFirst {
            delete(project.projectDir / "build",
                   project.projectDir / "libtorch",
                   project.projectDir / "libtorch_android",
                   fileTree(project.projectDir) { include("**.zip") })
        }
    }

    register("compileJNI") {
        doFirst {
            // You have to use an environment with CUDA persets for Linux and Windows
            when {
                "windows" in os -> downloadBuild(VERSION, "win", FLAVOR)
                "mac" in os -> downloadBuild(VERSION, "osx", FLAVOR, false, isAarch64)
                "linux" in os -> downloadBuild(VERSION, "linux", FLAVOR, isPrecxx11, isAarch64)
                else -> throw IllegalStateException("Unknown Architecture $osName-$FLAVOR")
            }

            delete(fileTree("$home/.djl.ai/pytorch/") {
                include("**/*djl_torch.*")
            })
        }
    }

    // Create a placeholder jar without classifier to pass sonatype tests but throws an Exception if loaded
    jar {
        val placeholder = buildDirectory / "placeholder"
        // this line is to enforce gradle to build the jar
        // otherwise it doesn't generate the placeholder jar at times
        // when there is no java code inside src/main
        outputs.dir("build/libs")
        doFirst {
            val dir = placeholder / "native/lib"
            dir.mkdirs()
            val propFile = placeholder / "native/lib/pytorch.properties"
            propFile.text = "placeholder=true\nversion=${project.version}\n"
        }

        from(placeholder)
    }

    withType<GenerateModuleMetadata> { enabled = false }

    signing {
        isRequired = project.hasProperty("staging") || project.hasProperty("snapshot")
        val signingKey = findProperty("signingKey").toString()
        val signingPassword = findProperty("signingPassword").toString()
        useInMemoryPgpKeys(signingKey, signingPassword)
        sign(publishing.publications["maven"])
    }

    register("downloadPyTorchNativeLib") {
        doLast {
            prepareNativeLib("$BINARY_ROOT", VERSION)
        }
    }

    register("uploadS3") {
        doLast {
            delete("$BINARY_ROOT")
            prepareNativeLib("$BINARY_ROOT", VERSION)

            exec {
                commandLine("sh", "-c", "find $BINARY_ROOT -type f | xargs gzip")
            }

            (BINARY_ROOT / "files.txt").text = buildString {
                val uploadDirs = listOf(BINARY_ROOT / "cpu/linux-x86_64/native/lib/",
                                        BINARY_ROOT / "cpu/osx-aarch64/native/lib/",
                                        BINARY_ROOT / "cpu/osx-x86_64/native/lib/",
                                        BINARY_ROOT / "cpu/win-x86_64/native/lib/",
                                        BINARY_ROOT / "cpu-precxx11/linux-aarch64/native/lib/",
                                        BINARY_ROOT / "cpu-precxx11/linux-x86_64/native/lib/",
                                        BINARY_ROOT / "cu121/linux-x86_64/native/lib/",
                                        BINARY_ROOT / "cu121/win-x86_64/native/lib/",
                                        BINARY_ROOT / "cu121-precxx11/linux-x86_64/native/lib/")
                for (item in uploadDirs)
                    fileTree(item).files.map { it.name }.forEach {
                        val out = item.relativeTo(File("${BINARY_ROOT}/")).absolutePath
                        appendLine(out + URLEncoder.encode(it, "UTF-8"))
                    }
            }
            exec {
                commandLine("aws", "s3", "sync", "$BINARY_ROOT", "s3://djl-ai/publish/pytorch/$VERSION/")
            }
        }
    }

}

java {
    withJavadocJar()
    withSourcesJar()
}

formatCpp {
    exclusions = listOf("main/patch/**")
}

publishing.repositories {
    maven {
        if (project.hasProperty("snapshot")) {
            name = "snapshot"
            url = uri("https://oss.sonatype.org/content/repositories/snapshots/")
            credentials {
                username = findProperty("ossrhUsername").toString()
                password = findProperty("ossrhPassword").toString()
            }
        } else if (project.hasProperty("staging")) {
            name = "staging"
            url = uri("https://oss.sonatype.org/service/local/staging/deploy/maven2/")
            credentials {
                username = findProperty("ossrhUsername").toString()
                password = findProperty("ossrhPassword").toString()
            }
        } else {
            name = "local"
            url = uri("build/repo")
        }
    }
}

val flavorNames = file(BINARY_ROOT).list() ?: emptyArray()
for (flavor in flavorNames) {

    val platformNames = BINARY_ROOT.resolve(flavor).list() ?: emptyArray()

    val artifactsNames = ArrayList<Task>()

    for (osName in platformNames) {
        tasks.create<Jar>("${flavor}-${osName}Jar") {
            doFirst {
                val propFile = BINARY_ROOT / "pytorch.properties"
                propFile.delete()
                val dsStore = BINARY_ROOT / flavor / osName / "native/lib/.DS_Store"
                dsStore.delete()

                val versionName = "${project.version}-$nowFormatted"
                val dir = BINARY_ROOT / flavor / osName / "native/lib"
                propFile.text = buildString {
                    append("version=$versionName\nflavor=$flavor\nclassifier=$osName\nlibraries=")
                    var first = true
                    for (name in dir.list()!!.sorted()) {
                        if (first)
                            first = false
                        else
                            append(',')
                        append(name)
                    }
                }
                val metaInf = BINARY_ROOT / flavor / osName / "META-INF"
                metaInf.mkdirs()
                val licenseFile = metaInf / "LICENSE"
                licenseFile.text = "https://raw.githubusercontent.com/pytorch/pytorch/master/LICENSE".url.text

                val binaryLicenseFile = metaInf / "NOTICE"
                binaryLicenseFile.text = "https://raw.githubusercontent.com/pytorch/pytorch/master/NOTICE".url.text

                if ("-precxx11" in flavor) {
                    val libstd = metaInf / "ATTRIBUTION"
                    libstd.text = "https://publish.djl.ai/extra/THIRD-PARTY-LICENSES_qHnMKgbdWa.txt".url.text
                }
            }
            from(BINARY_ROOT / flavor / osName / "/native/lib") {
                into("pytorch/$flavor/$osName")
            }
            from(BINARY_ROOT / "pytorch.properties") {
                into("native/lib")
            }
            from("src/main/resources")
            archiveClassifier = osName
            archiveBaseName = "pytorch-native-$flavor"

            manifest {
                attributes("Automatic-Module-Name" to "ai.djl.pytorch_native_${flavor}_$osName")
            }
        }
        artifactsNames += tasks["$flavor-${osName}Jar"]
    }

    // Only publish if the project directory equals the current directory
    // This means that publishing from the main project does not publish the native jars
    // and the native jars have to be published separately
    if (project.projectDir.toString() == System.getProperty("user.dir")) {
        publishing.publications.create<MavenPublication>(flavor) {
            artifactId = "pytorch-native-$flavor"
            from(components["java"])
            setArtifacts(artifactsNames)
            artifact(tasks.jar)
            artifact(tasks.javadocJar)
            artifact(tasks.sourcesJar)
            pom {
                name = "DJL release for PyTorch native binaries"
                description = "Deep Java Library (DJL) provided PyTorch native library binary distribution"
                url = "http://www.djl.ai/engines/pytorch/pytorch-native"
                packaging = "jar"

                licenses {
                    license {
                        name = "The Apache License, Version 2.0"
                        url = "https://www.apache.org/licenses/LICENSE-2.0"
                    }
                }

                scm {
                    connection = "scm:git:git@github.com:deepjavalibrary/djl.git"
                    developerConnection = "scm:git:git@github.com:deepjavalibrary/djl.git"
                    url = "https://github.com/deepjavalibrary/djl"
                    tag = "HEAD"
                }

                developers {
                    developer {
                        name = "DJL.AI Team"
                        email = "djl-dev@amazon.com"
                        organization = "Amazon AI"
                        organizationUrl = "https://amazon.com"
                    }
                }
            }
        }
    }
}

tasks {
    // Gradle 8.0 requires explicitly dependency
    withType<PublishToMavenRepository> {
        for (flavor in flavorNames) {
            dependsOn("sign${flavor.substring(0, 1).uppercase() + flavor.substring(1)}Publication")

            val platformNames = (BINARY_ROOT / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }
    withType<Sign> {
        for (flavor in flavorNames) {
            val platformNames = (BINARY_ROOT / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }

    clean {
        doFirst {
            delete(project.projectDir / "jnilib")
            delete(project.projectDir.parentFile / "pytorch-jni/jnilib")
        }
    }
}