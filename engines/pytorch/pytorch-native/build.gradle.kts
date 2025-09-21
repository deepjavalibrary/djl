import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.Base64
import kotlin.collections.ArrayList

plugins {
    ai.djl.javaProject
    ai.djl.cppFormatter
    `maven-publish`
    signing
}

group = "ai.djl.pytorch"

var ptVersion = libs.versions.pytorch.get()
if (project.hasProperty("pt_version") && project.property("pt_version") != "") {
    ptVersion = project.property("pt_version").toString()
}
val ptFlavor = when {
    project.hasProperty("cuda") -> project.property("cuda").toString()
    else -> "cpu"
}

val isRelease = project.hasProperty("release") || project.hasProperty("staging")
val isPrecxx11 = project.hasProperty("precxx11")
val isAarch64 = project.hasProperty("aarch64") || arch == "aarch64"

version = ptVersion + if (isRelease) "" else "-SNAPSHOT"

tasks {
    val injected = project.objects.newInstance<InjectedOps>()
    val binaryRoot = buildDirectory / "download"
    val dir = projectDir
    val buildDir = buildDirectory
    val djlVersion = libs.versions.djl.get()
    val ptVersion = ptVersion
    val logger = project.logger

    fun downloadBuild(
        os: String,
        flavor: String,
        isPrecxx11: Boolean = false,
        isAarch64: Boolean = false
    ) {
        val arch = if (isAarch64) "aarch64" else "x86_64"
        injected.exec.exec {
            workingDir = dir
            if (os == "win")
                commandLine(dir / "build.cmd", ptVersion, flavor)
            else
                commandLine("bash", "build.sh", ptVersion, flavor, if (isPrecxx11) "precxx11" else "cxx11", arch)
        }

        // for nightly ci
        // the reason why we duplicate the folder here is to insert djl_version into the path
        // so different versions of JNI wouldn't override each other. We don't also want publishDir
        // to have djl_version as engine would require to know that during the System.load()
        val classifier = "$os-$arch"
        val maybePrecxx11 = if (isPrecxx11) "-precxx11" else ""
        val ciDir = dir / "jnilib/${djlVersion}/$classifier/$flavor$maybePrecxx11"
        injected.fs.copy {
            from(buildDir) {
                include("**/libdjl_torch.*", "**/djl_torch.dll")
            }
            into(ciDir)
        }
        val dll = ciDir / "Release/djl_torch.dll"
        if (dll.exists()) {
            dll.renameTo(ciDir / "djl_torch.dll")
        }
    }

    fun downloadBuildAndroid() {
        for (abi in listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")) {
            injected.exec.exec {
                workingDir = dir
                commandLine("bash", "build_android.sh", ptVersion, abi)
            }
            val ciDir = dir / "jnilib/${djlVersion}/android/$abi"
            injected.fs.copy {
                from(buildDir / "libdjl_torch.so")
                into(ciDir)
            }
            injected.fs.delete { delete("$buildDir/") }
        }
    }

    fun copyNativeLibToOutputDir(fileStoreMap: Map<String, String>, url: String) {
        for ((key, value) in fileStoreMap) {
            logger.lifecycle("Downloading $url/$key")
            val outputDir = File("$binaryRoot/$value")
            val file = outputDir / "libtorch.zip"
            file.parentFile.mkdirs()
            "$url/$key".url into file
            injected.exec.exec {
                workingDir = dir
                commandLine("unzip", "-q", "-d", outputDir.absolutePath, file.absolutePath)
            }
            injected.fs.delete { delete("$outputDir/libtorch/lib/*.lib") }
            injected.fs.delete { delete("$outputDir/libtorch/lib/*.a") }

            injected.fs.copy {
                from("$outputDir/libtorch/lib/") {
                    include(
                        "libarm_compute*",
                        "libc10_cuda.so",
                        "libc10.*",
                        "libcaffe2_nvrtc.so",
                        "libcu*",
                        "libgfortran-*",
                        "libgomp*",
                        "libiomp*",
                        "libnv*",
                        "libopenblasp-*",
                        "libtorch_cpu.*",
                        "libtorch_cuda*.so",
                        "libtorch.*",
                        "asmjit.dll",
                        "c10_cuda.dll",
                        "c10.dll",
                        "caffe2_nvrtc.dll",
                        "cu*.dll",
                        "fbgemm.dll",
                        "nv*.dll",
                        "torch_cpu.dll",
                        "torch_cuda*.dll",
                        "torch.dll",
                        "uv.dll",
                        "zlibwapi.dll"
                    )
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
            injected.fs.delete { delete(file) }
            injected.fs.delete { delete(outputDir / "libtorch") }
        }
    }

    fun prepareNativeLib(packageType: String?) {
        if ("mac" !in os)
            throw GradleException("This command must be run from osx")

        val officialPytorchUrl = "https://download.pytorch.org/libtorch"
        val aarch64PytorchUrl = "https://djl-ai.s3.amazonaws.com/publish/pytorch"
        val cuda = "cu128"
        if (packageType == "gpu") {
            // @formatter:off
            val files = mapOf(
                "$cuda/libtorch-cxx11-abi-shared-with-deps-$ptVersion%2B$cuda.zip" to "$cuda/linux-x86_64",
                "$cuda/libtorch-win-shared-with-deps-$ptVersion%2B$cuda.zip"       to "$cuda/win-x86_64",
            )
            // @formatter:on

            copyNativeLibToOutputDir(files, officialPytorchUrl)
        } else {
            // @formatter:off
            val files = mapOf(
                "cpu/libtorch-cxx11-abi-shared-with-deps-$ptVersion%2Bcpu.zip"     to "cpu/linux-x86_64",
                "cpu/libtorch-macos-arm64-$ptVersion.zip"                          to "cpu/osx-aarch64",
                "cpu/libtorch-win-shared-with-deps-$ptVersion%2Bcpu.zip"           to "cpu/win-x86_64",
            )
            // @formatter:on

            val aarch64Files = mapOf("$ptVersion/libtorch-linux-aarch64-$ptVersion.zip" to "cpu/linux-aarch64")
            copyNativeLibToOutputDir(files, officialPytorchUrl)
            copyNativeLibToOutputDir(aarch64Files, aarch64PytorchUrl)

            if ("mac" in os) {
                injected.exec.exec {
                    workingDir = dir
                    commandLine(
                        "install_name_tool",
                        "-add_rpath",
                        "@loader_path",
                        "$binaryRoot/cpu/osx-aarch64/native/lib/libtorch_cpu.dylib"
                    )
                }
                injected.exec.exec {
                    workingDir = dir
                    commandLine(
                        "install_name_tool",
                        "-add_rpath",
                        "@loader_path",
                        "$binaryRoot/cpu/osx-aarch64/native/lib/libtorch.dylib"
                    )
                }
            }
        }
    }

    register("compileAndroidJNI") {
        doFirst {
            downloadBuildAndroid()
        }
    }

    register("cleanJNI") {
        val injected = project.objects.newInstance<InjectedOps>()
        val files = fileTree(project.projectDir) { include("**.zip") }
        val dir = projectDir
        doFirst {
            injected.fs.delete {
                delete(
                    dir / "build",
                    dir / "libtorch",
                    dir / "libtorch_android"
                )
            }
            injected.fs.delete { delete(files) }
        }
    }

    register("compileJNI") {
        val ptFlavor = ptFlavor
        val isPrecxx11 = isPrecxx11
        val isAarch64 = isAarch64
        val files = fileTree("$home/.djl.ai/pytorch/") {
            include("**/*djl_torch.*")
        }
        doFirst {
            // You have to use an environment with CUDA persets for Linux and Windows
            when {
                "windows" in os -> downloadBuild("win", ptFlavor)
                "mac" in os -> downloadBuild(
                    "osx",
                    ptFlavor,
                    false,
                    isAarch64
                )

                "linux" in os -> downloadBuild(
                    "linux",
                    ptFlavor,
                    isPrecxx11,
                    isAarch64
                )

                else -> throw IllegalStateException("Unknown Architecture $osName-$ptFlavor")
            }
            injected.fs.delete { delete(files) }
        }
    }

    register("downloadPyTorchNativeLib") {
        val packageType = project.findProperty("package_type")?.toString()
        doLast {
            prepareNativeLib(packageType)
        }
    }

    register("uploadS3") {
        doLast {
            delete("$binaryRoot")
            prepareNativeLib("cpu")
            prepareNativeLib("gpu")

            injected.exec.exec {
                workingDir = dir
                commandLine("sh", "-c", "find $binaryRoot -type f | xargs gzip")
            }

            (binaryRoot / "files.txt").text = buildString {
                val uploadDirs = listOf(
                    binaryRoot / "cpu/linux-x86_64/native/lib/",
                    binaryRoot / "cpu/linux-aarch64/native/lib/",
                    binaryRoot / "cpu/osx-aarch64/native/lib/",
                    binaryRoot / "cpu/win-x86_64/native/lib/",
                    binaryRoot / "cu128/linux-x86_64/native/lib/",
                    binaryRoot / "cu128/win-x86_64/native/lib/",
                )
                for (item in uploadDirs)
                    fileTree(item).files.map { it.name }.forEach {
                        val out = item.toRelativeString(File("${binaryRoot}/"))
                        appendLine(out + "/" + URLEncoder.encode(it, "UTF-8"))
                    }
            }
            injected.exec.exec {
                workingDir = dir
                commandLine("aws", "s3", "sync", "$binaryRoot", "s3://djl-ai/publish/pytorch/$ptVersion/")
            }
        }
    }

    // Create a placeholder jar without classifier to pass sonatype tests but throws an Exception if loaded
    jar {
        val placeholder = buildDirectory / "placeholder"
        // this line is to enforce gradle to build the jar
        // otherwise it doesn't generate the placeholder jar at times
        // when there is no java code inside src/main
        outputs.dir("build/libs")
        val version = project.version
        doFirst {
            val dir = placeholder / "native/lib"
            dir.mkdirs()
            val propFile = placeholder / "native/lib/pytorch.properties"
            propFile.text = "placeholder=true\nversion=${version}\n"
        }

        from(placeholder)
    }

    java {
        withJavadocJar()
        withSourcesJar()
    }

    withType<GenerateModuleMetadata> { enabled = false }

    val requireSigning = project.hasProperty("staging") || project.hasProperty("snapshot")
    val flavorNames: Array<String> = file(binaryRoot).list() ?: emptyArray()
    for (flavor in flavorNames) {

        val platformNames = binaryRoot.resolve(flavor).list() ?: emptyArray()

        val artifactsNames = ArrayList<Task>()

        for (osName in platformNames) {
            register<Jar>("$flavor-${osName}Jar") {
                doFirst {
                    val propFile = binaryRoot / "pytorch.properties"
                    propFile.delete()
                    val dsStore = binaryRoot / flavor / osName / "native/lib/.DS_Store"
                    dsStore.delete()

                    val versionName = "${project.version}-$nowFormatted"
                    val dir = binaryRoot / flavor / osName / "native/lib"
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
                    val metaInf = binaryRoot / flavor / osName / "META-INF"
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
                from(binaryRoot / flavor / osName / "/native/lib") {
                    into("pytorch/$flavor/$osName")
                }
                from(binaryRoot / "pytorch.properties") {
                    into("native/lib")
                }
                from("src/main/resources")
                archiveClassifier = osName
                archiveBaseName = "pytorch-native-$flavor"

                manifest {
                    attributes("Automatic-Module-Name" to "ai.djl.pytorch_native_${flavor}_$osName")
                }
            }
            artifactsNames += named("${flavor}-${osName}Jar").get()
        }

        // Only publish if the project directory equals the current directory
        // This means that publishing from the main project does not publish the native jars
        // and the native jars have to be published separately
        if (project.projectDir.toString() == System.getProperty("user.dir")) {
            publishing.publications.register<MavenPublication>(flavor) {
                artifactId = "pytorch-native-$flavor"
                from(project.components["java"])
                setArtifacts(artifactsNames)
                artifact(jar)
                artifact(named("javadocJar"))
                artifact(named("sourcesJar"))
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

    signing {
        isRequired = requireSigning
        if (isRequired) {
            val signingKey = findProperty("signingKey").toString()
            val signingPassword = findProperty("signingPassword").toString()
            useInMemoryPgpKeys(signingKey, signingPassword)
            sign(publishing.publications)
        }
    }

    // Gradle 8.0 requires explicitly dependency
    withType<PublishToMavenRepository> {
        for (flavor in flavorNames) {
            if (requireSigning) {
                dependsOn("sign${flavor.take(1).uppercase() + flavor.substring(1)}Publication")
            }

            val platformNames = (binaryRoot / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }
    withType<Sign> {
        for (flavor in flavorNames) {
            val platformNames = (binaryRoot / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }

    publishing.repositories {
        maven {
            if (project.hasProperty("snapshot")) {
                name = "snapshot"
                url = uri("https://central.sonatype.com/repository/maven-snapshots/")
                credentials {
                    username = findProperty("sonatypeUsername").toString()
                    password = findProperty("sonatypePassword").toString()
                }
            } else if (project.hasProperty("staging")) {
                name = "staging"
                url = uri("https://ossrh-staging-api.central.sonatype.com/service/local/staging/deploy/maven2/")
                credentials {
                    username = findProperty("sonatypeUsername").toString()
                    password = findProperty("sonatypePassword").toString()
                }
            } else {
                name = "local"
                url = uri("build/repo")
            }
        }
    }

    clean {
        val dir = project.projectDir
        doFirst {
            delete(dir / "jnilib")
            delete(dir.parentFile / "pytorch-jni/jnilib")
        }
    }
}

// Post-publish task to make deployment visible in Central Publisher Portal.
// See https://central.sonatype.org/publish/publish-portal-ossrh-staging-api/#ensuring-deployment-visibility-in-the-central-publisher-portal
if (project.hasProperty("staging")) {
    val url = "https://ossrh-staging-api.central.sonatype.com/manual/upload/defaultRepository/${project.group}"
    val username = findProperty("sonatypeUsername").toString()
    val password = findProperty("sonatypePassword").toString()
    val token = Base64.getEncoder().encodeToString("${username}:${password}".toByteArray())

    tasks.register("postPublish") {
        doLast {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Authorization", "Bearer $token")
            val status = conn.responseCode
            if (status != HttpURLConnection.HTTP_OK) {
                project.logger.error("Failed to POST '${url}'. Received status code ${status}: ${conn.responseMessage}")
            }
        }
    }

    tasks.named("publish") {
        finalizedBy(tasks.named("postPublish"))
    }
}

formatCpp {
    exclusions = listOf("main/patch/**")
}

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations

    @get:Inject
    val exec: ExecOperations
}
