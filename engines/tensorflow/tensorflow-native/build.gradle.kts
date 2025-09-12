import java.net.HttpURLConnection
import java.net.URL
import java.util.Base64
import java.util.zip.GZIPOutputStream
import java.util.zip.ZipInputStream
import kotlin.collections.ArrayList

plugins {
    ai.djl.javaProject
    `maven-publish`
    signing
}

group = "ai.djl.tensorflow"

val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = libs.versions.tensorflow.get() + if (isRelease) "" else "-SNAPSHOT"
val tfJava = libs.versions.tensorflowCore.get()

infix fun URL.unzipInto(dir: File) {
    dir.mkdirs()
    ZipInputStream(openStream()).use { zis ->
        generateSequence { zis.nextEntry }.forEach {
            if (!it.isDirectory && it.name.startsWith("org/tensorflow/internal/c_api")) {
                val fileName = it.name.split("/").last()
                if (it.name.matches(Regex(".+(\\.so(\\.\\d+)?|\\.dylib|\\.dll|LICENSE.*)"))) {
                    project.logger.lifecycle("gzip $fileName ...")
                    GZIPOutputStream((dir / "${fileName}.gz").outputStream()).use { out ->
                        zis.copyTo(out)
                    }
                }
            }
        }
    }
}

tasks {
    register("uploadTensorflowNativeLibs") {
        val baseDir = project.projectDir
        val tensorflowVersion = libs.versions.tensorflow.get()
        val buildDir = buildDirectory
        val tfJavaVer = tfJava
        val injected = project.objects.newInstance<InjectedOps>()

        doLast {
            val dir = buildDir / "download"
            injected.fs.delete { delete(dir) }

            val url =
                "https://repo1.maven.org/maven2/org/tensorflow/tensorflow-core-native/${tfJavaVer}/tensorflow-core-native-$tfJavaVer"
            val aarch64Url = "https://publish.djl.ai/tensorflow/${tensorflowVersion}/linux-arm64.jar"

            "${url}-macosx-arm64.jar".url unzipInto dir / "cpu/osx-aarch64/native/lib"
            "${url}-windows-x86_64.jar".url unzipInto dir / "cpu/win-x86_64/native/lib"
            "${url}-linux-x86_64.jar".url unzipInto dir / "cpu/linux-x86_64/native/lib"
            "${url}-linux-x86_64-gpu.jar".url unzipInto dir / "cu121/linux-x86_64/native/lib"
            aarch64Url.url unzipInto dir / "cpu/linux-aarch64/native/lib"
            (dir / "cpu/linux-aarch64/native/lib/libomp.so.gz").renameTo(dir / "cpu/linux-aarch64/native/lib/libomp-e9212f90.so.5.gz")

            (dir / "files.txt").text = buildString {
                fileTree(dir).files.forEach {
                    appendLine(it.toRelativeString(dir))
                }
            }

            injected.exec.exec {
                workingDir = baseDir
                commandLine(
                    "aws",
                    "s3",
                    "sync",
                    "$buildDir/download/",
                    "s3://djl-ai/publish/tensorflow/${tensorflowVersion}/"
                )
            }
        }
    }

    jar {
        // this line is to enforce gradle to build the jar
        // otherwise it don't generate the placeholder jar at times
        // when there is no java code inside src/main
        outputs.dir("build/libs")
        var versionName = project.version.toString()
        if (!isRelease)
            versionName += "-$nowFormatted"

        val dir = buildDirectory / "classes/java/main/native/lib"
        doFirst {
            dir.mkdirs()
            val propFile = dir / "tensorflow.properties"
            propFile.text = "placeholder=true\nversion=${versionName}\n"
        }
    }

    java {
        withJavadocJar()
        withSourcesJar()
    }

    withType<GenerateModuleMetadata> { enabled = false }

    val binaryRoot = buildDirectory / "download"
    (binaryRoot / "files.txt").delete()
    val requireSigning = project.hasProperty("staging") || project.hasProperty("snapshot")
    val flavorNames: Array<String> = binaryRoot.list() ?: emptyArray()
    for (flavor in flavorNames) {
        val platformNames = (binaryRoot / flavor).list() ?: emptyArray()

        val artifactsNames = ArrayList<Task>()

        for (osName in platformNames) {
            register<Jar>("$flavor-${osName}Jar") {
                doFirst {
                    val propFile = binaryRoot / flavor / osName / "native/lib/tensorflow.properties"
                    propFile.delete()
                    val dsStore = binaryRoot / flavor / osName / "native/lib/.DS_Store"
                    dsStore.delete()

                    val versionName = "${project.version}-$nowFormatted"
                    val dir = binaryRoot / flavor / osName / "native/lib"
                    propFile.text = buildString {
                        append("version=$versionName\nclassifier=$flavor-$osName\nlibraries=")
                        var first = true
                        for (name in dir.list()!!.sorted()) {
                            if (first)
                                first = false
                            else
                                append(',')
                            append(name)
                        }
                    }

                    from("src/main/resources")
                }
                from(binaryRoot / flavor / osName)
                archiveClassifier = osName
                archiveBaseName = "tensorflow-native-$flavor"

                manifest {
                    attributes("Automatic-Module-Name" to "ai.djl.tensorflow_native_${flavor}_$osName")
                }
            }
            artifactsNames += named("${flavor}-${osName}Jar").get()
        }

        // Only publish if the project directory equals the current directory
        // This means that publishing from the main project does not publish the native jars
        // and the native jars have to be published separately
        if (project.projectDir.toString() == System.getProperty("user.dir")) {
            publishing.publications.create<MavenPublication>(flavor) {
                artifactId = "tensorflow-native-$flavor"
                from(project.components["java"])
                setArtifacts(artifactsNames)
                artifact(jar)
                artifact(named("javadocJar"))
                artifact(named("sourcesJar"))
                pom {
                    name = "DJL release for TensorFlow native binaries"
                    description = "Deep Java Library (DJL) provided TensorFlow native library binary distribution"
                    url = "http://www.djl.ai/engines/tensorflow/${project.name}"
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
        if (requireSigning) {
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

    register("downloadTensorflowNativeLib") {
        doLast {
            delete(binaryRoot)

            val url = "https://publish.djl.ai/tensorflow/${libs.versions.tensorflow.get()}"
            val list = "${url}/files.txt".url.text.split(Regex("\n"))
            for (f in list) {
                if (f.isBlank()) {
                    continue
                }
                project.logger.lifecycle("Downloading $f")
                val file = binaryRoot / f.dropLast(3)
                file.parentFile.mkdirs()
                "$url/$f".url gzipInto file
            }
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

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations
    @get:Inject
    val exec: ExecOperations
}
