plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tensorrt"

dependencies {
    api(project(":api"))

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
    testImplementation(libs.slf4j.simple)
    testRuntimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-engine"))
}

//compileJava.dependsOn(processResources)
//
//processResources {
//    outputs.dir file("${project.buildDir}/classes/java/main/native/lib")
//    doLast {
//        def url = "https://publish.djl.ai/tensorrt/${trt_version}/jnilib/${djl_version}"
//        def files = [
//                "linux-x86_64/libdjl_trt.so": "linux-x86_64/libdjl_trt.so"
//        ]
//        def jnilibDir = "${project.projectDir}/jnilib/${djl_version}"
//        files.each { entry ->
//            def file = new File("${jnilibDir}/${entry.value}")
//            if (file.exists()) {
//                project.logger.lifecycle("prebuilt or cached file found for ${entry.value}")
//            } else if (!project.hasProperty("jni")) {
//                project.logger.lifecycle("Downloading ${url}/${entry.key}")
//                file.getParentFile().mkdirs()
//                new URL("${url}/${entry.key}").withInputStream { i -> file.withOutputStream { it << i } }
//            }
//        }
//        file("${project.buildDir}/classes/java/main/native/lib").mkdirs()
//        copy {
//            from jnilibDir
//            into "${project.buildDir}/classes/java/main/native/lib"
//        }
//
//        // write properties
//        def propFile = file("${project.buildDir}/classes/java/main/native/lib/tensorrt.properties")
//        propFile.text = "version=${trt_version}-${version}"
//    }
//}
//
//test {
//    environment "PATH", "src/test/bin:${environment.PATH}"
//}
//
//publishing {
//    publications {
//        maven(MavenPublication) {
//            artifactId "tensorrt"
//            pom {
//                name = "DJL Engine Adapter for TensorRT"
//                description = "Deep Java Library (DJL) Engine Adapter for TensorRT"
//                url = "http://www.djl.ai/extensions/${project.name}"
//            }
//        }
//    }
//}
//
//apply from: file("${rootProject.projectDir}/tools/gradle/cpp-formatter.gradle")
//
//def buildJNI(String os) {
//    exec {
//        commandLine 'bash', 'build.sh'
//    }
//    // for nightly ci
//    // the reason why we duplicate the folder here is to insert djl_version into the path
//    // so different versions of JNI wouldn't override each other. We don't also want publishDir
//    // to have djl_version as engine would require to know that during the System.load()
//    def classifier = "${os}-x86_64"
//    def ciDir = "${project.projectDir}/jnilib/${djl_version}/${classifier}"
//    copy {
//        def tree = fileTree(project.buildDir)
//        tree.include("libdjl_trt.*")
//        from tree.files
//        into ciDir
//    }
//}
//
//tasks.register('compileJNI') {
//    doFirst {
//        if (System.properties['os.name'].startsWith("Linux")) {
//            buildJNI("linux")
//        } else {
//            throw new IllegalStateException("Unsupported os: " + System.properties['os.name'])
//        }
//    }
//    delete System.getProperty("user.home") + "/.djl.ai/tensorrt"
//}
//
//clean.doFirst {
//    delete System.getProperty("user.home") + "/.djl.ai/tensorrt"
//}
