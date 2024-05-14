plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.audio"

dependencies {
    api(project(":api"))
    api(project(":basicdataset"))
    api(libs.bytedeco.javacv) {
        exclude("org.bytedeco", "opencv")
        exclude("org.bytedeco", "openblas")
        exclude("org.bytedeco", "flycapture")
        exclude("org.bytedeco", "libdc1394")
        exclude("org.bytedeco", "libfreenect")
        exclude("org.bytedeco", "libfreenect2")
        exclude("org.bytedeco", "librealsense")
        exclude("org.bytedeco", "librealsense2")
        exclude("org.bytedeco", "videoinput")
        exclude("org.bytedeco", "artoolkitplus")
        exclude("org.bytedeco", "flandmark")
        exclude("org.bytedeco", "leptonica")
        exclude("org.bytedeco", "tesseract")
        exclude("org.bytedeco", "tesseract")
    }
    api(libs.wendykierp.jTransforms)

    runtimeOnly(libs.bytedeco.ffmpeg) { artifact { classifier = "macosx-x86_64" } }
    runtimeOnly(libs.bytedeco.ffmpeg) { artifact { classifier = "linux-x86_64" } }
    runtimeOnly(libs.bytedeco.ffmpeg) { artifact { classifier = "windows-x86_64" } }
    runtimeOnly(libs.bytedeco.ffmpeg) { artifact { classifier = "macosx-arm64" } }
    runtimeOnly(libs.bytedeco.ffmpeg) { artifact { classifier = "linux-arm64" } }

    testImplementation(project(":testing"))
    testRuntimeOnly(libs.apache.log4j.slf4j)
    testRuntimeOnly(project(":engines:pytorch:pytorch-engine"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-jni"))
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL Audio processing extension"
                description = "DJL Audio processing extension"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
