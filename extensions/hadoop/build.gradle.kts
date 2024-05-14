plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.hadoop"

dependencies {
    api(project(":api"))
    api(libs.apache.hadoop.client) {
        exclude("ch.qos.reload4j", "reload4j")
        exclude("org.slf4j", "slf4j-reload4j")
        exclude("org.apache.hadoop", "hadoop-yarn-api")
        exclude("org.apache.hadoop", "hadoop-yarn-client")
        exclude("org.apache.hadoop", "hadoop-mapreduce-client-core")
        exclude("org.apache.hadoop", "hadoop-mapreduce-client-jobclient")
        exclude("org.apache.avro", "avro")
        exclude("org.eclipse.jetty", "jetty-servlet")
        exclude("org.eclipse.jetty", "jetty-webapp")
        exclude("javax.servlet.jsp", "jsp-api")
        exclude("com.sun.jersey", "jetty-servlet")
        exclude("com.sun.jersey", "jersey-servlet")
    }

    // manually upgrade jackson to latest version for CVEs
    api(libs.jackson.core)

    testImplementation(libs.apache.hadoop.minicluster)
    testImplementation(libs.mockito)
    testImplementation(project(":testing"))

    testRuntimeOnly(libs.junit) // hadoop-client test depends on junit
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "Hadoop hdfs integration for DJL"
                description = "Hadoop hdfs integration for DJL"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
