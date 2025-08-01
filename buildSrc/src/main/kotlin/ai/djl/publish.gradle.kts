package ai.djl

import org.gradle.kotlin.dsl.*

plugins {
    java
    `maven-publish`
    signing
}

tasks {
    withType<GenerateModuleMetadata> { enabled = false }

    javadoc {
        title = "Deep Java Library $version - ${project.name} API"
        options {
            this as StandardJavadocDocletOptions // https://github.com/gradle/gradle/issues/7038
            encoding = "UTF-8"
            overview = "src/main/javadoc/overview.html"

            addBooleanOption("-allow-script-in-comments", true)
            header =
                "<script type='text/javascript' src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML'></script>"
        }
    }

    publishing {
        publications {
            create<MavenPublication>("maven") {
                from(components["java"])

                pom {
                    name = "Deep Java Library - ${project.name}"
                    description = "Deep Java Library - ${project.name}"
                    url = "http://www.djl.ai/${project.name}"

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

        repositories {
            maven {
                if (project.hasProperty("snapshot")) {
                    name = "snapshot"
                    url = uri("https://central.sonatype.com/repository/maven-snapshots/")
                    // TODO switch name to ossrh and skip credentials with
                    // credentials(PasswordCredentials::class)
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
    }

    // after publications to find the `maven` one just created
    signing {
        isRequired = project.hasProperty("staging") || project.hasProperty("snapshot")
        val signingKey = findProperty("signingKey") as String?
        val signingPassword = findProperty("signingPassword") as String?
        useInMemoryPgpKeys(signingKey, signingPassword)
        sign(publishing.publications["maven"])
    }
}

java {
    withJavadocJar()
    withSourcesJar()
}