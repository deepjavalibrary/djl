/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.uploader;

import ai.djl.Application;
import java.io.IOException;
import java.util.Map;

public abstract class MetaBuilder<T> {

    private String name;
    private String description;
    private String artifactId;
    private String baseDir;
    private Application application;
    private Map<String, String> properties;
    private Map<String, Object> arguments;

    public abstract T self();

    public abstract T optFilePath(String filePath);

    public abstract T optPythonPath(String pythonPath);

    public T optProperties(Map<String, String> properties) {
        this.properties = properties;
        return self();
    }

    public T optArguments(Map<String, Object> arguments) {
        this.arguments = arguments;
        return self();
    }

    public MetadataBuilder prepareBuild() throws IOException, InterruptedException {
        return MetadataBuilder.builder()
                .setName(name)
                .setDescription(description)
                .setArtifactId(artifactId)
                .setBaseDir(baseDir)
                .setApplication(application)
                .addProperties(properties)
                .addArguments(arguments);
    }

    public T setBaseDir(String baseDir) {
        this.baseDir = baseDir;
        return self();
    }

    public T setName(String name) {
        this.name = name;
        return self();
    }

    public T setDescription(String description) {
        this.description = description;
        return self();
    }

    public T setApplication(Application application) {
        this.application = application;
        return self();
    }

    public T setArtifactId(String artifactId) {
        this.artifactId = artifactId;
        return self();
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public String getArtifactId() {
        return artifactId;
    }

    public String getBaseDir() {
        return baseDir;
    }

    public Application getApplication() {
        return application;
    }

    public Map<String, String> getProperties() {
        return properties;
    }

    public Map<String, Object> getArguments() {
        return arguments;
    }
}
