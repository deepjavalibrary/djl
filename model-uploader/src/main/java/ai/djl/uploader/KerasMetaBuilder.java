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
import ai.djl.uploader.arguments.KerasArgs;
import java.io.IOException;

public class KerasMetaBuilder {

    private String filePath = "python/tensorflow/keras_import.py";
    private String pythonPath = "python";
    private KerasArgs args;
    private String name;
    private String description;
    private String artifactId;
    private String artifactName;
    private String baseDir;
    private Application application;

    public KerasMetaBuilder optFilePath(String filePath) {
        this.filePath = filePath;
        return this;
    }

    public KerasMetaBuilder optPythonPath(String pythonPath) {
        this.pythonPath = pythonPath;
        return this;
    }

    public KerasMetaBuilder setArgs(KerasArgs args) {
        this.args = args;
        return this;
    }

    public KerasMetaBuilder setName(String name) {
        this.name = name;
        return this;
    }

    public KerasMetaBuilder setDescription(String description) {
        this.description = description;
        return this;
    }

    public KerasMetaBuilder setApplication(Application application) {
        this.application = application;
        return this;
    }

    public KerasMetaBuilder setArtifactId(String artifactId) {
        this.artifactId = artifactId;
        return this;
    }

    public KerasMetaBuilder setArtifactName(String artifactName) {
        this.artifactName = artifactName;
        return this;
    }

    public KerasMetaBuilder setBaseDir(String baseDir) {
        this.baseDir = baseDir;
        return this;
    }

    public MetadataBuilder prepareBuild() throws IOException, InterruptedException {
        Exporter.processSpawner(filePath, pythonPath, args);
        return MetadataBuilder.builder()
                .setGroupId("ai.djl.tensorflow")
                .setApplication(application)
                .setArtifactDir(args.getArtifactPath())
                .setArtifactName(artifactName)
                .setName(name)
                .setDescription(description)
                .setArtifactId(artifactId)
                .setBaseDir(baseDir);
    }
}
