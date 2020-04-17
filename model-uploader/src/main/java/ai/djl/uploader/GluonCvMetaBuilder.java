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
import ai.djl.uploader.arguments.GluonCvArgs;
import java.io.IOException;

public final class GluonCvMetaBuilder {
    private String filePath = "python/mxnet/gluoncv_import.py";
    private String pythonPath = "python";
    private GluonCvArgs args;
    private String name;
    private String description;
    private String artifactId;
    private String baseDir;
    private Application application;

    public GluonCvMetaBuilder optFilePath(String filePath) {
        this.filePath = filePath;
        return this;
    }

    public GluonCvMetaBuilder optPythonPath(String pythonPath) {
        this.pythonPath = pythonPath;
        return this;
    }

    public GluonCvMetaBuilder setArgs(GluonCvArgs args) {
        this.args = args;
        return this;
    }

    public GluonCvMetaBuilder setBaseDir(String baseDir) {
        this.baseDir = baseDir;
        return this;
    }

    public GluonCvMetaBuilder setName(String name) {
        this.name = name;
        return this;
    }

    public GluonCvMetaBuilder setDescription(String description) {
        this.description = description;
        return this;
    }

    public GluonCvMetaBuilder setApplication(Application application) {
        this.application = application;
        return this;
    }

    public GluonCvMetaBuilder setArtifactId(String artifactId) {
        this.artifactId = artifactId;
        return this;
    }

    public MetadataBuilder prepareBuild() throws IOException, InterruptedException {
        Exporter.processSpawner(filePath, pythonPath, args);
        return MetadataBuilder.builder()
                .setGroupId("ai.djl.mxnet")
                .setApplication(application)
                .setArtifactDir(args.getOutputPath())
                .setArtifactName(args.getName())
                .setName(name)
                .setDescription(description)
                .setArtifactId(artifactId)
                .setBaseDir(baseDir)
                .addArgument("shape", args.getShape());
    }
}
