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
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class GluonCvMetaBuilder extends MetaBuilder<GluonCvMetaBuilder> {

    private String filePath = "python/mxnet/gluoncv_import.py";
    private String pythonPath = "python";
    private GluonCvArgs args;

    public GluonCvMetaBuilder setArgs(GluonCvArgs args) {
        this.args = args;
        return this;
    }

    @Override
    public GluonCvMetaBuilder self() {
        return this;
    }

    @Override
    public GluonCvMetaBuilder optFilePath(String filePath) {
        this.filePath = filePath;
        return this;
    }

    @Override
    public GluonCvMetaBuilder optPythonPath(String pythonPath) {
        this.pythonPath = pythonPath;
        return this;
    }

    @Override
    public MetadataBuilder prepareBuild() throws IOException, InterruptedException {
        Exporter.processSpawner(filePath, pythonPath, args);
        MetadataBuilder builder = super.prepareBuild();
        if (getApplication() == Application.CV.IMAGE_CLASSIFICATION
                && "imagenet".equals(getProperties().get("dataset"))) {
            Path synset = Paths.get(args.getOutputPath(), args.getName(), "synset.txt");
            if (!Files.exists(synset)) {
                URL url =
                        new URL(
                                "https://mlrepo.djl.ai/model/cv/image_classification/ai/djl/mxnet/synset.txt");
                try (InputStream is = url.openStream()) {
                    Files.copy(is, synset);
                }
            }
        }
        return builder.setGroupId("ai.djl.mxnet")
                .setArtifactDir(Paths.get(args.getOutputPath(), args.getName()))
                .setArtifactName(args.getName());
    }
}
