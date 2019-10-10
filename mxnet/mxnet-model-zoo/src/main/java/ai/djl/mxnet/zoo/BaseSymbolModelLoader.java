/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.mxnet.zoo;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.zoo.BaseModelLoader;
import ai.djl.zoo.ModelNotFoundException;
import ai.djl.zoo.ZooModel;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

public abstract class BaseSymbolModelLoader<I, O> extends BaseModelLoader<I, O> {

    protected BaseSymbolModelLoader(Repository repository, MRL mrl, String version) {
        super(repository, mrl, version);
    }

    @Override
    public ZooModel<I, O> loadModel(Map<String, String> criteria, Device device)
            throws IOException, ModelNotFoundException {
        Artifact artifact = match(criteria);
        if (artifact == null) {
            throw new ModelNotFoundException("Model not found.");
        }
        repository.prepare(artifact);
        Path dir = repository.getCacheDirectory();
        String relativePath = artifact.getResourceUri().getPath();
        Path modelPath = dir.resolve(relativePath);
        Model model = loadModel(artifact, modelPath, device);
        return new ZooModel<>(model, getTranslator());
    }

    @Override
    protected Model loadModel(Artifact artifact, Path modelPath, Device device) throws IOException {
        Model model = Model.newInstance(device);
        model.load(modelPath, artifact.getName());
        return model;
    }
}
