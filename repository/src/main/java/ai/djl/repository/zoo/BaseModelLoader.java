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
package ai.djl.repository.zoo;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Metadata;
import ai.djl.repository.Repository;
import ai.djl.repository.VersionRange;
import ai.djl.translate.Translator;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public abstract class BaseModelLoader<I, O> {

    protected Repository repository;
    protected MRL mrl;
    protected String version;

    private Metadata metadata;

    protected BaseModelLoader(Repository repository, MRL mrl, String version) {
        this.repository = repository;
        this.mrl = mrl;
        this.version = version;
    }

    public abstract Translator<I, O> getTranslator();

    private Metadata getMetadata() throws IOException, ModelNotFoundException {
        if (metadata == null) {
            metadata = repository.locate(mrl);
            if (metadata == null) {
                throw new ModelNotFoundException(mrl.getArtifactId() + " Models not found.");
            }
        }
        return metadata;
    }

    public ZooModel<I, O> loadModel(Map<String, String> criteria)
            throws IOException, ModelNotFoundException {
        return loadModel(criteria, Device.defaultDevice());
    }

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

    protected Model loadModel(Artifact artifact, Path modelPath, Device device) throws IOException {
        Model model = Model.newInstance(device);
        model.load(modelPath, artifact.getName());
        return model;
    }

    public Artifact match(Map<String, String> criteria) throws IOException, ModelNotFoundException {
        List<Artifact> list = search(criteria);
        if (list.isEmpty()) {
            return null;
        }

        list.sort(Artifact.COMPARATOR);
        return list.get(0);
    }

    public List<Artifact> search(Map<String, String> criteria)
            throws IOException, ModelNotFoundException {
        return getMetadata().search(VersionRange.parse(version), criteria);
    }
}
