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
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Metadata;
import ai.djl.repository.Repository;
import ai.djl.repository.VersionRange;
import ai.djl.translate.Translator;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Shared code for the {@link ModelLoader} implementations.
 *
 * @param <I> the input type to the loaded model
 * @param <O> the output type of the loaded model
 */
public abstract class BaseModelLoader<I, O> implements ModelLoader<I, O> {

    protected Repository repository;
    protected MRL mrl;
    protected String version;

    private Metadata metadata;

    /**
     * Constructs a {@link ModelLoader} given the repository, mrl, and version.
     *
     * @param repository the repository to load the model from
     * @param mrl the mrl of the model to load
     * @param version the version of the model to load
     */
    protected BaseModelLoader(Repository repository, MRL mrl, String version) {
        this.repository = repository;
        this.mrl = mrl;
        this.version = version;
    }

    /**
     * Returns the default {@link Translator} to use for the loaded model.
     *
     * @param artifact the artifact of the model to load that informs the default translator
     * @return the default translator
     */
    public abstract Translator<I, O> getTranslator(Artifact artifact);

    /** {@inheritDoc} */
    @Override
    public ZooModel<I, O> loadModel(Map<String, String> criteria, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Artifact artifact = match(criteria);
        if (artifact == null) {
            throw new ModelNotFoundException("Model not found.");
        }

        repository.prepare(artifact, progress);

        if (progress != null) {
            progress.reset("Loading", 2);
            progress.update(1);
        }
        try {
            Path dir = repository.getCacheDirectory();
            String relativePath = artifact.getResourceUri().getPath();
            Path modelPath = dir.resolve(relativePath);

            Model model = loadModel(artifact, modelPath, device);
            return new ZooModel<>(model, getTranslator(artifact));
        } finally {
            if (progress != null) {
                progress.end();
            }
        }
    }

    protected Model loadModel(Artifact artifact, Path modelPath, Device device)
            throws IOException, MalformedModelException {
        Model model = Model.newInstance(device);
        model.load(modelPath, artifact.getName());
        return model;
    }

    /** {@inheritDoc} */
    @Override
    public List<Artifact> listModels() throws IOException, ModelNotFoundException {
        return getMetadata().getArtifacts();
    }

    /**
     * Returns the first artifact that matches a given criteria.
     *
     * @param criteria the criteria to match against
     * @return the first artifact that matches the criteria. Null will be returned if no artifact
     *     matches
     * @throws IOException for errors while loading the model
     * @throws ModelNotFoundException if the metadata to get artifacts from is not found
     */
    public Artifact match(Map<String, String> criteria) throws IOException, ModelNotFoundException {
        List<Artifact> list = search(criteria);
        if (list.isEmpty()) {
            return null;
        }
        return list.get(0);
    }

    /**
     * Returns all the artifacts that match a given criteria.
     *
     * @param criteria the criteria to match against
     * @return all the artifacts that match a given criteria
     * @throws IOException for errors while loading the model
     * @throws ModelNotFoundException if the metadata to get artifacts from is not found
     */
    public List<Artifact> search(Map<String, String> criteria)
            throws IOException, ModelNotFoundException {
        return getMetadata().search(VersionRange.parse(version), criteria);
    }

    private Metadata getMetadata() throws IOException, ModelNotFoundException {
        if (metadata == null) {
            metadata = repository.locate(mrl);
            if (metadata == null) {
                throw new ModelNotFoundException(mrl.getArtifactId() + " Models not found.");
            }
        }
        return metadata;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(repository.getName())
                .append(':')
                .append(mrl.getGroupId())
                .append(':')
                .append(mrl.getArtifactId())
                .append(" [\n");
        try {
            for (Artifact artifact : listModels()) {
                sb.append('\t').append(artifact).append('\n');
            }
        } catch (IOException | ModelNotFoundException e) {
            sb.append("\tFailed load metadata.");
        }
        sb.append("\n]");
        return sb.toString();
    }
}
