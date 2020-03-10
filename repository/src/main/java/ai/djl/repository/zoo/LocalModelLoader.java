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
package ai.djl.repository.zoo;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.repository.Artifact;
import ai.djl.repository.Repository;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.Translator;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** A {@link ModelLoader} loads a particular {@link ZooModel} from a local folder. */
public class LocalModelLoader implements ModelLoader<NDList, NDList> {

    private Repository repository;

    /**
     * Creates the model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public LocalModelLoader(Repository repository) {
        this.repository = repository;
    }

    /** {@inheritDoc} */
    @Override
    public String getArtifactId() {
        return repository.getName();
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <S, T> ZooModel<S, T> loadModel(Criteria<S, T> criteria)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Progress progress = criteria.getProgress();
        try {
            Translator<S, T> translator = criteria.getTranslator();
            if (translator == null) {
                translator = (Translator<S, T>) new NoopTranslator();
            }

            if (progress != null) {
                progress.reset("Loading", 2);
                progress.update(1);
            }

            Path dir = repository.getCacheDirectory();
            Model model = Model.newInstance(criteria.getDevice());
            model.load(dir);

            return new ZooModel<>(model, translator);
        } finally {
            if (progress != null) {
                progress.end();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public ZooModel<NDList, NDList> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    /** {@inheritDoc} */
    @Override
    public List<Artifact> listModels() {
        return Collections.emptyList();
    }
}
