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
package ai.djl.fasttext.dataset;

import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.ZooDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/** An abstract class loads fastText dataset. */
public abstract class FtDataset implements Dataset, ZooDataset {

    protected Repository repository;
    protected Usage usage;
    private Artifact artifact;
    private boolean prepared;

    /**
     * Returns cached fastText dataset file path.
     *
     * @return cached fastText dataset file path
     * @throws IOException when IO operation fails in loading a resource
     * @throws TranslateException if there is an error while processing input
     */
    public Path getInputFile() throws IOException, TranslateException {
        prepare(null);

        Map<String, Artifact.Item> map = artifact.getFiles();
        Artifact.Item item;
        switch (usage) {
            case TRAIN:
                item = map.get("train");
                break;
            case TEST:
                item = map.get("test");
                break;
            case VALIDATION:
            default:
                item = map.get("validation");
                break;
        }
        return repository.getFile(item, "").toAbsolutePath();
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Batch> getData(NDManager manager) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Repository getRepository() {
        return repository;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact getArtifact() {
        return artifact;
    }

    /** {@inheritDoc} */
    @Override
    public Usage getUsage() {
        return usage;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isPrepared() {
        return prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void setPrepared(boolean prepared) {
        this.prepared = prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void useDefaultArtifact() throws IOException {
        artifact = repository.resolve(getMrl(), "1.0", null);
    }

    /** {@inheritDoc} */
    @Override
    public void prepareData(Usage usage) {}
}
