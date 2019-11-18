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
package ai.djl.repository.dataset;

import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.util.Progress;
import java.io.IOException;

/**
 * A {@link Dataset} whose data is found in the dataset zoo of a {@link Repository}.
 *
 * <p>The {@code ZooDataset}s are all {@link PreparedDataset}s.
 */
public interface ZooDataset extends Dataset, PreparedDataset {

    /**
     * Returns the {@link MRL} of the dataset.
     *
     * @return the {@link MRL} of the dataset
     */
    MRL getMrl();

    /**
     * Returns the {@link Repository} the dataset is found in.
     *
     * @return the {@link Repository} the dataset is found in
     */
    Repository getRepository();

    /**
     * Returns the {@link Artifact} the dataset is found in.
     *
     * @return the {@link Artifact} the dataset is found in
     */
    Artifact getArtifact();

    /**
     * Returns the {@link ai.djl.training.dataset.Dataset.Usage} of the dataset.
     *
     * @return the {@link ai.djl.training.dataset.Dataset.Usage} of the dataset
     */
    Usage getUsage();

    /**
     * Returns whether the dataset has been prepared.
     *
     * @return true if the dataset has been prepared
     */
    boolean isPrepared();

    /**
     * Sets if the dataset has been prepared.
     *
     * @param prepared true if the dataset has been prepared
     */
    void setPrepared(boolean prepared);

    /**
     * Sets the artifact to the default one.
     *
     * <p>The default artifact is usually found by searching within the repository with a default
     * mrl, version, and filter.
     *
     * @throws IOException for various exceptions depending on the specific dataset
     */
    void useDefaultArtifact() throws IOException;

    /**
     * Prepares the {@link ZooDataset} with the dataset specific behavior.
     *
     * <p>This method is called only when the dataset is not prepared, has an artifact set, and the
     * repository artifact has already been prepared. {@link ZooDataset#setPrepared(boolean)} does
     * not need to be called within this method and will be called after.
     *
     * @param usage the usage to prepare
     * @throws IOException for various exceptions depending on the specific dataset
     */
    void prepareData(Usage usage) throws IOException;

    /** {@inheritDoc} */
    @Override
    default void prepare(Progress progress) throws IOException {
        if (!isPrepared()) {
            if (getArtifact() == null) {
                useDefaultArtifact();
                if (getArtifact() == null) {
                    throw new IOException(getMrl() + " dataset not found.");
                }
            }
            getRepository().prepare(getArtifact(), progress);
            prepareData(getUsage());
            setPrepared(true);
        }
    }
}
