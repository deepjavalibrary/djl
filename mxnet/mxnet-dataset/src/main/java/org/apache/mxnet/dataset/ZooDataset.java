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
package org.apache.mxnet.dataset;

import java.io.IOException;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.Dataset;

public interface ZooDataset extends Dataset, PreparedDataset {

    MRL getMrl();

    Repository getRepository();

    Artifact getArtifact();

    Usage getUsage();

    boolean isPrepared();

    void setPrepared(boolean prepared);

    void useDefaultArtifact() throws IOException;

    void prepareData(Usage usage) throws IOException;

    @Override
    default void prepare() throws IOException {
        if (!isPrepared()) {
            if (getArtifact() == null) {
                useDefaultArtifact();
                if (getArtifact() == null) {
                    throw new IOException(getMrl() + " dataset not found.");
                }
            }
            getRepository().prepare(getArtifact());
            prepareData(getUsage());
            setPrepared(true);
        }
    }
}
