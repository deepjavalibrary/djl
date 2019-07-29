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
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.repository.Anchor;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.Dataset;

public class Mnist implements Dataset {

    private static final String ARTIFACT_ID = "mnist";

    private Repository repository;
    private Artifact artifact;
    private boolean prepared;

    public Mnist(Repository repository, Artifact artifact) {
        this.repository = repository;
        this.artifact = artifact;
    }

    public static Mnist newInstance() throws IOException {
        return newInstance(Datasets.REPOSITORY);
    }

    public static Mnist newInstance(Repository repository) throws IOException {
        Anchor anchor = new Anchor("dataset", "cv", Datasets.GROUP_ID, ARTIFACT_ID, "1.0");
        Artifact artifact = repository.resolve(anchor);
        return new Mnist(repository, artifact);
    }

    public void prepare() throws IOException {
        if (!prepared) {
            repository.prepare(artifact);
            prepared = true;
        }
    }

    @Override
    public Iterable<NDList> getData(Usage usage, int batchSize, int seed) {
        return null;
    }
}
