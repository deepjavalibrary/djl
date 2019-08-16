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
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.DataIterable;
import software.amazon.ai.training.dataset.DataLoadingConfiguration;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.util.Pair;

public abstract class SimpleDataset extends RandomAccessDataset {
    private NDManager manager;
    private Repository repository;
    private Artifact artifact;
    private NDArray data;
    private NDArray labels;
    private boolean prepared;
    private Usage usage;

    public SimpleDataset(NDManager manager, Usage usage, DataLoadingConfiguration config) {
        this(manager, Datasets.REPOSITORY, usage, config);
    }

    public SimpleDataset(
            NDManager manager,
            Repository repository,
            Usage usage,
            DataLoadingConfiguration config) {
        super(config);
        this.repository = repository;
        this.manager = manager;
        this.usage = usage;
        this.prepared = false;
    }

    public SimpleDataset(
            NDManager manager,
            Repository repository,
            Artifact artifact,
            Usage usage,
            DataLoadingConfiguration config) {
        super(config);
        this.repository = repository;
        this.manager = manager;
        this.artifact = artifact;
        this.usage = usage;
        this.prepared = false;
    }

    public void prepare() throws IOException {
        if (!prepared) {
            if (artifact == null) {
                MRL mrl = new MRL(MRL.Dataset.CV, Datasets.GROUP_ID, getArtifactID());
                artifact = repository.resolve(mrl, "1.0", null);
                if (artifact == null) {
                    throw new IOException(String.format("%s dataset not found.", getArtifactID()));
                }
            }
            repository.prepare(artifact);
            loadData(usage);
            prepared = true;
        }
    }

    public abstract String getArtifactID();

    public abstract void loadData(Usage usage) throws IOException;

    @Override
    public Pair<NDList, NDList> get(long index) {
        if (!prepared) {
            throw new IllegalStateException("please call prepare() before using the dataser");
        }
        return new Pair<>(new NDList(data.get(index)), new NDList(labels.get(index)));
    }

    @Override
    public Iterable<Record> getRecords() {
        if (!prepared) {
            throw new IllegalStateException("please call prepare() before using the dataser");
        }
        return new DataIterable(this, getDataLoadingConfiguration());
    }

    public NDManager getManager() {
        return manager;
    }

    public Repository getRepository() {
        return repository;
    }

    public Artifact getArtifact() {
        return artifact;
    }

    public NDArray getData() {
        return data;
    }

    public void setData(NDArray data) {
        this.data = data;
    }

    public NDArray getLabels() {
        return labels;
    }

    public void setLabels(NDArray labels) {
        this.labels = labels;
    }
}
