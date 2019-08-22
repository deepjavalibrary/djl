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
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.dataset.DataIterable;
import software.amazon.ai.training.dataset.DataLoadingConfiguration;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.training.dataset.Sampler;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

public abstract class SimpleDataset extends RandomAccessDataset<NDArray, NDArray> {
    private NDManager manager;
    private Repository repository;
    private Artifact artifact;
    private NDArray data;
    private NDArray labels;
    private boolean prepared;
    private Usage usage;

    public SimpleDataset(
            NDManager manager, Usage usage, Sampler sampler, DataLoadingConfiguration config) {
        this(manager, Datasets.REPOSITORY, usage, sampler, config);
    }

    public SimpleDataset(
            NDManager manager,
            Repository repository,
            Usage usage,
            Sampler sampler,
            DataLoadingConfiguration config) {
        super(sampler, config);
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
            Sampler sampler,
            DataLoadingConfiguration config) {
        super(sampler, config);
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
    public Pair<NDArray, NDArray> get(long index) {
        if (!prepared) {
            throw new IllegalStateException("please call prepare() before using the dataser");
        }
        return new Pair<>(data.get(index), labels.get(index));
    }

    @Override
    public Iterable<Record> getRecords(Trainer<NDArray, NDArray, ?> trainer) {
        if (!prepared) {
            throw new IllegalStateException("please call prepare() before using the dataser");
        }
        return new DataIterable<>(this, trainer, sampler, config);
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

    public static class DefaultTranslator implements TrainTranslator<NDArray, NDArray, NDArray> {

        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) throws Exception {
            return list.get(0);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, NDArray input) throws Exception {
            return new NDList(input);
        }

        @Override
        public Record processInput(TranslatorContext ctx, NDArray input, NDArray label)
                throws Exception {
            return new Record(new NDList(input), new NDList(label));
        }
    }
}
