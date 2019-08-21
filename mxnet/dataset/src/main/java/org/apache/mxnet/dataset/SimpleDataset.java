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
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.DataIterable;
import software.amazon.ai.training.dataset.MultithreadingDataIterable;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

public abstract class SimpleDataset extends RandomAccessDataset<NDArray, NDArray> {
    protected NDManager manager;
    protected Repository repository;
    protected Artifact artifact;
    protected NDArray data;
    protected NDArray labels;
    protected boolean prepared;
    protected Usage usage;

    public SimpleDataset(BaseBuilder<?> builder) {
        super(builder);
        this.repository = builder.getRepository();
        this.manager = builder.getManager();
        this.artifact = builder.getArtifact();
        this.usage = builder.getUsage();
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
    public Iterable<Batch> getData(Trainer<NDArray, NDArray, ?> trainer) {
        if (!prepared) {
            throw new IllegalStateException("please call prepare() before using the dataser");
        }
        if (config.getExecutor() != null) {
            return new MultithreadingDataIterable<>(this, trainer, sampler, config);
        } else {
            return new DataIterable<>(this, trainer, sampler, config);
        }
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

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<B extends BaseBuilder>
            extends RandomAccessDataset.BaseBuilder<B> {

        private NDManager manager;
        private Usage usage;
        private Repository repository = Datasets.REPOSITORY;
        private Artifact artifact;

        public NDManager getManager() {
            return manager;
        }

        public B setManager(NDManager manager) {
            this.manager = manager;
            return self();
        }

        public Usage getUsage() {
            return usage;
        }

        public B setUsage(Usage usage) {
            this.usage = usage;
            return self();
        }

        public Repository getRepository() {
            return repository;
        }

        public B optRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        public Artifact getArtifact() {
            return artifact;
        }

        public B optArtifact(Artifact artifact) {
            this.artifact = artifact;
            return self();
        }
    }
}
