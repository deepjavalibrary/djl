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
import java.io.InputStream;
import java.util.Map;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.DataIterable;
import software.amazon.ai.training.dataset.DataLoadingConfiguration;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.Utils;

// TODO we will have download dataset that CIFAR10, MNIST would extend from

/**
 * MNIST handwritten digits dataset from http://yann.lecun.com/exdb/mnist
 *
 * <p>Each sample is an image (in 3D NDArray) with shape (28, 28, 1).
 */
public final class Mnist implements RandomAccessDataset {

    private static final String ARTIFACT_ID = "mnist";

    private final Repository repository;
    private final NDManager manager;
    private final Artifact artifact;
    private final DataLoadingConfiguration config;
    private NDArray data;
    private NDArray labels;
    private long size;

    private Mnist(Builder builder) throws IOException {
        this.repository = builder.repository;
        this.manager = builder.manager;
        this.artifact = builder.artifact;
        this.config = builder.config;
        repository.prepare(artifact);
        loadData(builder.usage);
    }

    @Override
    public Iterable<Record> getRecords() {
        return new DataIterable(this, config);
    }

    @Override
    public Pair<NDList, NDList> get(long index) {
        return new Pair<>(new NDList(data.get(index)), new NDList(labels.get(index)));
    }

    @Override
    public long size() {
        return size;
    }

    private void loadData(Usage usage) throws IOException {
        Map<String, Artifact.Item> map = artifact.getFiles();
        Artifact.Item imageItem;
        Artifact.Item labelItem;
        switch (usage) {
            case TRAIN:
                imageItem = map.get("train_data");
                labelItem = map.get("train_labels");
                break;
            case TEST:
                imageItem = map.get("test_data");
                labelItem = map.get("test_labels");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        labels = readLabel(labelItem);
        size = labels.size();
        data = readData(imageItem, labels.size());
    }

    private NDArray readData(Artifact.Item item, long length) throws IOException {
        try (InputStream is = repository.openStream(item, null)) {
            if (is.skip(16) != 16) {
                throw new AssertionError("Failed skip data.");
            }

            byte[] buf = Utils.toByteArray(is);
            try (NDArray array = manager.create(new Shape(length, 28, 28, 1), DataType.UINT8)) {
                array.set(buf);
                return array.asType(DataType.FLOAT32, true);
            }
        }
    }

    private NDArray readLabel(Artifact.Item item) throws IOException {
        try (InputStream is = repository.openStream(item, null)) {
            if (is.skip(8) != 8) {
                throw new AssertionError("Failed skip data.");
            }

            byte[] buf = Utils.toByteArray(is);
            try (NDArray array = manager.create(new Shape(buf.length), DataType.UINT8)) {
                array.set(buf);
                return array.asType(DataType.FLOAT32, true);
            }
        }
    }

    public static final class Builder {
        private NDManager manager;
        private Repository repository;
        private Artifact artifact;
        private Usage usage;
        private DataLoadingConfiguration config;

        public Builder(NDManager manager) {
            this.manager = manager;
            this.repository = Datasets.REPOSITORY;
        }

        public Builder(NDManager manager, Repository repository) {
            this.manager = manager;
            this.repository = repository;
        }

        public Builder setUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        public Builder setArtifact(Artifact artifact) {
            this.artifact = artifact;
            return this;
        }

        public Builder setDataLoadingProperty(boolean shuffle, int batchSize, boolean dropLast) {
            this.config =
                    new DataLoadingConfiguration.Builder()
                            .setShuffle(shuffle)
                            .setBatchSize(batchSize)
                            .setDropLast(dropLast)
                            .build();
            return this;
        }

        public Builder setDataLoadingProperty(DataLoadingConfiguration config) {
            if (this.config != null) {
                throw new IllegalArgumentException(
                        "either setDataLoading or setDataLoadingConfig, not both");
            }
            this.config = config;
            return this;
        }

        public Mnist build() throws IOException {
            if (artifact == null) {
                MRL mrl = new MRL(MRL.Dataset.CV, Datasets.GROUP_ID, ARTIFACT_ID);
                artifact = repository.resolve(mrl, "1.0", null);
                // TODO to be refactored
                if (artifact == null) {
                    throw new IOException("MMIST dataset not found.");
                }
            }
            if (this.config == null) {
                this.config =
                        new DataLoadingConfiguration.Builder()
                                .setShuffle(false)
                                .setBatchSize(1)
                                .setDropLast(false)
                                .build();
            }
            return new Mnist(this);
        }
    }
}
