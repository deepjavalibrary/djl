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

/**
 * CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html Each sample
 * is an image (in 3D NDArray) with shape (32, 32, 3).
 */
public final class Cifar10 implements RandomAccessDataset {

    private static final String ARTIFACT_ID = "cifar10";
    // 3072 = 32 * 32 * 3, i.e. one image size, +1 here is label
    private static final int DATA_AND_LABLE_SIZE = 32 * 32 * 3 + 1;

    private final Repository repository;
    private final NDManager manager;
    private final Artifact artifact;
    private final DataLoadingConfiguration config;
    private NDArray data;
    private NDArray labels;
    private long size;

    private Cifar10(Builder builder) throws IOException {
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
        Artifact.Item item;
        switch (usage) {
            case TRAIN:
                item = map.get("data_batch.bin");
                break;
            case TEST:
                item = map.get("test_batch.bin");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        NDArray dataAndLabels = readData(item);
        data = dataAndLabels.get(":, 1:").reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1);
        labels = dataAndLabels.get(":,0");
        // check if data and labels have the same size
        if (data.size(0) != labels.size(0)) {
            throw new IOException(
                    String.format(
                            "the size of data %d didn't match with the size of labels %d",
                            data.size(0), labels.size(0)));
        }
        size = labels.size();
    }

    public NDArray readData(Artifact.Item item) throws IOException {
        try (InputStream is = repository.openStream(item, null)) {
            byte[] buf = Utils.toByteArray(is);
            int length = buf.length / DATA_AND_LABLE_SIZE;
            try (NDArray array =
                    manager.create(new Shape(length, DATA_AND_LABLE_SIZE), DataType.UINT8)) {
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

        public Cifar10 build() throws IOException {
            if (artifact == null) {
                MRL mrl = new MRL(MRL.Dataset.CV, Datasets.GROUP_ID, ARTIFACT_ID);
                artifact = repository.resolve(mrl, "1.0", null);
                // TODO to be refactored
                if (artifact == null) {
                    throw new IOException("CIFAR10 dataset not found.");
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
            return new Cifar10(this);
        }
    }
}
