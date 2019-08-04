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
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.RandomAccess;
import java.util.zip.GZIPInputStream;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.Batchifier;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.dataset.Sampler;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.Utils;

public class Mnist implements Dataset, RandomAccess {

    private static final String ARTIFACT_ID = "mnist";

    private Repository repository;
    private NDManager manager;
    private Artifact artifact;
    private boolean prepared;
    private NDArray trainData;
    private NDArray trainLabels;
    private NDArray testData;
    private NDArray testLabels;

    public Mnist(NDManager manager, Repository repository, Artifact artifact) {
        this.manager = manager;
        this.repository = repository;
        this.artifact = artifact;
    }

    public static Mnist newInstance(NDManager manager) throws IOException {
        return newInstance(manager, Datasets.REPOSITORY);
    }

    public static Mnist newInstance(NDManager manager, Repository repository) throws IOException {
        MRL mrl = new MRL(MRL.Dataset.CV, Datasets.GROUP_ID, ARTIFACT_ID);
        Artifact artifact = repository.resolve(mrl, "1.0", null);
        if (artifact == null) {
            throw new IOException("MMIST dataset not found.");
        }
        return new Mnist(manager, repository, artifact);
    }

    public void prepare() throws IOException {
        if (!prepared) {
            repository.prepare(artifact);
            loadTrainData();
            loadTestData();
            prepared = true;
        }
    }

    // TODO: Does label concept applies to all dataset?
    @Override
    public Iterator<Pair<NDList, NDList>> getData(Usage usage, int batchSize, Sampler sampler) {
        // TODO: How to batch should be determind by each dataset, right?
        Batchifier batchifier = Batchifier.STACK_BATCHIFIER;
        switch (usage) {
            case TRAIN:
                sampler.init((int) trainLabels.size());
                return new DataIterator(trainData, trainLabels, sampler, batchSize, batchifier);
            case TEST:
                sampler.init((int) testLabels.size());
                return new DataIterator(testData, testLabels, sampler, batchSize, batchifier);
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
    }

    // Multithreading may not make sense for all dataset, it's update to each dataset's
    // implementation to honor concurrentWorker.
    public Iterator<Pair<NDList, NDList>> getData(
            Usage usage, int batchSize, Sampler sampler, int concurrentWorker) {
        return null;
    }

    void loadTrainData() throws IOException {
        if (trainData == null) {
            Map<String, Artifact.Item> map = artifact.getFiles();
            Artifact.Item imageItem = map.get("train_data");
            Artifact.Item labelItem = map.get("train_labels");
            trainLabels = readLabel(labelItem);
            trainData = readData(imageItem, trainLabels.size());
        }
    }

    void loadTestData() throws IOException {
        if (testData == null) {
            Map<String, Artifact.Item> map = artifact.getFiles();
            Artifact.Item imageItem = map.get("test_data");
            Artifact.Item labelItem = map.get("test_labels");
            testLabels = readLabel(labelItem);
            testData = readData(imageItem, testLabels.size());
        }
    }

    private NDArray readData(Artifact.Item item, long length) throws IOException {
        try (InputStream is = repository.openStream(item);
                GZIPInputStream zis = new GZIPInputStream(is)) {
            if (zis.skip(16) != 16) {
                throw new AssertionError("Failed skip data.");
            }

            byte[] buf = Utils.toByteArray(zis);
            try (NDArray array = manager.create(new Shape(length, 28, 28), DataType.UINT8)) {
                array.set(buf);
                return array.asType(DataType.FLOAT32, true);
            }
        }
    }

    private NDArray readLabel(Artifact.Item item) throws IOException {
        try (InputStream is = repository.openStream(item);
                GZIPInputStream zis = new GZIPInputStream(is)) {
            if (zis.skip(8) != 8) {
                throw new AssertionError("Failed skip data.");
            }

            byte[] buf = Utils.toByteArray(zis);
            try (NDArray array = manager.create(new Shape(buf.length), DataType.UINT8)) {
                array.set(buf);
                return array.asType(DataType.FLOAT32, true);
            }
        }
    }

    private static class DataIterator implements Iterator<Pair<NDList, NDList>> {

        private Sampler sampler;
        private int batchSize;
        private Batchifier batchifier;
        private NDArray[] d;
        private NDArray[] l;
        private NDArray data;
        private NDArray labels;
        private Pair<NDList, NDList> current;

        DataIterator(
                NDArray data,
                NDArray labels,
                Sampler sampler,
                int batchSize,
                Batchifier batchifier) {
            this.data = data;
            this.labels = labels;
            this.sampler = sampler;
            this.batchSize = batchSize;
            this.batchifier = batchifier;
            d = new NDArray[batchSize];
            l = new NDArray[batchSize];
        }

        @Override
        public boolean hasNext() {
            if (current == null) {
                current = nextInternal();
            }
            return current != null;
        }

        @Override
        public Pair<NDList, NDList> next() {
            if (current == null) {
                current = nextInternal();
            }
            if (current == null) {
                throw new NoSuchElementException();
            }
            return current;
        }

        private Pair<NDList, NDList> nextInternal() {
            for (int i = 0; i < batchSize; ++i) {
                if (!sampler.hasNext()) {
                    return null;
                }

                int index = sampler.next();
                d[i] = data.get(index);
                l[i] = labels.get(index);
            }
            NDList key = new NDList(batchifier.batch(d));
            NDList value = new NDList(batchifier.batch(l));
            return new Pair<>(key, value);
        }
    }
}
