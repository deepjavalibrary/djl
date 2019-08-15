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
package software.amazon.ai.training.dataset;

import java.util.Iterator;
import java.util.List;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.Pair;

// TODO abstract a interface that could be inherited by this and Stream DataIterable
// where the random reads is expensive
public class DataIterable implements Iterable<Record> {
    private RandomAccessDataset dataset;
    private DataLoadingConfiguration config;

    public DataIterable(RandomAccessDataset dataset, DataLoadingConfiguration config) {
        this.dataset = dataset;
        this.config = config;
    }

    @Override
    public Iterator<Record> iterator() {
        return new DataIterator(dataset, config);
    }

    private static class DataIterator implements Iterator<Record> {
        private RandomAccessDataset dataset;
        private long batchSize;
        private boolean shuffle;
        private Sampler<Long> sampler;
        private Sampler<List<Long>> batchSampler;
        private int numWorkers;
        private Batchifier batchifier;
        private boolean pinMemory;
        private boolean dropLast;

        public DataIterator(RandomAccessDataset dataset, DataLoadingConfiguration config) {
            this.dataset = dataset;
            this.batchSize = config.getBatchSize();
            this.shuffle = config.getShuffle();
            this.sampler = config.getSampler();
            this.batchSampler = config.getBatchSampler();
            this.numWorkers = config.getNumWorkers();
            this.batchifier = config.getBatchifier();
            this.pinMemory = config.getPinMemory();
            this.dropLast = config.getDropLast();

            // parameter check
            if (sampler != null && shuffle) {
                throw new IllegalArgumentException(
                        "sampler option is mutually exclusive with shuffle");
            }

            // parameter check
            if (batchSampler != null) {
                if (batchSize != 1 || shuffle || sampler != null || dropLast) {
                    throw new IllegalArgumentException(
                            "batchSampler option is mutually exclusive with batchSize, shuffle, sampler and dropLast");
                }
            }

            if (numWorkers > 0) {
                throw new UnsupportedOperationException("Multi-threading is not support yet");
            }

            if (pinMemory) {
                throw new UnsupportedOperationException("pin memory is not support yet");
            }

            if (sampler == null) {
                if (shuffle) {
                    sampler = new RandomSampler(dataset.size());
                } else {
                    sampler = new SequenceSampler(dataset.size());
                }
            }

            if (batchSampler == null) {
                batchSampler = new BatchSampler(sampler, batchSize, dropLast);
            }

            if (batchifier == null) {
                // default batchifier is StackBatchifier
                batchifier = new StackBatchifier();
            }
        }

        @Override
        public boolean hasNext() {
            return batchSampler.hasNext();
        }

        @Override
        public Record next() {
            List<Long> indices = batchSampler.next();
            NDList[] data = new NDList[indices.size()];
            NDList[] labels = new NDList[indices.size()];
            for (int i = 0; i < indices.size(); i++) {
                Pair<NDList, NDList> dataItem = dataset.get(indices.get(i));
                data[i] = dataItem.getKey();
                labels[i] = dataItem.getValue();
            }
            return new Record(batchifier.batchify(data), batchifier.batchify(labels));
        }
    }
}
