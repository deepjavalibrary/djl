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
package ai.djl.basicdataset.utils;

import ai.djl.basicdataset.TextDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Sampler;
import ai.djl.util.RandomUtils;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * {@code FixedBucketSampler} is a {@code Sampler} to be used with {@link TextDataset}, and {@link
 * ai.djl.translate.PaddingStackBatchifier}. It groups text data of same length, and samples them
 * together so that the amount of padding required is minimised. It also makes sure that the
 * sampling is random across epochs.
 */
public class FixedBucketSampler implements Sampler {
    private Set<Bucket> buckets;
    private int numBuckets;
    private int batchSize;
    private boolean dropLast;
    private boolean shuffle;

    /**
     * Constructs a new instance of {@link FixedBucketSampler} with the given number of buckets, and
     * the given batch size.
     *
     * @param batchSize the batch size
     * @param numBuckets the number of buckets
     * @param dropLast whether to drop the last incomplete batch
     * @param shuffle whether to shuffle data randomyl while sampling
     */
    public FixedBucketSampler(int batchSize, int numBuckets, boolean dropLast, boolean shuffle) {
        this.numBuckets = numBuckets;
        this.batchSize = batchSize;
        this.dropLast = dropLast;
        this.shuffle = shuffle;
    }

    /**
     * Constructs a new instance of {@link FixedBucketSampler} with the given number of buckets, and
     * the given batch size.
     *
     * @param batchSize the batch size
     * @param numBuckets the number of buckets
     */
    public FixedBucketSampler(int batchSize, int numBuckets) {
        this(numBuckets, batchSize, false, true);
    }

    /**
     * Constructs a new instance of {@link FixedBucketSampler} with the given number of buckets, and
     * the given batch size.
     *
     * @param batchSize the batch size
     */
    public FixedBucketSampler(int batchSize) {
        this(10, batchSize);
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<List<Long>> sample(RandomAccessDataset dataset) {
        return new Iterate(dataset);
    }

    /** {@inheritDoc} */
    @Override
    public int getBatchSize() {
        return batchSize;
    }

    private static class Sample {
        int sentenceLength;
        long index;

        public Sample(int index, int sentenceLength) {
            this.index = index;
            this.sentenceLength = sentenceLength;
        }
    }

    private static class Bucket {
        Set<Sample> samples;
        int index;

        public Bucket(int index, Set<Sample> samples) {
            this.index = index;
            this.samples = samples;
        }
    }

    private class Iterate implements Iterator<List<Long>> {
        private long current;
        private long size;

        public Iterate(RandomAccessDataset dataset) {
            if (dropLast) {
                this.size = dataset.size() / batchSize;
            } else {
                this.size = (dataset.size() + batchSize - 1) / batchSize;
            }

            if (!(dataset instanceof TextDataset)) {
                throw new IllegalStateException(
                        "FixedBucketSampler can only be used with TextDataset");
            }
            if (buckets == null) {
                List<Sample> samples = new ArrayList<>();
                for (int i = 0; i < dataset.size(); i++) {
                    samples.add(
                            new Sample(
                                    i, ((TextDataset) dataset).getProcessedText(i, true).size()));
                }
                samples.sort(Comparator.comparingInt(o -> o.sentenceLength));
                buckets = new TreeSet<>(Comparator.comparingInt(o -> o.index));
                int bucketSize = samples.size() / numBuckets;
                int bucketNumber = 0;
                for (int i = 0; i < samples.size(); i = i + bucketSize) {
                    int end = i + bucketSize;
                    if (end > samples.size()) {
                        end = samples.size();
                    }
                    buckets.add(new Bucket(bucketNumber++, new HashSet<>(samples.subList(i, end))));
                }
            }
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return current < size;
        }

        /** {@inheritDoc} */
        @Override
        public List<Long> next() {
            int collected = 0;
            List<Sample> allSamples = new ArrayList<>();

            Iterator<Bucket> iterator = buckets.iterator();
            Bucket bucket = firstBucket(iterator);
            while (collected < batchSize) {
                Set<Sample> samples = bucket.samples;
                List<Sample> bucketSamples = new ArrayList<>();
                for (Sample sample : samples) {
                    bucketSamples.add(sample);
                    collected++;
                    if (collected >= batchSize) {
                        break;
                    }
                }
                for (Sample sample : bucketSamples) {
                    samples.remove(sample);
                }
                allSamples.addAll(bucketSamples);
                if (collected >= batchSize) {
                    break;
                }
                if (!iterator.hasNext()) {
                    if (shuffle) {
                        iterator = buckets.iterator();
                    } else {
                        throw new IllegalStateException("Code should never reach here");
                    }
                }
                bucket = iterator.next();
            }
            List<Long> next = new ArrayList<>();
            for (Sample sample : allSamples) {
                next.add(sample.index);
            }
            current++;
            return next;
        }

        private Bucket firstBucket(Iterator<Bucket> iterator) {
            if (shuffle) {
                int firstIndex = RandomUtils.nextInt(buckets.size());
                for (int i = 0; i < firstIndex - 1; i++) {
                    iterator.next();
                }
                return iterator.next();
            }
            return iterator.next();
        }
    }
}
