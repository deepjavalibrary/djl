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

import ai.djl.basicdataset.nlp.TextDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Sampler;
import ai.djl.util.RandomUtils;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code FixedBucketSampler} is a {@code Sampler} to be used with {@link TextDataset}, and {@link
 * ai.djl.translate.PaddingStackBatchifier}. It groups text data of same length, and samples them
 * together so that the amount of padding required is minimised. It also makes sure that the
 * sampling is random across epochs.
 */
public class FixedBucketSampler implements Sampler {

    private static final Logger logger = LoggerFactory.getLogger(FixedBucketSampler.class);

    private int numBuckets;
    private int batchSize;
    private boolean shuffle;

    /**
     * Constructs a new instance of {@link FixedBucketSampler} with the given number of buckets, and
     * the given batch size.
     *
     * @param batchSize the batch size
     * @param numBuckets the number of buckets
     * @param shuffle whether to shuffle data randomly while sampling
     */
    public FixedBucketSampler(int batchSize, int numBuckets, boolean shuffle) {
        this.numBuckets = numBuckets;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        if (batchSize == 1) {
            logger.warn("FixedBucketSampler is not meaningful with batch size 1.");
        }
    }

    /**
     * Constructs a new instance of {@link FixedBucketSampler} with the given number of buckets, and
     * the given batch size.
     *
     * @param batchSize the batch size
     * @param numBuckets the number of buckets
     */
    public FixedBucketSampler(int batchSize, int numBuckets) {
        this(batchSize, numBuckets, true);
    }

    /**
     * Constructs a new instance of {@link FixedBucketSampler} with the given number of buckets, and
     * the given batch size.
     *
     * @param batchSize the batch size
     */
    public FixedBucketSampler(int batchSize) {
        this(batchSize, 10);
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<List<Long>> sample(RandomAccessDataset dataset) {
        if (!(dataset instanceof TextDataset)) {
            throw new IllegalArgumentException(
                    "FixedBucketSampler can only be used with TextDataset");
        }
        return new Iterate((TextDataset) dataset);
    }

    /** {@inheritDoc} */
    @Override
    public int getBatchSize() {
        return batchSize;
    }

    private class Iterate implements Iterator<List<Long>> {

        private List<List<TextDataset.Sample>> buckets;
        private List<int[]> bucketBatch;
        private int current;

        public Iterate(TextDataset dataset) {
            buckets = new ArrayList<>(numBuckets);
            bucketBatch = new ArrayList<>();
            List<TextDataset.Sample> samples = dataset.getSamples();
            int min = samples.get(0).getSentenceLength();
            int max = samples.get(samples.size() - 1).getSentenceLength();
            int step = Math.max((1 + max - min) / numBuckets, 1);
            Set<Integer> set = new HashSet<>(numBuckets);
            for (int i = 0; i < numBuckets; ++i) {
                set.add(Math.max(max - (numBuckets - i - 1) * step, min));
            }
            int[] bucketKeys = set.stream().mapToInt(Integer::intValue).toArray();

            int index = 0;
            List<TextDataset.Sample> list = new ArrayList<>();
            for (TextDataset.Sample sample : samples) {
                if (sample.getSentenceLength() > bucketKeys[index]) {
                    if (!list.isEmpty()) {
                        buckets.add(list);
                        list = new ArrayList<>();
                    }
                    ++index;
                }
                list.add(sample);
            }
            if (!list.isEmpty()) {
                buckets.add(list);
            }
            for (int i = 0; i < buckets.size(); ++i) {
                List<TextDataset.Sample> bucket = buckets.get(i);
                for (int j = 0; j < bucket.size(); j += batchSize) {
                    bucketBatch.add(new int[] {i, j});
                }
            }
            if (shuffle) {
                Collections.shuffle(bucketBatch, RandomUtils.RANDOM);
                buckets.forEach(l -> Collections.shuffle(l, RandomUtils.RANDOM));
            }
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return current < bucketBatch.size();
        }

        /** {@inheritDoc} */
        @Override
        public List<Long> next() {
            int[] batch = bucketBatch.get(current);
            List<Long> ret = new ArrayList<>();
            List<TextDataset.Sample> bucket = buckets.get(batch[0]);
            int end = Math.min(bucket.size(), batch[1] + batchSize);
            for (int i = batch[1]; i < end; ++i) {
                ret.add(bucket.get(i).getIndex());
            }
            current++;
            return ret;
        }
    }
}
