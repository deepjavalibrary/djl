/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.dataset;

import ai.djl.ndarray.NDManager;
import ai.djl.util.Progress;
import ai.djl.util.RandomUtils;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class RandomSamplerTest {

    @Test
    public void testTwoRecordOrders() {
        RandomUtils.RANDOM.setSeed(0);
        Assert.assertEquals(sample(2), Arrays.asList(0L, 1L));

        RandomUtils.RANDOM.setSeed(4096);
        Assert.assertEquals(sample(2), Arrays.asList(1L, 0L));
    }

    @Test
    public void testThreeRecordPermutations() {
        Set<List<Long>> permutations = new HashSet<>();
        for (int seed = 0; seed < 16; ++seed) {
            RandomUtils.RANDOM.setSeed(seed);
            permutations.add(sample(3));
        }

        Set<List<Long>> expected =
                new HashSet<>(
                        Arrays.asList(
                                Arrays.asList(0L, 1L, 2L),
                                Arrays.asList(0L, 2L, 1L),
                                Arrays.asList(1L, 0L, 2L),
                                Arrays.asList(1L, 2L, 0L),
                                Arrays.asList(2L, 0L, 1L),
                                Arrays.asList(2L, 1L, 0L)));
        Assert.assertEquals(permutations, expected);
    }

    @DataProvider(name = "datasetSizes")
    public Object[][] datasetSizes() {
        return new Object[][] {{0}, {1}, {2}, {3}, {100}};
    }

    @Test(dataProvider = "datasetSizes")
    public void testSampleContainsEachIndexOnce(int size) {
        RandomUtils.RANDOM.setSeed(0);
        List<Long> indices = sample(size);
        List<Long> expected = LongStream.range(0, size).boxed().collect(Collectors.toList());
        indices.sort(null);
        Assert.assertEquals(indices, expected);
    }

    @Test
    public void testReproducibleOrder() {
        RandomUtils.RANDOM.setSeed(1234);
        List<Long> first = sample(100);
        RandomUtils.RANDOM.setSeed(1234);
        Assert.assertEquals(sample(100), first);
    }

    private List<Long> sample(int size) {
        Iterator<Long> iterator = new RandomSampler().sample(new SizedDataset(size));
        List<Long> indices = new ArrayList<>();
        iterator.forEachRemaining(indices::add);
        Assert.assertFalse(iterator.hasNext());
        Assert.assertThrows(NoSuchElementException.class, iterator::next);
        return indices;
    }

    private static final class SizedDataset extends RandomAccessDataset {

        private long size;

        SizedDataset(long size) {
            this.size = size;
            limit = Long.MAX_VALUE;
        }

        @Override
        protected long availableSize() {
            return size;
        }

        @Override
        public Record get(NDManager manager, long index) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void prepare(Progress progress) {}
    }
}
