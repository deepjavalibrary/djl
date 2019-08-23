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
package software.amazon.ai.integration.tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.dataset.ArrayDataset;
import software.amazon.ai.training.dataset.BatchSampler;
import software.amazon.ai.training.dataset.DataLoadingConfiguration;
import software.amazon.ai.training.dataset.RandomSampler;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.training.dataset.Sampler;
import software.amazon.ai.training.dataset.SequenceSampler;

public class DatasetTest {
    public static void main(String[] args) {
        // TODO remove this once NumpyMode is defualt
        String[] cmd = {"-c", DatasetTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testSequenceSampler() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Sampler<Long> sequenceSampler = new SequenceSampler(manager.arange(100).size());
            List<Long> original = new ArrayList<>();
            sequenceSampler.forEachRemaining(original::add);
            List<Long> expected = LongStream.range(0, 100).boxed().collect(Collectors.toList());
            Assertions.assertTrue(original.equals(expected), "SequentialSampler test failed");
        }
    }

    @RunAsTest
    public void testRandomSampler() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Sampler<Long> randomSampler = new RandomSampler(manager.arange(10).size());
            List<Long> original = new ArrayList<>();
            randomSampler.forEachRemaining(original::add);
            Assertions.assertTrue(original.size() == 10, "SequentialSampler test failed");
        }
    }

    @RunAsTest
    public void testBatchSampler() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.arange(100);
            Sampler<List<Long>> batchSampler =
                    new BatchSampler(new SequenceSampler(data.size()), 27, false);
            List<List<Long>> originalList = new ArrayList<>();
            batchSampler.forEachRemaining(originalList::add);
            Assertions.assertTrue(originalList.size() == 4, "size of BatchSampler is not correct");
            List<Long> expected = LongStream.range(0, 27).boxed().collect(Collectors.toList());
            Assertions.assertTrue(
                    originalList.get(0).equals(expected), "data from BatchSampler is not correct");
            Sampler<Long> randomSampler = new RandomSampler(data.size());
            batchSampler = new BatchSampler(randomSampler, 33, true);
            originalList = new ArrayList<>();
            batchSampler.forEachRemaining(originalList::add);
            Assertions.assertTrue(originalList.size() == 3, "size of BatchSampler is not correct");
            // test case when dataset is smaller than batchSize
            batchSampler = new BatchSampler(new SequenceSampler(data.size()), 101, true);
            originalList = new ArrayList<>();
            batchSampler.forEachRemaining(originalList::add);
            Assertions.assertTrue(originalList.isEmpty());
            batchSampler = new BatchSampler(new SequenceSampler(data.size()), 101, false);
            originalList = new ArrayList<>();
            batchSampler.forEachRemaining(originalList::add);
            Assertions.assertTrue(
                    originalList.size() == 1 && originalList.get(0).size() == 100,
                    "dropLast test failed!");
        }
    }

    @RunAsTest
    public void testArrayDataset() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.arange(200).reshape(100, 2);
            // TODO this should be (100) not (100, 1) due to NumpyShape off
            NDArray label = manager.arange(100).reshape(100, 1);
            ArrayDataset dataset =
                    new ArrayDataset(
                            data,
                            label,
                            new DataLoadingConfiguration.Builder().setBatchSize(20).build());
            int index = 0;
            for (Record record : dataset.getRecords()) {
                Assertions.assertEquals(
                        record.getData().get(0),
                        manager.arange(2 * index, 2 * index + 40).reshape(20, 2));
                Assertions.assertEquals(
                        record.getLabels().get(0),
                        manager.arange(index, index + 20).reshape(20, 1));
                index += 20;
            }
            dataset =
                    new ArrayDataset(
                            data,
                            label,
                            new DataLoadingConfiguration.Builder().setBatchSize(15).build());
            index = 0;
            for (Record record : dataset.getRecords()) {
                if (index != 90) {
                    Assertions.assertEquals(
                            record.getData().get(0),
                            manager.arange(2 * index, 2 * index + 30).reshape(15, 2));
                    Assertions.assertEquals(
                            record.getLabels().get(0),
                            manager.arange(index, index + 15).reshape(15, 1));
                } else {
                    // last batch
                    Assertions.assertEquals(
                            record.getData().get(0),
                            manager.arange(2 * index, 2 * index + 20).reshape(10, 2));
                    Assertions.assertEquals(
                            record.getLabels().get(0),
                            manager.arange(index, index + 10).reshape(10, 1));
                }
                index += 15;
            }
        }
    }
}
