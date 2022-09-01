/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.basicdataset;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.BatchSampler;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.dataset.SequenceSampler;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class RandomAccessDatasetTest {

    @Test
    public void testRandomSplit() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.arange(0, 100, 1, DataType.INT64);
            ArrayDataset dataset =
                    new ArrayDataset.Builder()
                            .setData(data)
                            .setSampling(new BatchSampler(new SequenceSampler(), 1, false))
                            .build();

            Assert.assertEquals(dataset.size(), 100);

            RandomAccessDataset[] sets = dataset.randomSplit(5, 4, 1);
            Assert.assertEquals(sets[0].size(), 50);
            Assert.assertEquals(sets[1].size(), 40);
            Assert.assertEquals(sets[2].size(), 10);

            long total = sets[0].size() + sets[1].size() + sets[2].size();
            Assert.assertEquals(total, dataset.size());

            RandomAccessDataset subset = dataset.subDataset(10, 25);
            Assert.assertEquals(15, subset.size());
            Record record = subset.get(manager, 0);
            Assert.assertEquals(record.getData().head().getLong(), 10);

            /** Tests {@link #ArrayDataset.getByRange()} */
            List<Long> indices =
                    LongStream.range(10, 25).mapToObj(n -> n).collect(Collectors.toList());
            subset = dataset.subDataset(indices);
            Assert.assertEquals(15, subset.size());
            record = subset.get(manager, 0);
            Assert.assertEquals(record.getData().head().getLong(), 10);

            /** Tests {@link #ArrayDataset.getByIndices()} */
            Collections.reverse(indices);
            subset = dataset.subDataset(indices);
            record = subset.get(manager, 0);
            Assert.assertEquals(record.getData().head().getLong(), 24);

            /** Tests {@link #RandomAccessDataset.subDataset(recordKeys, subRecordKeys)} */
            List<String> subRecordKeys =
                    indices.stream().map(k -> "key" + k).collect(Collectors.toList());
            List<String> recordKeys =
                    LongStream.range(0, 100).mapToObj(k -> "key" + k).collect(Collectors.toList());
            subset = dataset.subDataset(recordKeys, subRecordKeys);
            record = subset.get(manager, 0);
            Assert.assertEquals(record.getData().head().getLong(), 24);
        }
    }
}
