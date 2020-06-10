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
package ai.djl.basicdataset;

import ai.djl.basicdataset.utils.FixedBucketSampler;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.PaddingStackBatchifier;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

public class FixedBucketSamplerTest {
    @Test
    public void testFixedBucketSampler() throws IOException {
        FixedBucketSampler fixedBucketSampler = new FixedBucketSampler(10, 100, false, true);
        TatoebaEnglishFrenchDataset dataset =
                TatoebaEnglishFrenchDataset.builder()
                        .setSampling(fixedBucketSampler)
                        .optDataBatchifier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(true)
                                        .addPad(0, 0, (m) -> m.zeros(new Shape(1)), 10)
                                        .build())
                        .optLabelBatchifier(
                                PaddingStackBatchifier.builder()
                                        .optIncludeValidLengths(true)
                                        .addPad(0, 0, (m) -> m.ones(new Shape(1)), 10)
                                        .build())
                        .build();
        dataset.prepare();

        Iterator<List<Long>> iterator = fixedBucketSampler.sample(dataset);
        long count = 0;
        Set<Long> indicesSet = new HashSet<>();
        while (iterator.hasNext()) {
            List<Long> indices = iterator.next();
            int max = Integer.MIN_VALUE;
            int min = Integer.MAX_VALUE;
            for (Long index : indices) {
                int size = dataset.getProcessedText(index, true).size();
                if (size > max) {
                    max = size;
                }
                if (size < min) {
                    min = size;
                }
            }
            indicesSet.addAll(indices);
            count = count + indices.size();
        }
        Assert.assertEquals(count, dataset.size());
        Assert.assertEquals(indicesSet.size(), dataset.size());

        fixedBucketSampler = new FixedBucketSampler(10, 100, false, false);
        iterator = fixedBucketSampler.sample(dataset);
        count = 0;
        indicesSet = new HashSet<>();
        while (iterator.hasNext()) {
            List<Long> indices = iterator.next();
            int max = Integer.MIN_VALUE;
            int min = Integer.MAX_VALUE;
            for (Long index : indices) {
                int size = dataset.getProcessedText(index, true).size();
                if (size > max) {
                    max = size;
                }
                if (size < min) {
                    min = size;
                }
            }
            indicesSet.addAll(indices);
            count = count + indices.size();
        }
        Assert.assertEquals(count, dataset.size());
        Assert.assertEquals(indicesSet.size(), dataset.size());
    }
}
