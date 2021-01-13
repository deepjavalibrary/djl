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

import ai.djl.basicdataset.nlp.TatoebaEnglishFrenchDataset;
import ai.djl.basicdataset.utils.FixedBucketSampler;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

public class FixedBucketSamplerTest {

    @Test
    public void testFixedBucketSampler() throws IOException, TranslateException {
        FixedBucketSampler fixedBucketSampler = new FixedBucketSampler(10, 10, false);
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
                        .optLimit(200)
                        .build();

        dataset.prepare();

        Iterator<List<Long>> iterator = fixedBucketSampler.sample(dataset);
        long count = 0;
        Set<Long> indicesSet = new HashSet<>();
        while (iterator.hasNext()) {
            List<Long> indices = iterator.next();
            indicesSet.addAll(indices);
            count += indices.size();
        }
        Assert.assertEquals(count, dataset.size());
        Assert.assertEquals(indicesSet.size(), dataset.size());

        fixedBucketSampler = new FixedBucketSampler(10, 5, true);
        iterator = fixedBucketSampler.sample(dataset);
        count = 0;
        indicesSet.clear();
        while (iterator.hasNext()) {
            List<Long> indices = iterator.next();
            indicesSet.addAll(indices);
            count = count + indices.size();
        }
        Assert.assertEquals(count, dataset.size());
        Assert.assertEquals(indicesSet.size(), dataset.size());
    }
}
