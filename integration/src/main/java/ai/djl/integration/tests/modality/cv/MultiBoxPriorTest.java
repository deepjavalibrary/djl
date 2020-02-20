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
package ai.djl.integration.tests.modality.cv;

import ai.djl.modality.cv.MultiBoxPrior;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MultiBoxPriorTest {
    @Test
    public void testMultiBoxPriors() {
        try (NDManager manager = NDManager.newBaseManager()) {
            List<Float> sizes = Arrays.asList(0.2f, 0.272f);
            List<Float> ratios = Arrays.asList(1f, 2f, 0.5f);
            MultiBoxPrior multiBoxPriors =
                    MultiBoxPrior.builder().setSizes(sizes).setRatios(ratios).build();
            NDArray anchors =
                    multiBoxPriors.generateAnchorBoxes(
                            manager.arange(3.0f * 512.0f * 512.0f)
                                    .reshape(new Shape(1, 3, 512, 512)));
            Assert.assertEquals(anchors.getShape(), new Shape(1, 1048576, 4));
        }
    }
}
