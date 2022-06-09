/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn.norm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class GhostBatchNormTest {

    @Test
    public void originalBatchOfSizeOneSubBatchIntoOneVbsListOfLengthOne() {
        testSubBatchingShapeSize(new Shape(1, 1, 10), 1, 1);
    }

    @Test
    public void originalBatchOfSizeSixSubBatchIntoTwoVbsListOfLengthThree() {
        testSubBatchingShapeSize(new Shape(6, 1, 10), 2, 3);
    }

    @Test
    public void originalBatchOfSize60SubBatchInto64VbsListOfLengthOne() {
        testSubBatchingShapeSize(new Shape(60, 10), 64, 1);
    }

    private void testSubBatchingShapeSize(Shape shape, int vbs, int expectedSize) {
        NDList[] zeroSubList = generateZerosArrayAndSubBatch(shape, vbs);
        Assert.assertEquals(expectedSize, zeroSubList.length);
    }

    private NDList[] generateZerosArrayAndSubBatch(Shape shape, int vbs) {
        try (NDManager manager = NDManager.newBaseManager()) {
            GhostBatchNorm gbn =
                    new GhostBatchNorm(GhostBatchNorm.builder().optVirtualBatchSize(vbs));
            NDArray zerosInput = manager.zeros(shape);
            return gbn.split(new NDList(zerosInput));
        }
    }
}
