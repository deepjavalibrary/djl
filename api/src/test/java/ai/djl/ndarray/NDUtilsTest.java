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
package ai.djl.ndarray;

import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

public class NDUtilsTest {

    @Test
    public void testGetShape() {
        Shape shape = new Shape(1, 2, 3);
        Shape result = NDUtils.getShapeFromEmptyNDArrayForReductionOp(shape, 1);
        Shape expected = new Shape(1, 3);
        Assert.assertEquals(result, expected);

        Shape empty = new Shape(0, 1);
        Assert.assertThrows(() -> NDUtils.getShapeFromEmptyNDArrayForReductionOp(empty, 0));
    }
}
