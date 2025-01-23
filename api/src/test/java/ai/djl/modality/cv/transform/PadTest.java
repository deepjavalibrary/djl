/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.transform;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

public class PadTest {

    @Test
    public void test() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(60, 50, 3));
            Pad pad = new Pad(128);
            NDArray output = pad.transform(array);
            long[] shape = output.getShape().getShape();
            Assert.assertEquals(shape[0], 60);
            Assert.assertEquals(shape[1], 60);

            array = manager.zeros(new Shape(50, 60, 3));
            output = pad.transform(array);
            shape = output.getShape().getShape();
            Assert.assertEquals(shape[0], 60);
            Assert.assertEquals(shape[1], 60);
        }
    }
}
