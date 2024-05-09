/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.integration;

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class PtNDArrayTest {

    @Test
    public void testStringTensor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            String[] str = {"a", "b", "c"};
            NDArray arr = manager.create(str);
            Assert.assertEquals(arr.toString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(), Arrays.toString(str));
            Assert.assertEquals(arr.toDebugString(true), Arrays.toString(str));

            Assert.assertThrows(UnsupportedOperationException.class, () -> arr.get(0));
        }
    }

    @Test
    public void testLargeTensor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(10 * 2850, 18944), DataType.FLOAT32);
            Assert.assertThrows(EngineException.class, array::toByteArray);
        }
    }
}
