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
package ai.djl.integration.tests.translate;

import ai.djl.integration.util.TestUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.SimplePaddingStackBatchifier;

import org.testng.Assert;
import org.testng.annotations.Test;

public class SimplePaddingStackBatchifierTest {

    @Test
    public void testBatchify() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDList[] input = new NDList[5];
            for (int i = 0; i < 5; i++) {
                NDArray array1 = manager.zeros(new Shape(10, i + 1));
                NDArray array2 = manager.zeros(new Shape());
                array1.setName("array1");
                array2.setName("array2");
                input[i] = new NDList(array1, array2);
            }
            Batchifier batchifier = new SimplePaddingStackBatchifier();
            NDList actual = batchifier.batchify(input);

            Assert.assertEquals(actual.size(), 2);
            Assert.assertEquals(actual.get(0).getShape(), new Shape(5, 10, 5));
            Assert.assertEquals(actual.get(1).getShape(), new Shape(5));
            // assert names are kept after bachify
            Assert.assertEquals(actual.get(0).getName(), "array1");
            Assert.assertEquals(actual.get(1).getName(), "array2");
        }
    }

    @Test
    public void testUnbatchify() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDList input =
                    new NDList(manager.zeros(new Shape(10, 11)), manager.zeros(new Shape(10)));
            Batchifier batchifier = new SimplePaddingStackBatchifier();
            NDList[] actual = batchifier.unbatchify(input);

            Assert.assertEquals(actual.length, 10);
            for (NDList arrays : actual) {
                Assert.assertEquals(arrays.size(), 2);
                Assert.assertEquals(arrays.get(0).getShape(), new Shape(11));
                Assert.assertEquals(arrays.get(1).getShape(), new Shape());
            }
        }
    }

    @Test
    public void testSplit() {
        try (NDManager manager = NDManager.newBaseManager(TestUtils.getEngine())) {
            NDList input =
                    new NDList(manager.zeros(new Shape(10, 11)), manager.zeros(new Shape(10)));
            Batchifier batchifier =
                    PaddingStackBatchifier.builder()
                            .optIncludeValidLengths(false)
                            .addPad(0, 1, (mngr) -> mngr.zeros(new Shape(10, 1)))
                            .build();
            NDList[] actual = batchifier.split(input, 3, false);

            Assert.assertEquals(actual.length, 3);
            for (int i = 0; i < actual.length - 1; i++) { // Test all but last
                NDList arrays = actual[i];
                Assert.assertEquals(arrays.size(), 2);
                Assert.assertEquals(arrays.get(0).getShape(), new Shape(4, 11));
                Assert.assertEquals(arrays.get(1).getShape(), new Shape(4));
            }

            NDList lastArrays = actual[2];
            Assert.assertEquals(lastArrays.size(), 2);
            Assert.assertEquals(lastArrays.get(0).getShape(), new Shape(2, 11));
            Assert.assertEquals(lastArrays.get(1).getShape(), new Shape(2));
        }
    }
}
