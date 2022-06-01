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
package ai.djl.mxnet.integration;

import ai.djl.mxnet.engine.MxNDArray;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ManagerAttachmentTest {
    @Test
    public void testReturnResource() {
        try (MxNDManager manager = (MxNDManager) NDManager.newBaseManager()) {
            MxNDArray array3x4 = (MxNDArray) manager.ones(new Shape(3, 4));
            try (MxNDManager subManager = (MxNDManager) NDManager.newBaseManager()) {
                array3x4.tempAttach(subManager);
                Assert.assertEquals(array3x4.getManager(), subManager);
            }
            Assert.assertEquals(array3x4.getManager(), manager);
        }
    }

    @Test
    public void testIndexationUsesSpecificManager() {
        try (MxNDManager manager = (MxNDManager) NDManager.newBaseManager()) {
            MxNDArray array3x4 = (MxNDArray) manager.ones(new Shape(3, 4));
            array3x4.setName("Test()");
            NDArray array4 = array3x4.get(1);
            Assert.assertEquals(array4.getManager(), manager);
            try (MxNDManager subManager = (MxNDManager) NDManager.newBaseManager()) {
                NDArray array4_1 = array3x4.get(subManager, 1);
                Assert.assertEquals(array4_1.getManager(), subManager);
                NDArray array4_2 = array3x4.get(new NDIndex(1), subManager);
                Assert.assertEquals(array4_2.getManager(), subManager);
            }
        }
    }
}
