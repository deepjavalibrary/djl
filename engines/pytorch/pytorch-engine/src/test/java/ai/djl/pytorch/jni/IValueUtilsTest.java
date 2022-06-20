/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.jni;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class IValueUtilsTest {
    @Test
    public void getInputsTestTupleSyntax() {
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager()) {
            PtNDArray array1 = (PtNDArray) manager.zeros(new Shape(1));
            array1.setName("Test()");
            PtNDArray array2 = (PtNDArray) manager.ones(new Shape(1));
            array2.setName("Test()");
            NDList input = new NDList(array1, array2);
            input.attach(manager);

            IValue[] iValues = IValueUtils.getInputs(input);
            Assert.assertEquals(iValues.length, 1);
            Assert.assertTrue(iValues[0].isTuple());
            Assert.assertEquals(iValues[0].toIValueTuple().length, 2);

            Arrays.stream(iValues).forEach(IValue::close);
        }
    }
}
