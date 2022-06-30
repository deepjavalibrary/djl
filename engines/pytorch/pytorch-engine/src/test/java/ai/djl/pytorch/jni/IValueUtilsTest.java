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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class IValueUtilsTest {

    @Test
    public void testTuple() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.zeros(new Shape(1));
            array1.setName("input1()");
            NDArray array2 = manager.ones(new Shape(1));
            array2.setName("input1()");
            NDList input = new NDList(array1, array2);
            // the NDList is mapped to (input1: Tuple(Tensor))
            input.attach(manager);

            IValue[] iValues = IValueUtils.getInputs(input);
            Assert.assertEquals(iValues.length, 1);
            Assert.assertTrue(iValues[0].isTuple());
            Assert.assertEquals(iValues[0].toIValueTuple().length, 2);

            Arrays.stream(iValues).forEach(IValue::close);
        }
    }

    @Test
    public void testMapOfTensor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.zeros(new Shape(1));
            array1.setName("input1.key1");
            NDArray array2 = manager.ones(new Shape(1));
            array2.setName("input1.key2");
            NDArray array3 = manager.zeros(new Shape(1));
            array3.setName("input2.key1");
            NDArray array4 = manager.ones(new Shape(1));
            array4.setName("input2.key2");
            NDArray array5 = manager.ones(new Shape(1));
            array5.setName("input2.key3");
            NDList input = new NDList(array1, array2, array3, array4, array5);
            // the NDList is mapped to (input1: Dict(str, Tensor), input2: Dict(str, Tensor))
            // the first part of NDArray name is the variable name of the inputs
            // the 2nd part is the key of each value in the dict
            input.attach(manager);

            IValue[] iValues = IValueUtils.getInputs(input);
            Assert.assertEquals(iValues.length, 2);
            Assert.assertTrue(iValues[0].isMap());
            Assert.assertEquals(iValues[1].toIValueMap().size(), 3);

            Arrays.stream(iValues).forEach(IValue::close);
        }
    }

    @Test
    public void testListOfTensor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.zeros(new Shape(1));
            array1.setName("input1[]");
            NDArray array2 = manager.ones(new Shape(1));
            array2.setName("input1[]");
            NDArray array3 = manager.ones(new Shape(1));
            array3.setName("input2");
            NDList input = new NDList(array1, array2, array3);
            // the NDList is mapped to (input1: list(Tensor), input2: Tensor)
            input.attach(manager);

            IValue[] iValues = IValueUtils.getInputs(input);
            Assert.assertEquals(iValues.length, 2);
            Assert.assertTrue(iValues[0].isList());
            Assert.assertEquals(iValues[0].toIValueArray().length, 2);

            Arrays.stream(iValues).forEach(IValue::close);
        }
    }
}
