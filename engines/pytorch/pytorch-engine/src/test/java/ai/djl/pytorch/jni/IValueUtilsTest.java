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
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.testing.TestRequirements;
import ai.djl.util.Pair;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;

public class IValueUtilsTest {

    @Test
    public void testTuple() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.zeros(new Shape(1));
            array1.setName("input1()");
            NDArray array2 = manager.ones(new Shape(1));
            array2.setName("input1()");
            NDList input = new NDList(array1, array2);
            // the NDList is mapped to (input1: Tuple(Tensor))
            input.attach(manager);

            Pair<IValue[], String> result = IValueUtils.getInputs(input);
            IValue[] iValues = result.getKey();
            Assert.assertEquals(iValues.length, 1);
            Assert.assertTrue(iValues[0].isTuple());
            Assert.assertEquals(iValues[0].toIValueTuple().length, 2);
            Assert.assertEquals(result.getValue(), "forward");

            Arrays.stream(iValues).forEach(IValue::close);
        }
    }

    @Test
    public void testTupleOfTuple() {
        TestRequirements.notMacX86();

        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager()) {
            NDArray array1 = manager.zeros(new Shape(1));
            array1.setName("input1(2,3)");
            NDArray array2 = manager.ones(new Shape(2));
            array2.setName("input1(2,3)");
            NDArray array3 = manager.ones(new Shape(3));
            array3.setName("input1(2,3)");
            NDArray array4 = manager.ones(new Shape(4));
            array4.setName("input1(2,3)");
            NDArray array5 = manager.ones(new Shape(5));
            array5.setName("input1(2,3)");
            NDArray array6 = manager.ones(new Shape(6));
            array6.setName("input1(2,3)");
            NDList input = new NDList(array1, array2, array3, array4, array5, array6);
            // the NDList is mapped to (input1: Tuple((t1, t2, t3), (t34, t5, t6))
            input.attach(manager);

            Pair<IValue[], String> result = IValueUtils.getInputs(input);
            IValue[] iValues = result.getKey();
            Assert.assertEquals(result.getValue(), "forward");
            Assert.assertEquals(iValues.length, 1);
            Assert.assertTrue(iValues[0].isTuple());

            IValue[] tuple = iValues[0].toIValueTuple();
            Assert.assertEquals(tuple.length, 2);
            Assert.assertTrue(tuple[0].isTuple());
            IValue[] subTuple = tuple[1].toIValueTuple();
            Assert.assertEquals(subTuple.length, 3);
            NDList output = iValues[0].toNDList(manager);
            Assert.assertEquals(output.size(), 6);
            Assert.assertEquals(output.get(5).getShape().get(0), 6);

            Arrays.stream(iValues).forEach(IValue::close);

            NDList input2 = new NDList(array1, array2);
            Assert.assertThrows(() -> IValueUtils.getInputs(input2));
        }
    }

    @Test
    public void testMapOfTensor() {
        TestRequirements.notMacX86();

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

            IValue[] iValues = IValueUtils.getInputs(input).getKey();
            Assert.assertEquals(iValues.length, 2);
            Assert.assertTrue(iValues[0].isMap());
            Assert.assertEquals(iValues[1].toIValueMap().size(), 3);

            Arrays.stream(iValues).forEach(IValue::close);
        }
    }

    @Test
    public void testListOfTensor() {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array1 = manager.zeros(new Shape(1));
            array1.setName("input1[]");
            NDArray array2 = manager.ones(new Shape(1));
            array2.setName("input1[]");
            NDArray array3 = manager.ones(new Shape(1));
            array3.setName("input2");
            NDArray array4 = manager.create("Hello World");
            array4.setName("module_method:get_text_feature");
            NDList input = new NDList(array1, array2, array3, array4);
            // the NDList is mapped to (input1: list(Tensor), input2: Tensor)
            input.attach(manager);

            Pair<IValue[], String> result = IValueUtils.getInputs(input);
            IValue[] iValues = result.getKey();
            Assert.assertEquals(iValues.length, 2);
            Assert.assertTrue(iValues[0].isList());
            Assert.assertEquals(iValues[0].toIValueArray().length, 2);
            Assert.assertEquals(result.getValue(), "get_text_feature");

            Arrays.stream(iValues).forEach(IValue::close);
        }
    }
}
