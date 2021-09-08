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
package ai.djl.pytorch.integration;

import ai.djl.ModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;
import ai.djl.pytorch.jni.IValue;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

public class IValueTest {

    @Test
    public void testIValue() {
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager()) {
            PtNDArray array1 = (PtNDArray) manager.zeros(new Shape(1));
            PtNDArray array2 = (PtNDArray) manager.ones(new Shape(1));

            try (IValue ivalue = IValue.from(array1)) {
                Assert.assertTrue(ivalue.isTensor());
                Assert.assertEquals(ivalue.getType(), "Tensor");
                NDArray ret = ivalue.toTensor(manager);
                Assert.assertEquals(ret, array1);
                NDList list = ivalue.toNDList(manager);
                Assert.assertEquals(list.size(), 1);
                Assert.assertEquals(list.head(), array1);
            }

            try (IValue ivalue = IValue.from(true)) {
                Assert.assertTrue(ivalue.isBoolean());
                Assert.assertEquals(ivalue.getType(), "bool");
                Assert.assertTrue(ivalue.toBoolean());
            }

            try (IValue ivalue = IValue.from(1)) {
                Assert.assertTrue(ivalue.isLong());
                Assert.assertEquals(ivalue.getType(), "int");
                Assert.assertEquals(ivalue.toLong(), 1);
            }

            try (IValue ivalue = IValue.from(1d)) {
                Assert.assertTrue(ivalue.isDouble());
                Assert.assertEquals(ivalue.getType(), "float");
                Assert.assertEquals(ivalue.toDouble(), 1d);
            }

            try (IValue ivalue = IValue.from("test")) {
                Assert.assertTrue(ivalue.isString());
                Assert.assertEquals(ivalue.getType(), "str");
                Assert.assertEquals(ivalue.toStringValue(), "test");
            }

            try (IValue ivalue = IValue.listFrom(true, false)) {
                Assert.assertTrue(ivalue.isList());
                Assert.assertEquals(ivalue.getType(), "bool[]");
                Assert.assertTrue(ivalue.isBooleanList());
                Assert.assertEquals(ivalue.toBooleanArray(), new boolean[] {true, false});
            }

            try (IValue ivalue = IValue.listFrom(1, 2)) {
                Assert.assertTrue(ivalue.isLongList());
                Assert.assertEquals(ivalue.getType(), "int[]");
                Assert.assertEquals(ivalue.toLongArray(), new long[] {1, 2});
            }

            try (IValue ivalue = IValue.listFrom(1d, 2d)) {
                Assert.assertTrue(ivalue.isDoubleList());
                Assert.assertEquals(ivalue.getType(), "float[]");
                Assert.assertEquals(ivalue.toDoubleArray(), new double[] {1d, 2d});
            }

            try (IValue ivalue = IValue.listFrom(array1, array2)) {
                Assert.assertTrue(ivalue.isTensorList());
                Assert.assertEquals(ivalue.getType(), "Tensor[]");
                NDArray[] ret = ivalue.toTensorArray(manager);
                Assert.assertEquals(ret.length, 2);
                NDList list = ivalue.toNDList(manager);
                Assert.assertEquals(list.size(), 2);
                Assert.assertEquals(list.head(), array1);

                IValue[] iValues = ivalue.toIValueArray();
                Assert.assertEquals(iValues.length, 2);
                Assert.assertTrue(iValues[0].isTensor());
                Arrays.stream(iValues).forEach(IValue::close);
            }

            Map<String, PtNDArray> map = new ConcurrentHashMap<>();
            map.put("data1", array1);
            map.put("data2", array2);
            try (IValue ivalue = IValue.stringMapFrom(map)) {
                Assert.assertTrue(ivalue.isMap());
                Assert.assertEquals(ivalue.getType(), "Dict(str, Tensor)");
                Map<String, IValue> ret = ivalue.toIValueMap();
                Assert.assertEquals(ret.size(), 2);

                NDList list = ivalue.toNDList(manager);
                Assert.assertEquals(list.size(), 2);
                Assert.assertEquals(list.get("data1"), array1);
            }

            try (IValue iv1 = IValue.from(1);
                    IValue iv2 = IValue.from(2);
                    IValue ivalue = IValue.listFrom(iv1, iv2)) {
                Assert.assertTrue(ivalue.isList());
                Assert.assertEquals(ivalue.getType(), "int[]");
                IValue[] ret = ivalue.toIValueArray();
                Assert.assertEquals(ret[1].toLong(), 2);
            }

            try (IValue iv1 = IValue.listFrom(array1, array2);
                    IValue iv2 = IValue.from(array2);
                    IValue ivalue = IValue.listFrom(iv1, iv2)) {
                Assert.assertTrue(ivalue.isList());
                NDList list = ivalue.toNDList(manager);
                Assert.assertEquals(list.size(), 3);
            }

            try (IValue iv1 = IValue.from(array1);
                    IValue iv2 = IValue.from(array2);
                    IValue ivalue = IValue.tupleFrom(iv1, iv2)) {
                NDList list = ivalue.toNDList(manager);
                Assert.assertEquals(list.size(), 2);
            }

            // Test List<List<int>>
            try (IValue iv1 = IValue.listFrom(1, 2);
                    IValue iv2 = IValue.listFrom(2, 1);
                    IValue ivalue = IValue.listFrom(iv1, iv2)) {
                Assert.assertTrue(ivalue.isList());
                Assert.assertEquals(ivalue.getType(), "int[][]");
                IValue[] ret = ivalue.toIValueArray();
                Assert.assertTrue(ret[1].isList());
            }

            // Test python Tuple: (int[], str, float)
            try (IValue iv1 = IValue.listFrom(1, 2);
                    IValue iv2 = IValue.from("data1");
                    IValue iv3 = IValue.from(1f);
                    IValue ivalue = IValue.tupleFrom(iv1, iv2, iv3)) {
                Assert.assertEquals(ivalue.getType(), "(int[], str, float)");
                Assert.assertTrue(ivalue.isTuple());
                IValue[] ret = ivalue.toIValueTuple();
                Assert.assertTrue(ret[0].isList());
                Assert.assertEquals(ret[2].toDouble(), 1d);
            }
        }
    }

    @Test
    public void testIValueModel() throws IOException, ModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("https://resources.djl.ai/test-models/ivalue_jit.zip")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            PtSymbolBlock block = (PtSymbolBlock) model.getBlock();
            IValue tokens = IValue.listFrom(1, 2, 3);
            IValue cls = IValue.from(0);
            IValue sep = IValue.from(4);
            IValue ret = block.forward(tokens, cls, sep);
            long[] actual = ret.toLongArray();
            Assert.assertEquals(actual, new long[] {0, 1, 2, 3, 4});

            tokens.close();
            cls.close();
            sep.close();
            ret.close();
        }
    }
}
