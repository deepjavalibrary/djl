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

package ai.djl.mxnet.engine;
// CHECKSTYLE:OFF:AvoidStaticImport

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import ai.djl.mxnet.jna.LibUtils;
import ai.djl.mxnet.jna.MxnetLibrary;
import ai.djl.mxnet.jna.PointerArray;
import ai.djl.mxnet.test.MockMxnetLibrary;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.SequentialBlock;
import software.amazon.ai.util.PairList;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class CachedOpTest extends PowerMockTestCase {

    private static final Logger logger = LoggerFactory.getLogger(CachedOpTest.class);

    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        MxnetLibrary library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
    }

    @Test
    public void testForward() {
        try (MxNDManager manager = MxNDManager.getSystemManager().newSubManager()) {
            List<Parameter> params = new ArrayList<>();
            params.add(
                    new Parameter(
                            "array0",
                            new SequentialBlock(),
                            manager.create(new Shape(2)),
                            ParameterType.WEIGHT,
                            true));
            params.add(
                    new Parameter(
                            "array1",
                            new SequentialBlock(),
                            manager.create(new Shape(3)),
                            ParameterType.WEIGHT,
                            true));
            params.add(
                    new Parameter(
                            "array2",
                            new SequentialBlock(),
                            manager.create(new Shape(5)),
                            ParameterType.WEIGHT,
                            true));
            params.add(
                    new Parameter(
                            "array3",
                            new SequentialBlock(),
                            manager.create(new Shape(6)),
                            ParameterType.WEIGHT,
                            true));
            List<String> names = Arrays.asList("array0", "array1", "array2", "array3");
            List<Integer> locations = Arrays.asList(1, 2, 5, 6);
            PairList<String, Integer> paramIndices = new PairList<>(names, locations);
            names = Arrays.asList("data0", "data1", "data2");
            locations = Arrays.asList(0, 3, 4);
            PairList<String, Integer> dataIndices = new PairList<>(names, locations);
            CachedOp co =
                    new CachedOp(new PointerArray(), manager, params, paramIndices, dataIndices);
            logger.info("Test: Positioned input");
            NDList input =
                    new NDList(
                            manager.create(new Shape(2)),
                            manager.create(new Shape(4)),
                            manager.create(new Shape(5)));
            co.forward(input);
            MxNDArray[] inputNDArray = co.getInputNDArray();
            Assert.assertEquals(inputNDArray[0].getShape(), new Shape(2));
            Assert.assertEquals(inputNDArray[3].getShape(), new Shape(4));
            Assert.assertEquals(inputNDArray[4].getShape(), new Shape(5));
            logger.info("Test: Named input");
            input = new NDList();
            input.add("data2", manager.create(new Shape(2)));
            input.add("data1", manager.create(new Shape(4)));
            input.add("data0", manager.create(new Shape(5)));
            co.forward(input);
            inputNDArray = co.getInputNDArray();
            Assert.assertEquals(inputNDArray[0].getShape(), new Shape(5));
            Assert.assertEquals(inputNDArray[3].getShape(), new Shape(4));
            Assert.assertEquals(inputNDArray[4].getShape(), new Shape(2));
            logger.info("Test: No input, expect warnings");
            input = new NDList();
            co.forward(input);
            inputNDArray = co.getInputNDArray();
            Assert.assertEquals(inputNDArray[0].getShape(), new Shape(1));
            Assert.assertEquals(inputNDArray[3].getShape(), new Shape(1));
            Assert.assertEquals(inputNDArray[4].getShape(), new Shape(1));
            logger.info("Test: Check the remaining params");
            Assert.assertEquals(inputNDArray[1].getShape(), new Shape(2));
            Assert.assertEquals(inputNDArray[2].getShape(), new Shape(3));
            Assert.assertEquals(inputNDArray[5].getShape(), new Shape(5));
            Assert.assertEquals(inputNDArray[6].getShape(), new Shape(6));
            logger.info("Test: Illegal inputs");
            final NDList input2 = new NDList();
            input2.add("data_not_exist", null);
            Assert.assertThrows(IllegalArgumentException.class, () -> co.forward(input2));
        }
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
