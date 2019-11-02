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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
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

    @Test(expectedExceptions = NullPointerException.class)
    public void testForward() {
        try (MxNDManager manager = MxNDManager.getSystemManager().newSubManager()) {
            List<Parameter> params = new ArrayList<>();
            Parameter parameter =
                    new Parameter("data0", new SequentialBlock(), ParameterType.OTHER, false);
            parameter.setArray(manager.create(new Shape(2)));
            params.add(parameter);

            parameter = new Parameter("array0", new SequentialBlock(), ParameterType.WEIGHT, true);
            parameter.setArray(manager.create(new Shape(2)));
            params.add(parameter);

            parameter = new Parameter("array1", new SequentialBlock(), ParameterType.WEIGHT, true);
            parameter.setArray(manager.create(new Shape(3)));
            params.add(parameter);

            parameter = new Parameter("data1", new SequentialBlock(), ParameterType.OTHER, false);
            parameter.setArray(manager.create(new Shape(2)));
            params.add(parameter);

            parameter = new Parameter("data2", new SequentialBlock(), ParameterType.OTHER, false);
            parameter.setArray(manager.create(new Shape(5)));
            params.add(parameter);

            parameter = new Parameter("array2", new SequentialBlock(), ParameterType.WEIGHT, true);
            parameter.setArray(manager.create(new Shape(5)));
            params.add(parameter);

            parameter = new Parameter("array3", new SequentialBlock(), ParameterType.WEIGHT, true);
            parameter.setArray(manager.create(new Shape(6)));
            params.add(parameter);

            List<Integer> paramIndices = Arrays.asList(1, 2, 5, 6);
            List<String> names = Arrays.asList("data0", "data1", "data2");
            List<Integer> locations = Arrays.asList(0, 3, 4);
            PairList<String, Integer> dataIndices = new PairList<>(names, locations);

            ParameterStore parameterStore = new ParameterStore(manager, false);
            CachedOp co =
                    new CachedOp(new PointerArray(), manager, params, paramIndices, dataIndices);
            logger.info("Test: Positioned input");
            NDList input =
                    new NDList(
                            manager.create(new Shape(2)),
                            manager.create(new Shape(4)),
                            manager.create(new Shape(5)));
            co.forward(parameterStore, input);
            MxNDArray[] inputNDArray = co.getInputNDArray();
            Assert.assertEquals(inputNDArray[0].getShape(), new Shape(2));
            Assert.assertEquals(inputNDArray[3].getShape(), new Shape(4));
            Assert.assertEquals(inputNDArray[4].getShape(), new Shape(5));
            logger.info("Test: Named input");
            NDArray data0 = manager.create(new Shape(5));
            data0.setName("data0");
            NDArray data1 = manager.create(new Shape(4));
            data1.setName("data1");
            NDArray data2 = manager.create(new Shape(2));
            input = new NDList(data2, data1, data0);
            co.forward(parameterStore, input);
            inputNDArray = co.getInputNDArray();
            Assert.assertEquals(inputNDArray[0].getShape(), new Shape(5));
            Assert.assertEquals(inputNDArray[3].getShape(), new Shape(4));
            Assert.assertEquals(inputNDArray[4].getShape(), new Shape(2));
            logger.info("Test: Check the remaining params");
            Assert.assertEquals(inputNDArray[1].getShape(), new Shape(2));
            Assert.assertEquals(inputNDArray[2].getShape(), new Shape(3));
            Assert.assertEquals(inputNDArray[5].getShape(), new Shape(5));
            Assert.assertEquals(inputNDArray[6].getShape(), new Shape(6));
            logger.info("Test: Illegal inputs");
            NDList input2 = new NDList((NDArray) null);
            co.forward(parameterStore, input2);
        }
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
