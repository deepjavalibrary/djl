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

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import ai.djl.Device;
import ai.djl.mxnet.jna.LibUtils;
import ai.djl.mxnet.jna.NativeSize;
import ai.djl.mxnet.jna.PointerArray;
import ai.djl.mxnet.test.MockMxnetLibrary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import com.sun.jna.Pointer;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;

@PrepareForTest(LibUtils.class)
public class MxNDArrayTest extends PowerMockTestCase {

    private MockMxnetLibrary library;

    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
    }

    @AfterClass
    public void postProcessing() {
        library.resetFunctions();
    }

    @Test
    public void testNDArrayCreation() {
        // By default the Mock lib will return the following set up
        try (MxNDManager manager = MxNDManager.getSystemManager().newSubManager();
                MxNDArray nd = new MxNDArray(manager, new PointerArray())) {
            Assert.assertEquals(nd.getShape(), new Shape(1, 2, 3));
            Assert.assertEquals(nd.getDevice(), Device.gpu(1));
            Assert.assertEquals(nd.getDataType(), DataType.FLOAT32);
            Assert.assertEquals(nd.getSparseFormat(), SparseFormat.CSR);
        }
    }

    @Test
    public void testSet() {
        final float[][] fa = new float[][] {new float[1]};
        library.setFunction(
                "MXNDArraySyncCopyFromCPU",
                objects -> {
                    int size = ((NativeSize) objects[2]).intValue();
                    fa[0] = ((Pointer) objects[1]).getFloatArray(0, size);
                    return 0;
                });
        try (MxNDManager manager = MxNDManager.getSystemManager().newSubManager();
                NDArray nd = manager.create(new Shape(3))) {
            float[] input = {1.0f, 2.0f, 3.0f};
            nd.set(input);
            float[] fArr = fa[0];
            Assert.assertEquals(input, fArr);
        }
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
    }
}
