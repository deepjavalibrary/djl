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

package org.apache.mxnet.engine;
// CHECKSTYLE:OFF:AvoidStaticImport

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Layout;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.amazon.ai.test.MockMxnetLibrary;
import com.sun.jna.Pointer;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.jna.LibUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.apache.mxnet.jna.NativeSize;
import org.apache.mxnet.jna.PointerArray;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxNDArrayTest extends PowerMockTestCase {

    private MxnetLibrary library;
    private MxNDFactory factory;

    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
        factory = MxNDFactory.SYSTEM_FACTORY;
    }

    @AfterClass
    public void postProcessing() {
        ((MockMxnetLibrary) library).resetFunctions();
    }

    @Test
    public void testNDArrayCreation() {
        // By default the Mock lib will return the following set up
        MxNDArray nd = new MxNDArray(factory, null, null, null, null, new PointerArray());
        Assert.assertEquals(nd.getShape(), new Shape(1, 2, 3));
        Assert.assertEquals(nd.getContext(), Context.gpu(1));
        Assert.assertEquals(nd.getDataType(), DataType.FLOAT32);
        Assert.assertEquals(nd.getSparseFormat(), SparseFormat.CSR);
        Assert.assertEquals(nd.getLayout(), Layout.UNDEFINED);
    }

    @Test
    public void testSet() {
        final float[][] fa = new float[][] {new float[1]};
        ((MockMxnetLibrary) library)
                .setFunction(
                        "MXNDArraySyncCopyFromCPU",
                        objects -> {
                            int size = ((NativeSize) objects[2]).intValue();
                            fa[0] = ((Pointer) objects[1]).getFloatArray(0, size);
                            return 0;
                        });
        MxNDArray nd = factory.create(new DataDesc(new Shape(3)));
        List<Float> input = Arrays.asList(1.0f, 2.0f, 3.0f);
        nd.set(input);
        float[] fArr = fa[0];
        Assert.assertEquals(fArr.length, input.size());
        for (int i = 0; i < fArr.length; i++) {
            Assert.assertEquals(input.get(i), fArr[i]);
        }
    }
}
