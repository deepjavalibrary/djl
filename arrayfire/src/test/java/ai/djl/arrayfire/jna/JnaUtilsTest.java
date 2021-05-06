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
package ai.djl.arrayfire.jna;

import ai.djl.Device;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.sun.jna.Pointer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import org.testng.Assert;
import org.testng.annotations.Test;

public class JnaUtilsTest {

    @Test(enabled = false)
    public void testFunctionalities() {
        float[] input = new float[] {1, 2, 3, 4};
        int byteSize = input.length * DataType.FLOAT32.getNumOfBytes();
        ByteBuffer bb = ByteBuffer.allocateDirect(byteSize);
        bb.asFloatBuffer().put(input);
        bb.rewind();
        Shape expected = new Shape(2, 2);
        JnaUtils.setDevice(Device.cpu());
        JnaUtils.printInfo();
        Pointer ptr = JnaUtils.createNDArray(bb, expected, DataType.FLOAT32);
        Shape actual = JnaUtils.getShape(ptr);
        Assert.assertEquals(actual, expected);
        bb = ByteBuffer.allocateDirect(byteSize);
        JnaUtils.getByteBuffer(bb, ptr);
        FloatBuffer fb = bb.asFloatBuffer();
        float[] ret = new float[fb.remaining()];
        fb.get(ret);
        Assert.assertEquals(ret, input);
        Assert.assertEquals(JnaUtils.getRefCounts(ptr), 1);
        JnaUtils.releaseNDArray(ptr);
        Assert.assertEquals(JnaUtils.getRefCounts(ptr), 0);
    }
}
