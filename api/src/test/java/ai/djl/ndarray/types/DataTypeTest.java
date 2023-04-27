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
package ai.djl.ndarray.types;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteOrder;

public class DataTypeTest {

    @Test
    public void numpyTest() {
        char order = ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN ? '>' : '<';

        Assert.assertEquals(DataType.INT16.asNumpy(), order + "i2");
        Assert.assertEquals(DataType.UINT16.asNumpy(), order + "u2");
        Assert.assertEquals(DataType.UINT32.asNumpy(), order + "u4");
        Assert.assertEquals(DataType.UINT64.asNumpy(), order + "u8");
        Assert.assertEquals(DataType.STRING.asNumpy(), "|S1");
        Assert.assertEquals(DataType.fromNumpy("<i2"), DataType.INT16);
        Assert.assertEquals(DataType.fromNumpy(">u2"), DataType.UINT16);
        Assert.assertEquals(DataType.fromNumpy("=u4"), DataType.UINT32);
        Assert.assertEquals(DataType.fromNumpy(">u8"), DataType.UINT64);
        Assert.assertEquals(DataType.fromNumpy("|S1"), DataType.STRING);

        Assert.expectThrows(IllegalArgumentException.class, DataType.BFLOAT16::asNumpy);
        Assert.expectThrows(IllegalArgumentException.class, DataType.UNKNOWN::asNumpy);
        Assert.expectThrows(IllegalArgumentException.class, () -> DataType.fromNumpy("|i8"));
    }
}
