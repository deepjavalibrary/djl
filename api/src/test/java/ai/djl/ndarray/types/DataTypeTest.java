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

public class DataTypeTest {

    @Test
    public void numpyTest() {
        Assert.assertEquals("|S1", DataType.STRING.asNumpy());
        Assert.assertEquals(DataType.STRING, DataType.fromNumpy("|S1"));

        Assert.expectThrows(IllegalArgumentException.class, DataType.UNKNOWN::asNumpy);
        Assert.expectThrows(IllegalArgumentException.class, () -> DataType.fromNumpy("|i8"));
    }
}
