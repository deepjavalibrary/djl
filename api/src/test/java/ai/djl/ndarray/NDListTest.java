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

package ai.djl.ndarray;

import ai.djl.test.mock.MockNDArray;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDListTest {

    @Test
    public void testAdd() {
        NDList list = new NDList(3);
        Assert.assertEquals(list.size(), 0);
        list.add(new MockNDArray());
        Assert.assertEquals(list.size(), 1);
        NDArray array = new MockNDArray();
        array.setName("test1");
        list.add(array);
        Assert.assertTrue(list.contains("test1"));
        Assert.assertNotNull(list.remove("test1"));
        Assert.assertNotNull(list.singletonOrThrow());
    }
}
