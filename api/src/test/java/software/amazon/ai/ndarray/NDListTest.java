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

package software.amazon.ai.ndarray;

import org.testng.Assert;
import org.testng.annotations.Test;
import software.amazon.ai.test.mock.MockNDArray;

public class NDListTest {
    @Test
    public void testAdd() {
        NDList list = new NDList(3);
        Assert.assertEquals(list.size(), 0);
        list.add(new MockNDArray());
        Assert.assertEquals(list.size(), 1);
        list.add("test1", new MockNDArray());
        Assert.assertTrue(list.contains("test1"));
        Assert.assertNotNull(list.remove("test1"));
        Assert.assertNotNull(list.get(0));
        list.addAll(list);
        Assert.assertEquals(list.size(), 2);
        NDArray[] arr = list.toArray();
        Assert.assertEquals(arr.length, 2);
    }
}
