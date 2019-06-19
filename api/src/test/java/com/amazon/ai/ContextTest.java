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

package com.amazon.ai;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ContextTest {

    @Test
    public void testContext() {
        Context ctx = new Context("GPU", 3);
        Assert.assertEquals(ctx, Context.gpu(3));
        Assert.assertEquals(Context.cpu(), new Context("CPU", 0));
        Assert.assertEquals(Context.gpu(), new Context("GPU", 0));
    }
}
