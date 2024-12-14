/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.translate;

import org.testng.Assert;
import org.testng.annotations.Test;

public class BatchifierTest {

    @Test
    public void testBatchifier() {
        Batchifier batchifier = Batchifier.fromString("stack");
        Assert.assertEquals(batchifier, Batchifier.STACK);

        batchifier = Batchifier.fromString("none");
        Assert.assertNull(batchifier);

        batchifier = Batchifier.fromString("padding");
        Assert.assertNotNull(batchifier);
        Assert.assertEquals(batchifier.getClass(), SimplePaddingStackBatchifier.class);

        batchifier = Batchifier.fromString("ai.djl.translate.SimplePaddingStackBatchifier");
        Assert.assertNotNull(batchifier);
        Assert.assertEquals(batchifier.getClass(), SimplePaddingStackBatchifier.class);

        Assert.assertThrows(() -> Batchifier.fromString("invalid"));
    }
}
