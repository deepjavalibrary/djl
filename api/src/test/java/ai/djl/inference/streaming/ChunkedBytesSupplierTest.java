/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.inference.streaming;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.TimeUnit;

public class ChunkedBytesSupplierTest {

    @Test
    public void test() throws InterruptedException, IOException {
        ChunkedBytesSupplier supplier = new ChunkedBytesSupplier();
        Assert.assertNull(supplier.poll());
        Assert.assertThrows(() -> supplier.nextChunk(1, TimeUnit.MICROSECONDS));

        supplier.appendContent(new byte[] {1, 2}, false);
        supplier.appendContent(new byte[] {3}, true);

        ByteBuffer bb = supplier.toByteBuffer();
        Assert.assertEquals(bb.array(), new byte[] {1, 2, 3});

        ChunkedBytesSupplier data = new ChunkedBytesSupplier();
        data.appendContent(new byte[0], false);
        data.appendContent(new byte[] {1, 2}, true);

        Assert.assertTrue(data.hasNext());
        Assert.assertEquals(data.poll().length, 0);
        Assert.assertEquals(data.nextChunk(1, TimeUnit.MILLISECONDS), new byte[] {1, 2});

        Assert.assertFalse(data.hasNext());
    }
}
