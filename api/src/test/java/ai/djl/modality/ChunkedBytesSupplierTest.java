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
package ai.djl.modality;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class ChunkedBytesSupplierTest {

    @Test
    public void test() throws InterruptedException, IOException {
        ChunkedBytesSupplier supplier = new ChunkedBytesSupplier();
        supplier.appendContent(new byte[] {1, 2}, false);
        supplier.appendContent(new byte[] {3}, true);

        Assert.assertTrue(supplier.isChunked());

        ByteBuffer bb = supplier.toByteBuffer();
        Assert.assertEquals(bb.array(), new byte[] {1, 2, 3});

        supplier = new ChunkedBytesSupplier();
        supplier.appendContent(new byte[0], false);
        supplier.appendContent(new byte[] {1, 2}, true);

        InputStream is = supplier.getChunkedInput();
        byte[] buf = new byte[2];

        Assert.assertEquals(is.read(), 1);
        Assert.assertEquals(is.read(), 2);
        Assert.assertEquals(is.read(), -1);
        Assert.assertEquals(is.read(), -1);
        Assert.assertEquals(is.read(buf), -1);
    }
}
