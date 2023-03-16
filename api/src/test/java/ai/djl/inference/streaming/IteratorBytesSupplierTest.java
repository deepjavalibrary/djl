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

import ai.djl.ndarray.BytesSupplier;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Iterator;
import java.util.stream.Stream;

public class IteratorBytesSupplierTest {

    @Test
    public void testIterate() {
        Iterator<BytesSupplier> iterator =
                Stream.of("a", "b", "c").map(BytesSupplier::wrap).iterator();
        IteratorBytesSupplier supplier = new IteratorBytesSupplier(iterator);

        Assert.assertTrue(supplier.hasNext());
        Assert.assertEquals(supplier.next(), new byte[] {97});
        Assert.assertEquals(supplier.next(), new byte[] {98});
        Assert.assertEquals(supplier.next(), new byte[] {99});
        Assert.assertFalse(supplier.hasNext());
    }

    @Test
    public void testAsBytes() {
        Iterator<BytesSupplier> iterator =
                Stream.of("a", "b", "c").map(BytesSupplier::wrap).iterator();
        IteratorBytesSupplier supplier = new IteratorBytesSupplier(iterator);

        Assert.assertEquals(supplier.getAsBytes(), new byte[] {97, 98, 99});
    }
}
