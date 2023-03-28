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

import java.util.concurrent.atomic.AtomicInteger;

public class PublisherBytesSupplierTest {

    @Test
    public void test() {
        AtomicInteger contentCount = new AtomicInteger();
        PublisherBytesSupplier supplier = new PublisherBytesSupplier();

        // Add to supplier without subscriber
        supplier.appendContent(new byte[] {1}, false);
        Assert.assertEquals(contentCount.get(), 0);

        // Subscribing with data should trigger subscriptions
        supplier.subscribe(
                d -> {
                    if (d == null) {
                        // Do nothing on completion
                        return;
                    }
                    contentCount.getAndIncrement();
                });
        Assert.assertEquals(contentCount.get(), 1);

        // Add to supplier with subscriber
        supplier.appendContent(new byte[] {1}, true);
        Assert.assertEquals(contentCount.get(), 2);
    }
}
